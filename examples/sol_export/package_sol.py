"""Package SOL-ExecBench L1 / L2 / Quant collection archives from RUDA kernels.

usage: package_sol.py <ptx_dir> <out_dir> <suite>=<parquet> [...]

For every problem with a RUDA solution section (solutions/<suite>/*.py,
`# problem: NNN` markers) the entry is: sol_launcher.py + a PTX bundle of the
kernels the section launches + the section code. Every problem must have a
RUDA solution; reference implementations are never packaged.
Writes <out_dir>/ruda-<suite>-v2.zip and <out_dir>/<suite>-manifest.json.
"""
import argparse
import ast
import base64
import json
import lzma
import re
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
UPLOAD_LIMIT = 10_000_000
EXTRACT_LIMIT = 100_000_000
LITERAL = re.compile(r"'([A-Za-z0-9_]+)'")
CALL = re.compile(r"\b(_[A-Za-z0-9_]+)\(")


def encode_bundle(bundle):
    source = json.dumps(bundle, separators=(',', ':')).encode()
    assert all(byte < 128 for byte in source)
    data = bytearray(b'RDN1')
    previous, offset = [0, 0], 0
    for match in re.finditer(rb'(%r|\bL)(0|[1-9][0-9]*)(?![0-9])', source):
        data.extend(source[offset:match.start()])
        kind = int(match[1] == b'%r')
        value = int(match[2])
        delta = value - previous[kind]
        previous[kind] = value
        encoded = 2 * delta if delta >= 0 else -2 * delta - 1
        data.append(254 + kind)
        while encoded >= 128:
            data.append((encoded & 127) | 128)
            encoded >>= 7
        data.append(encoded)
        offset = match.end()
    data.extend(source[offset:])
    return base64.b85encode(lzma.compress(bytes(data), preset=9)).decode()


def library_index(source):
    """function name -> (string literals, called helper names) for sol_lib.py."""
    index = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            body = ast.get_source_segment(source, node)
            index[node.name] = (set(LITERAL.findall(body)), set(CALL.findall(body)))
    return index


def kernels_used(section, library, ptx):
    names = set(LITERAL.findall(section))
    pending, seen = list(CALL.findall(section)), set()
    while pending:
        call = pending.pop()
        if call in seen or call not in library:
            continue
        seen.add(call)
        literals, calls = library[call]
        names |= literals
        pending += list(calls)
    generic = re.compile(r'^(?:gemm|flash|norm|softmax|rope|hnr|im2col|gather|colpart|colfin|rowsum|cast|permute|bin|scatterseq|un|adaln)_[hbfteg]+(?:_[0-9]+[g]?)?(?:_[0-9]+)*$')
    missing = sorted(n for n in names if generic.fullmatch(n) and n not in ptx)
    assert not missing, f'RUDA dependency kernels not exported: {missing}'
    return sorted(n for n in names if n in ptx)


def load_ptx(ptx_dir):
    kernels = {}
    for line in (ptx_dir / 'manifest.tsv').read_text().splitlines():
        if not line.strip():
            continue
        name, block, shared, widths, scalars = line.split('\t')[:5]
        source = (ptx_dir / (name + '.ptx')).read_text()
        assert '.target sm_100' in source, name
        kernels[name] = {'block': [int(v) for v in block.split(',')], 'shared': int(shared),
                         'widths': [int(v) for v in widths.split(',')],
                         'scalars': [[s.split(':')[0], int(s.split(':')[1])] for s in scalars.split(',') if s],
                         'ptx': source}
    return kernels


def load_solutions(suite):
    sections = {}
    for path in sorted((HERE / 'solutions' / suite).glob('*.py')):
        current = None
        for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
            marker = re.match(r'# problem: (\d{3})\s*$', line)
            if marker:
                current = marker.group(1)
                assert current not in sections, f'{suite}/{current} defined twice'
                sections[current] = ''
            elif current is not None:
                sections[current] += line
    return sections


def run_arity(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'run':
            return len(node.args.posonlyargs + node.args.args)
    raise ValueError('no top-level run()')


def build(ptx_dir, out_dir, suites, version):
    lib = (HERE / 'sol_lib.py').read_text(encoding='utf-8')
    library = library_index(lib)
    launcher = (HERE / 'sol_launcher.py').read_text(encoding='utf-8') + '\n' + lib
    ptx = load_ptx(ptx_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for suite, parquet in suites:
        rows = pq.read_table(parquet).to_pylist()
        solutions = load_solutions(suite)
        known = {row['name'][:3] for row in rows}
        assert set(solutions) <= known, f'{suite}: solutions for unknown problems {set(solutions) - known}'
        assert set(solutions) == known, f'{suite}: RUDA solutions missing: {sorted(known - set(solutions))}'
        archive_path = out_dir / f'ruda-{suite.lower()}-{version}.zip'
        manifest, extracted = [], 0
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for problem in rows:
                name, number = problem['name'], problem['name'][:3]
                inputs = len(json.loads(problem['inputs']))
                section = solutions[number]
                kernels = kernels_used(section, library, ptx)
                launched = set(re.findall(r"_launch\(\s*'([A-Za-z0-9_]+)'", section))
                missing = [k for k in launched if k not in ptx]
                assert not missing, f'{suite}/{name}: kernels not exported: {missing}'
                assert kernels, f'{suite}/{name}: no RUDA kernels in dependency closure'
                bundle = encode_bundle({k: ptx[k] for k in kernels})
                code = (launcher + '\n_BUNDLE = _decode_bundle(' + repr(bundle) + ')\n\n'
                        + section.strip() + '\n')
                kind = 'ruda'
                assert run_arity(code) == inputs, f'{suite}/{name}: run() arity differs from the definition ({inputs})'
                compile(code, name + '/submission.py', 'exec')
                body = code.encode()
                extracted += len(body)
                archive.writestr(name + '/submission.py', body)
                manifest.append({'problem': name, 'kind': kind, 'kernels': kernels})
        size = archive_path.stat().st_size
        assert size <= UPLOAD_LIMIT and extracted <= EXTRACT_LIMIT, (suite, size, extracted)
        (out_dir / f'{suite}-manifest.json').write_text(json.dumps(manifest, indent=1))
        summary[suite] = {'archive': archive_path.name, 'problems': len(rows), 'zip_bytes': size,
                          'ruda': sum(m['kind'] == 'ruda' for m in manifest),
                          'reference': sum(m['kind'] == 'reference' for m in manifest)}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ptx_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('suites', nargs='+')
    parser.add_argument('--version', default='v2')
    options = parser.parse_args()
    pairs = [(item.split('=', 1)[0], Path(item.split('=', 1)[1])) for item in options.suites]
    build(options.ptx_dir, options.out_dir, pairs, options.version)
