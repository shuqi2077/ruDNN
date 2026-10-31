import argparse
import ast
import base64
import json
import lzma
from pathlib import Path
import zipfile

import pyarrow.parquet as pq


def build(ptx_dir, dataset, output):
    manifest = {}
    for line in (ptx_dir / 'manifest.tsv').read_text().splitlines():
        name, block, shared, widths, scalars, maps = line.split('\t')
        source = (ptx_dir / (name + '.ptx')).read_bytes()
        assert b'.target sm_100\n' in source or b'.target sm_100a\n' in source
        manifest[name] = {'block': list(map(int, block.split(','))), 'shared': int(shared),
                          'widths': list(map(int, widths.split(','))),
                          'scalars': [(s.split(':')[0], int(s.split(':')[1])) for s in scalars.split(',') if s],
                          'maps': [list(map(int, s.split(':'))) for s in maps.split(',') if s],
                          'ptx': source.decode()}
    launcher = Path(__file__).with_name('launcher.py').read_text()
    policy = json.loads((ptx_dir / 'gemm_policy.json').read_text())
    rows = pq.read_table(dataset).to_pylist()
    assert len(rows) == 26 and len({r['name'] for r in rows}) == 26
    total = 0
    bundles = {}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for problem in rows:
            name = problem['name']
            number = int(name[:3])
            signature = next(n for n in ast.parse(problem['reference']).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
            args = [a.arg for a in signature.args.args]
            if number <= 3 or number >= 21:
                width = int(name.rsplit('h', 1)[1])
                keys = [f'rms_{width}_{int(number <= 3)}']
                expr = '_norm(hidden_states, ' + ('residual' if number <= 3 else 'None') + ', weight)'
            elif number <= 11:
                keys = [k for k in manifest if k.startswith(('mma_', 'blackwell_', 'gemv_', 'split_reduce_'))]
                expr = '_gemm(A, B)'
            elif number <= 19:
                label = {12: 'gqa_decode4',13: 'gqa_decode8',14: 'gqa_paged4',15: 'gqa_paged8',
                         16: 'gqa_ragged4',17: 'gqa_ragged8',18: 'mla_decode',19: 'mla_prefill'}[number]
                keys = [f'{label}_w{warps}_{finalize}' for warps in (1, 4) for finalize in (0, 1)]
                keys += ['merge_lse_' + ('512' if number >= 18 else '128')]
                if number < 18:
                    paged = number < 16
                    expr = f"_attention('{label}', q, " + ('k_cache, v_cache' if paged else 'k, v')
                    expr += ', None, None, ' + ('None' if number < 14 else 'qo_indptr') + ', kv_indptr, '
                    expr += ('kv_indices' if paged else 'None') + ', sm_scale)'
                else:
                    expr = f"_attention('{label}', q_nope, ckv_cache, ckv_cache, q_pe, kpe_cache, "
                    expr += ('None' if number == 18 else 'qo_indptr') + ', kv_indptr, kv_indices, sm_scale)'
            else:
                keys = [k for k in manifest if k.startswith('moe_')]
                expr = '_moe(' + ', '.join(args) + ')'
            selected = {key: manifest[key] for key in keys}
            bundle_key = tuple(keys)
            if bundle_key not in bundles:
                bundles[bundle_key] = base64.b85encode(lzma.compress(json.dumps(selected).encode(), preset=6)).decode()
            code = launcher + '\n_BUNDLE = json.loads(lzma.decompress(base64.b85decode(' + repr(bundles[bundle_key]) + ')))\n_GEMM_POLICY = ' + repr(policy) + '\n\ndef run(' + ', '.join(args) + '):\n'
            if number == 18:
                code += '    output, lse = ' + expr + '\n    return {"output": output, "lse": lse}\n'
            else:
                code += '    return ' + expr + '\n'
            compile(code, name + '/submission.py', 'exec')
            encoded = code.encode()
            total += len(encoded)
            archive.writestr(name + '/submission.py', encoded)
    assert total <= 100_000_000
    assert output.stat().st_size <= 10_000_000
    print(json.dumps({'problems': len(rows), 'python_bytes': total, 'zip_bytes': output.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ptx_dir', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.ptx_dir, args.dataset, args.output)
