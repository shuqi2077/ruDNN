import base64
import ctypes
import json
import lzma
import struct

import torch

_kernels = {}
_driver = None


def _check(code):
    if code:
        raise RuntimeError('CUDA Driver error ' + str(code))


def _cuda():
    global _driver
    if _driver is None:
        driver = ctypes.CDLL('libcuda.so.1')
        signatures = {
            'cuCtxGetCurrent': [ctypes.POINTER(ctypes.c_void_p)],
            'cuModuleLoadData': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            'cuModuleGetFunction': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
            'cuFuncSetAttribute': [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
            'cuTensorMapEncodeTiled': [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)] + [ctypes.c_int] * 4,
            'cuLaunchKernel': [ctypes.c_void_p] + [ctypes.c_uint] * 7
                + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)],
        }
        for name, args in signatures.items():
            fn = getattr(driver, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
        _driver = driver
    return _driver


def _tensor_map(driver, tensor, rows, columns, volume=False):
    storage = ctypes.create_string_buffer(128 + 63)
    address = (ctypes.addressof(storage) + 63) & ~63
    rank = 3 if volume else 2
    row_bytes = tensor.stride(0) * tensor.element_size()
    shape = (ctypes.c_uint64 * rank)(*([tensor.shape[1], 1, tensor.shape[0]] if volume else list(reversed(tensor.shape))))
    strides = (ctypes.c_uint64 * (rank - 1))(*([row_bytes] * (rank - 1)))
    box = (ctypes.c_uint32 * rank)(*([columns, 1, rows] if volume else [columns, rows]))
    element_strides = (ctypes.c_uint32 * rank)(*([1] * rank))
    dtype = 6 if tensor.dtype == torch.float16 else 9
    _check(driver.cuTensorMapEncodeTiled(address, dtype, rank, tensor.data_ptr(),
            shape, strides, box, element_strides, 0, 3, 0, 0))
    return storage, address


def _launch(name, buffers, grid, scalars=None, maps=None):
    driver = _cuda()
    context = ctypes.c_void_p()
    _check(driver.cuCtxGetCurrent(ctypes.byref(context)))
    key = (context.value, name)
    if key not in _kernels:
        spec = _BUNDLE[name]
        source = spec['ptx'].encode()
        ptx = ctypes.create_string_buffer(source)
        module = ctypes.c_void_p()
        _check(driver.cuModuleLoadData(ctypes.byref(module), ptx))
        function = ctypes.c_void_p()
        _check(driver.cuModuleGetFunction(ctypes.byref(function), module, name.encode()))
        if spec['shared'] > 49152:
            _check(driver.cuFuncSetAttribute(function, 8, spec['shared']))
        _kernels[key] = (module, function, spec)
    module, function, spec = _kernels[key]
    scalars = scalars or {}
    info = bytearray()
    for kind, count in spec['scalars']:
        values = scalars[kind]
        if len(values) != count:
            raise ValueError('scalar ABI mismatch: ' + name)
        info.extend(struct.pack('<' + kind * count, *values))
        info.extend(b'\0' * (-len(info) % 8))
    if len(buffers) != len(spec['widths']):
        raise ValueError('buffer ABI mismatch: ' + name)
    sizes = [tensor.numel() // width for tensor, width in zip(buffers, spec['widths'])]
    info.extend(struct.pack('<' + 'I' * (2 * len(sizes)), *(sizes + sizes)))
    info.extend(b'\0' * (-len(info) % 8))
    packed = ctypes.create_string_buffer(bytes(info))
    pointers = [ctypes.c_uint64(tensor.data_ptr()) for tensor in buffers]
    if maps is None:
        maps = [_tensor_map(driver, buffers[index], rows, columns) for index, rows, columns in spec['maps']]
    parameters = (ctypes.c_void_p * (len(maps) + len(buffers) + 1))(
        *[address for storage, address in maps], *[ctypes.addressof(p) for p in pointers], ctypes.addressof(packed))
    stream = torch.cuda.current_stream(buffers[0].device).cuda_stream
    _check(driver.cuLaunchKernel(function, *grid, *spec['block'], spec['shared'], stream, parameters, None))


def _empty(shape, like, dtype=None):
    return torch.empty(shape, dtype=dtype or like.dtype, device=like.device)


def _norm(x, residual, weight):
    with torch.cuda.device(x.device):
        out = _empty(x.shape, x)
        if x.shape[0]:
            rows_per_block = 4 if x.shape[1] <= 2048 else 2
            _launch('rms_' + str(x.shape[1]) + '_' + str(int(residual is not None)),
                    [x, residual if residual is not None else x, weight, out],
                    ((x.shape[0] + rows_per_block - 1) // rows_per_block, 1, 1))
        return out


def _gemm_plan(m, n, k, sms, sm100):
    policy = _GEMM_POLICY
    blackwell = sm100 and m >= policy['blackwell_min_m']
    if blackwell:
        wide = m >= policy['wide_min_m'] and n >= 256 and ((m + 127) // 128) * ((n + 255) // 256) >= max(1, sms)
        bm, bn = 128, 256 if wide else 128
    else:
        tile = next((t for t in policy['small_tiles'] if m <= t[0]), None)
        bm, bn = (tile[1], tile[2]) if tile else ((128, 128) if n >= 2 * k else (128, 64))
    tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
    minimum_k = policy['sparse_k'] if tiles * 4 < sms else policy['minimum_k']
    partitions = 1
    waves = policy['blackwell_split_waves'] if blackwell else policy['split_waves']
    while partitions < policy['max_splits'] and tiles * partitions < sms * waves \
            and k % (64 * partitions * 2) == 0 and k // (partitions * 2) >= minimum_k:
        partitions *= 2
    return blackwell, bm, bn, partitions, tiles


def _gemm(a, b):
    with torch.cuda.device(a.device):
        m, k = a.shape
        n = b.shape[0]
        out = _empty((m, n), a)
        if m == 0:
            return out
        if m == 1:
            _launch('gemv_' + str(m), [a, b, out], ((n + 3) // 4, 1, 1),
                    {'I': [k // 8, n, k // 8, k // 8, n]})
            return out
        device = torch.cuda.get_device_properties(a.device)
        sms = device.multi_processor_count
        blackwell, bm, bn, partitions, tiles = _gemm_plan(m, n, k, sms, (device.major, device.minor) == (10, 0))
        partial = out if partitions == 1 else _empty((partitions, m, n), a, torch.float32)
        part_k = k // partitions
        if blackwell:
            workers = min(tiles, max(1, sms))
            if partitions == 1:
                big = 1 << 30
                coords = [big, 0, big, 1, 0, 0, 0]
                maps = [_tensor_map(_cuda(), a, 128, 64, True), _tensor_map(_cuda(), b, bn, 64, True)]
                _launch('blackwell_ws_' + str(bn), [a, b, out, out, out, out, out, out],
                        (workers, 1, 1), {'I': [m, n, k, 1, 1, n // 8, 1, 0, 0, big, 0, 0, 0] + coords + coords,
                                         'f': [1.0, 0.0]}, maps)
            else:
                _launch('blackwell_128_' + str(bn) + '_' + str(partitions), [a, b, partial, partial],
                        (workers, 1, partitions), {'I': [m, n, k, part_k, workers, 0, 0], 'f': [1.0, 0.0]})
        else:
            _launch('mma_' + str(bm) + '_' + str(bn) + '_' + str(partitions), [a, b, partial, partial],
                    ((m + bm - 1) // bm, (n + bn - 1) // bn, partitions),
                    {'I': [m, n, k, part_k, 0, 0], 'f': [1.0, 0.0]})
        if partitions != 1:
            _launch('split_reduce_' + str(partitions), [partial, out], ((m * n + 255) // 256, 1, 1), {'I': [m * n]})
        return out


def _attention(label, q, k, v, qp, kp, qo, kv, pages, scale):
    with torch.cuda.device(q.device):
        rows, heads, dim = q.shape
        out = _empty(q.shape, q)
        lse = _empty((rows, heads), q, torch.float32)
        if rows:
            total_kv = pages.numel() if pages is not None else k.shape[0]
            average = (total_kv + kv.numel() - 2) // max(1, kv.numel() - 1)
            splits = min(32, max(1, (average + 255) // 256)) if rows * heads < 4096 else 1
            warps = 4 if average // splits >= 64 else 1
            finalize = splits == 1
            partials = lse if finalize else _empty((rows, heads, splits, dim + 2), q, torch.float32)
            _launch(label + '_w' + str(warps) + '_' + str(int(finalize)),
                    [q, k, v, qp if qp is not None else q, kp if kp is not None else k,
                     qo if qo is not None else kv, kv, pages if pages is not None else kv, partials, out, lse],
                    (rows, heads, splits), {'f': [float(scale)], 'I': [splits]})
            if not finalize:
                _launch('merge_lse_' + str(dim), [partials, out, lse], (rows * heads, 1, 1), {'I': [splits]})
        return out, lse


def _moe(logits, bias, x, xs, w1, s1, w2, s2, offset, factor):
    with torch.cuda.device(x.device):
        tokens = x.shape[0]
        out = _empty((tokens, 7168), x, torch.bfloat16)
        if tokens == 0:
            return out
        weights = _empty((tokens, 32), x, torch.float32)
        selected = _empty((tokens, 32), x, torch.int32)
        positions = _empty((tokens, 32), x, torch.int32)
        rows = _empty((32, tokens), x, torch.int32)
        counts = _empty((32,), x, torch.int32)
        offsets = _empty((32,), x, torch.int32)
        capacity = 8 * tokens + 32 * 31
        activated = _empty((capacity, 2048), x, torch.float32)
        second = _empty((capacity, 7168), x, torch.float32)
        _launch('moe_route', [logits, bias, weights, selected], ((tokens + 3) // 4, 1, 1), {'i': [int(offset)], 'f': [float(factor)]})
        _launch('moe_compact', [selected, rows, counts, positions], (32, 1, 1), {'I': [tokens]})
        _launch('moe_offsets', [counts, offsets], (1, 1, 1))
        hardware = torch.cuda.get_device_properties(x.device)
        workers = hardware.multi_processor_count * 2
        swiglu = 'moe_swiglu_sm100' if (hardware.major, hardware.minor) == (10, 0) else 'moe_swiglu'
        _launch(swiglu, [x, xs, w1, s1, rows, counts, offsets, activated],
                (workers, 1, 1), {'I': [tokens]})
        _launch('moe_project2', [x, activated, xs, w2, s2, rows, counts, offsets, second],
                (workers, 1, 1), {'I': [tokens]})
        _launch('moe_combine', [second, positions, offsets, weights, out], ((tokens * 7168 + 255) // 256, 1, 1))
        return out
