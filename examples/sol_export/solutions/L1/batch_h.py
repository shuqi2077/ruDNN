# L1 batch H: 058-067.

# problem: 058
def run(topk_idx):
    flat = _aligned(topk_idx).view(-1)
    n = flat.numel()
    dev = flat.device
    chunks = (n + 1023) // 1024
    hist = torch.empty((chunks, 256), dtype=torch.int32, device=dev)
    _launch('sort_hist', [flat, hist], (chunks, 1, 1), {'I': [n]})
    base = torch.empty_like(hist)
    offsets = torch.empty(257, dtype=torch.int32, device=dev)
    _launch('sort_scan', [hist, base, offsets], (1, 1, 1), {'I': [chunks]})
    out = torch.empty(n, dtype=torch.int32, device=dev)
    _launch('sort_scatter', [flat, base, out], (chunks, 1, 1), {'I': [n]})
    return out, offsets

# problem: 060
def run(grad_output, query, key, attn_weights, scaling):
    b, h, lq, lk = grad_output.shape
    d = query.shape[3]
    dev = query.device
    ldp = (lk + 7) // 8 * 8
    gs = torch.empty((b * h, lq, ldp), dtype=torch.bfloat16, device=dev)
    _launch('softmax_bwd_scaled_bf16', [_aligned(grad_output), _aligned(attn_weights), gs], (b * h * lq, 1, 1),
            {'I': [lk, ldp], 'f': [float(scaling)]})
    q3, k3 = _aligned(query).view(b * h, lq, d), _aligned(key).view(b * h, lk, d)
    grad_q = torch.empty((b, h, lq, d), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_01_00000', _opb(gs), _opb(k3), grad_q, lq, d, ldp, batches=b * h, ld=d, o_s1=lq * d)
    grad_k = torch.empty((b, h, lk, d), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_11_00000', _opb(gs), _opb(q3), grad_k, lk, d, lq, batches=b * h, ld=d, o_s1=lk * d)
    return grad_q, grad_k

# problem: 063
def run(attn_output, o_proj_weight):
    b, h, s, d = attn_output.shape
    n = o_proj_weight.shape[0]
    a = _aligned(attn_output)
    w = _aligned(o_proj_weight)
    out = torch.empty((b, s, n), dtype=a.dtype, device=a.device)
    _gemm('gemm_bb_256_00_00000', _Op(a, [d, b * h, s], [s * d * 2, d * 2], y_mul=h, y_kstep=1),
          _Op(w, [d, h, n], [d * 2, h * d * 2], y_mul=0, y_kstep=1), out, s, n, d, batches=b, k_groups=h, ld=n, o_s1=s * n)
    return out

# problem: 064
def run(compressed_kv, kv_a_layernorm_weight, kv_b_proj_weight, eps):
    b, s, r = compressed_kv.shape
    dev = compressed_kv.device
    normed = _norm('norm_bbb_001000', _aligned(compressed_kv), r, eps, weight=_aligned(kv_a_layernorm_weight))
    w = _aligned(kv_b_proj_weight)
    a = _Op(normed, [r, 1, b * s], [r * 2, r * 2], z_div=128, z_mul=s)
    outs = []
    for half in (0, 1):
        wv = w[half * 128:]
        o = torch.empty((b, 128, s, 128), dtype=torch.bfloat16, device=dev)
        _gemm('gemm_bb_128_00_00000', a, _Op(wv, [r, 128, 128], [256 * r * 2, r * 2], y_mod=128, y_mul=1), o,
              s, 128, r, batches=b * 128, ld=128, o_s1=s * 128)
        outs.append(o)
    return outs[0], outs[1]

# problem: 065
def run(x, weight, bias):
    b, c, t = x.shape
    dev = x.device
    xp = _aligned(x)
    if t % 4:
        tp = (t + 3) // 4 * 4
        padded = torch.empty((b, c, tp), dtype=torch.float32, device=dev)
        _launch('pad_rows_f32', [xp, padded], _blocks(padded.numel(), 256), {'I': [padded.numel(), t, tp]})
        xp = padded
    n = weight.shape[0]
    out = torch.empty((b, t, n), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_128_10_10000', _opb(xp), _op(_aligned(weight).view(n, c)), out, t, n, c, batches=b, ld=n, o_s1=t * n,
          bias=_aligned(bias))
    stats = out.transpose(1, 2)
    return stats[:, :n // 2], stats[:, n // 2:]

# problem: 067
def run(hidden_states, cos, sin, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    xa = _tf32x3(hidden_states.reshape(b * s, hid), True)
    proj = []
    for w in (q_proj_weight, k_proj_weight, v_proj_weight):
        o = torch.empty((b * s, w.shape[0]), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_00000', xa, _tf32x3(w, False), o, b * s, w.shape[0], hid, k_groups=3)
        proj.append(o)
    q, k, v = proj
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    c, sn = _aligned(cos), _aligned(sin)
    _hnr('hnr_fff_128_0100', q, qr, b * s * 32, 32, 1, cos=c, sin=sn)
    _hnr('hnr_fff_128_0100', k, kr, b * s * 8, 8, 1, cos=c, sin=sn)
    o = torch.empty((b, s, 32, 128), dtype=torch.float32, device=dev)
    _flash('flash_tf_128_64_2_010000', qr.view(b, s, 32, 128), kr.view(b, s, 8, 128), v.view(b, s, 8, 128), o,
           scale=128 ** -0.5)
    out = torch.empty((b * s, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_00000', o.view(b * s, 32 * 128), o_proj_weight, out, b * s, hid, 32 * 128)
    return out.view(b, s, hid)
