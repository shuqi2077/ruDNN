# L1 batch D: attention / MLP-backward solutions on the generic RUDA kernels.

# problem: 015
def run(hidden_states, cos, sin, attention_mask, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        q_norm_weight, k_norm_weight, rms_norm_eps, scaling):
    b, s, h = hidden_states.shape
    dev = hidden_states.device
    x = _aligned(hidden_states)
    q = _linear('gemm_bb_256_00_00000', x, q_proj_weight)
    k = _linear('gemm_bb_256_00_00000', x, k_proj_weight)
    v = _linear('gemm_bb_256_00_00000', x, v_proj_weight)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    c, sn = _aligned(cos), _aligned(sin)
    _hnr('hnr_bbb_128_1101', q, qr, b * s * 32, 32, 1, rms_norm_eps, weight=_aligned(q_norm_weight), cos=c, sin=sn)
    _hnr('hnr_bbb_128_1101', k, kr, b * s * 8, 8, 1, rms_norm_eps, weight=_aligned(k_norm_weight), cos=c, sin=sn)
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, 32, 128), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_128_128_2_000010', qr.view(b, s, 32, 128), kr.view(b, s, 8, 128), v.view(b, s, 8, 128), o,
           mask=mask, scale=scaling, mask_strides=(s * s, 0, s))
    return _linear('gemm_bb_256_00_00000', o.view(b, s, 32 * 128), o_proj_weight)

# problem: 017
def run(grad_output, x, gate_weight, up_weight, down_weight, gate, gate_sigmoid, gate_silu, up, intermediate):
    dev = grad_output.device
    t, hidden = grad_output.shape
    inter = gate_weight.shape[0]
    g2, x2 = _aligned(grad_output), _aligned(x)
    grad_down = torch.empty((hidden, inter), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(g2), _op(_aligned(intermediate)), grad_down, hidden, inter, t)
    gi = torch.empty((t, inter), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_01_00000', _op(g2), _op(_aligned(down_weight)), gi, t, inter, hidden)
    gu = torch.empty((t, inter), dtype=torch.bfloat16, device=dev)
    gg = torch.empty_like(gu)
    _launch('l1_017_mid', [gi, _aligned(up), _aligned(gate_silu), _aligned(gate), _aligned(gate_sigmoid), gu, gg],
            _blocks(t * inter, 256), {'I': [t * inter]})
    grad_up_w = torch.empty((inter, hidden), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(gu), _op(x2), grad_up_w, inter, hidden, t)
    grad_gate_w = torch.empty_like(grad_up_w)
    _gemm('gemm_bb_256_11_00000', _op(gg), _op(x2), grad_gate_w, inter, hidden, t)
    gx = torch.empty((t, hidden), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_01_00000', _op(gg), _op(_aligned(gate_weight)), gx, t, hidden, inter)
    _gemm('gemm_bf_256_01_00100', _op(gu), _op(_aligned(up_weight)), gx, t, hidden, inter, residual=gx)
    out = torch.empty((t, hidden), dtype=torch.bfloat16, device=dev)
    _launch('cast_fb', [gx, out], _blocks(gx.numel() // 8, 256), {'I': [gx.numel() // 8]})
    return out, grad_gate_w, grad_up_w, grad_down

# problem: 018
def run(query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight,
        inv_freq, rms_norm_eps):
    b, hq, s, d = query.shape
    hk = key.shape[1]
    q, k = _aligned(query), _aligned(key)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    pos, inv = _aligned(position_ids), _aligned(inv_freq)
    _hnr('hnr_bbb_128_2111', q, qr, b * hq * s, hq * s, s, rms_norm_eps, weight=_aligned(q_norm_weight),
         positions=pos, inv_freq=inv)
    _hnr('hnr_bbb_128_2111', k, kr, b * hk * s, hk * s, s, rms_norm_eps, weight=_aligned(k_norm_weight),
         positions=pos, inv_freq=inv)
    length = key_cache.shape[2]
    cp = _aligned(cache_position)
    n = kr.numel() // 8
    _launch('scatterseq_b', [kr, cp, key_cache], _blocks(n, 256), {'I': [n, s, length, d // 8]})
    _launch('scatterseq_b', [_aligned(value), cp, value_cache], _blocks(n, 256), {'I': [n, s, length, d // 8]})
    return qr, kr, key_cache, value_cache

# problem: 020
def run(hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
    n, c = hidden.shape
    parts, offset = [], 0
    for t, h, w in grid_thw.tolist():
        ti = torch.arange(t).view(t, 1, 1, 1, 1)
        hm = torch.arange(h // 2).view(1, -1, 1, 1, 1)
        wm = torch.arange(w // 2).view(1, 1, -1, 1, 1)
        i = torch.arange(2).view(1, 1, 1, 2, 1)
        j = torch.arange(2).view(1, 1, 1, 1, 2)
        parts.append((offset + (ti * h + hm * 2 + i) * w + wm * 2 + j).reshape(-1))
        offset += t * h * w
    index = torch.cat(parts).to(hidden.device)
    normed = _norm('norm_bbb_101100', _aligned(hidden), c, eps, weight=_aligned(ln_weight), bias=_aligned(ln_bias))
    shuffled = torch.empty((n, c), dtype=hidden.dtype, device=hidden.device)
    _launch('gather_b', [normed, index, shuffled], _blocks(n * c // 8, 256), {'I': [n * c // 8, c // 8]})
    h1 = _linear('gemm_bb_256_00_12001', shuffled.view(n // 4, 4 * c), fc1_weight, bias=_aligned(fc1_bias))
    return _linear('gemm_bb_256_00_10000', h1, fc2_weight, bias=_aligned(fc2_bias))

# problem: 022
def run(grad_output, hidden_state, fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc1_output, gelu_output):
    dev = grad_output.device
    s, h = grad_output.shape
    inter = fc1_weight.shape[0]
    go = _aligned(grad_output)
    grad_fc2_bias = torch.empty(h, dtype=torch.float32, device=dev)
    _colsum('colpart_f', 'colfin_f', go, s, h, grad_fc2_bias)
    grad_fc2_weight = torch.empty((h, inter), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_11_00000', go, _aligned(gelu_output), grad_fc2_weight, h, inter, s)
    g1 = torch.empty((s, inter), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_01_00600', go, _aligned(fc2_weight), g1, s, inter, h, residual=_aligned(fc1_output))
    grad_fc1_bias = torch.empty(inter, dtype=torch.float32, device=dev)
    _colsum('colpart_f', 'colfin_f', g1, s, inter, grad_fc1_bias)
    grad_fc1_weight = torch.empty((inter, h), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_11_00000', g1, _aligned(hidden_state), grad_fc1_weight, inter, h, s)
    grad_hidden = torch.empty((s, h), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_01_00000', g1, _aligned(fc1_weight), grad_hidden, s, h, inter)
    return grad_hidden, grad_fc1_weight, grad_fc1_bias, grad_fc2_weight, grad_fc2_bias
