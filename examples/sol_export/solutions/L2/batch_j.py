# L2 batch J: 007 027 (bf16 GQA attention blocks on tcgen05 flash attention).

# problem: 007
def run(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, o_weight, cos, sin, attention_mask):
    b, s, _ = hidden_states.shape
    dev = hidden_states.device
    q = _linear('gemm_bb_256_00_10000', hidden_states, q_weight, bias=_aligned(q_bias))
    k = _linear('gemm_bb_256_00_10000', hidden_states, k_weight, bias=_aligned(k_bias))
    v = _linear('gemm_bb_256_00_10000', hidden_states, v_weight, bias=_aligned(v_bias))
    tabs = []
    for t in (cos, sin):
        o = torch.empty((b, s, 128), dtype=torch.bfloat16, device=dev)
        _launch('mrope_b', [_aligned(t), o], _blocks(o.numel(), 256), {'I': [o.numel(), s, t.shape[2], b]})
        tabs.append(o)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_bbb_128_0101', q, qr, b * s * 28, 28, 1, cos=tabs[0], sin=tabs[1])
    _hnr('hnr_bbb_128_0101', k, kr, b * s * 4, 4, 1, cos=tabs[0], sin=tabs[1])
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, 28, 128), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_128_128_2_000010', qr.view(b, s, 28, 128), kr.view(b, s, 4, 128), v.view(b, s, 4, 128), o,
           mask=mask, scale=128 ** -0.5, mask_strides=(s * s, 0, s))
    return _linear('gemm_bb_256_00_00000', o.view(b, s, 28 * 128), o_weight)

# problem: 027
def run(hidden_states, position_ids, attention_mask, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps, attention_factor, scaling):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    q = _linear('gemm_bb_256_00_00000', hidden_states, q_proj_weight)
    k = _linear('gemm_bb_256_00_00000', hidden_states, k_proj_weight)
    v = _linear('gemm_bb_256_00_00000', hidden_states, v_proj_weight)
    qn = _norm('norm_bbb_001000', q, q.shape[-1], rms_norm_eps, weight=_aligned(q_norm_weight))
    kn = _norm('norm_bbb_001000', k, k.shape[-1], rms_norm_eps, weight=_aligned(k_norm_weight))
    cos = torch.empty((b * s, 128), dtype=torch.bfloat16, device=dev)
    sin = torch.empty_like(cos)
    _launch('rope_table_b', [_aligned(position_ids), _aligned(inv_freq), cos, sin], _blocks(cos.numel(), 256),
            {'I': [cos.numel(), 128], 'f': [float(attention_factor)]})
    qr, kr = torch.empty_like(qn), torch.empty_like(kn)
    _hnr('hnr_bbb_128_0101', qn, qr, b * s * 40, 40, 1, cos=cos, sin=sin)
    _hnr('hnr_bbb_128_0101', kn, kr, b * s * 8, 8, 1, cos=cos, sin=sin)
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, 40, 128), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_128_128_2_000010', qr.view(b, s, 40, 128), kr.view(b, s, 8, 128), v.view(b, s, 8, 128), o,
           mask=mask, scale=float(scaling), mask_strides=(s * s, 0, s))
    return _linear('gemm_bb_256_00_00000', o.view(b, s, hid), o_proj_weight)

# problem: 039
def run(hidden_states, position_ids, attention_mask, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        q_norm_weight, k_norm_weight, rope_theta, softcap, rms_norm_eps):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    q = _linear('gemm_bb_256_00_00000', hidden_states, q_proj_weight)
    k = _linear('gemm_bb_256_00_00000', hidden_states, k_proj_weight)
    v = _linear('gemm_bb_256_00_00000', hidden_states, v_proj_weight)
    inv = (1.0 / (float(rope_theta) ** (torch.arange(0, 256, 2, dtype=torch.float32) / 256))).to(dev)
    pos = _aligned(position_ids)
    qr, kr, vn = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _hnr('hnr_bbb_256_2111', q, qr, b * s * 8, 8, 1, rms_norm_eps, weight=_aligned(q_norm_weight), positions=pos, inv_freq=inv)
    _hnr('hnr_bbb_256_2111', k, kr, b * s, 1, 1, rms_norm_eps, weight=_aligned(k_norm_weight), positions=pos, inv_freq=inv)
    _hnr('hnr_bbb_256_5000', v, vn, b * s, 1, 1, rms_norm_eps)
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, 8, 256), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_256_64_1_000110', qr.view(b, s, 8, 256), kr.view(b, s, 1, 256), vn.view(b, s, 1, 256), o,
           mask=mask, scale=256 ** -0.5, cap=float(softcap), mask_strides=(s * s, 0, s))
    out = _linear('gemm_bb_256_00_00000', o.view(b, s, 8 * 256), o_proj_weight)
    return out, kr.view(b, 1, s, 256), vn.view(b, 1, s, 256)
