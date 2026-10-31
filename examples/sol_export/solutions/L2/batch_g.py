# L2 batch G: 020 (decoder layer, head_dim 96 padded to 128 for tcgen05 flash attention).

# problem: 020
def run(hidden_states, cos, sin, attention_mask, input_layernorm_weight, q_proj_weight, k_proj_weight, v_proj_weight,
        o_proj_weight, post_attention_layernorm_weight, gate_proj_weight, up_proj_weight, down_proj_weight, rms_norm_eps):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    heads, kv, hd = 64, 8, 96
    x = _aligned(hidden_states)
    h1 = _norm('norm_bbb_001001', x, hid, rms_norm_eps, weight=_aligned(input_layernorm_weight))
    h2 = h1.view(b * s, hid)
    def proj(w, nh):
        o = torch.zeros((b, s, nh, 128), dtype=torch.bfloat16, device=dev)
        _gemm('gemm_bb_256_00_00000', _op(h2), _op(_aligned(w)), o, b * s, nh * hd, hid, ld=nh * 128,
              col_split=hd, col_stride=128)
        return o
    q, k, v = proj(q_proj_weight, heads), proj(k_proj_weight, kv), proj(v_proj_weight, kv)
    c, sn = _aligned(cos), _aligned(sin)
    qr, kr = torch.zeros_like(q), torch.zeros_like(k)
    _hnr('hnr_bbb_96_01011', q, qr, b * s * heads, heads, 1, cos=c, sin=sn, stride=128)
    _hnr('hnr_bbb_96_01011', k, kr, b * s * kv, kv, 1, cos=c, sin=sn, stride=128)
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, heads, 128), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_128_128_2_000010', qr, kr, v, o, mask=mask, scale=hd ** -0.5, mask_strides=(s * s, 0, s))
    wo = _aligned(o_proj_weight)
    h3 = torch.empty((b * s, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_00_00101', _Op(o, [128, heads, b * s], [256, heads * 256], y_mul=0, y_kstep=1),
          _Op(wo, [hd, heads, hid], [hd * 2, heads * hd * 2], y_mul=0, y_kstep=1), h3, b * s, hid, hd, k_groups=heads,
          residual=x.view(b * s, hid))
    h4 = _norm('norm_bbb_001001', h3, hid, rms_norm_eps, weight=_aligned(post_attention_layernorm_weight))
    gate = _linear('gemm_bb_256_00_00001', h4, gate_proj_weight)
    act = _linear('gemm_bb_256_00_00201', h4, up_proj_weight, residual=gate)
    out = _linear('gemm_bb_256_00_00101', act, down_proj_weight, residual=h3)
    return out.view(b, s, hid)


# problem: 053
def run(hidden_states, attention_mask, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, gate_proj_weight,
        up_proj_weight, down_proj_weight, input_layernorm_weight, post_attention_layernorm_weight, rms_norm_eps, rope_theta):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    x = _aligned(hidden_states)
    h1 = _norm('norm_bbb_001001', x, hid, rms_norm_eps, weight=_aligned(input_layernorm_weight))
    q = _linear('gemm_bb_256_00_00000', h1, q_proj_weight)
    k = _linear('gemm_bb_256_00_00000', h1, k_proj_weight)
    v = _linear('gemm_bb_256_00_00000', h1, v_proj_weight)
    inv = (1.0 / (float(rope_theta) ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))).to(dev)
    pos = torch.arange(s).repeat(b).to(dev)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_bbb_128_0110', q, qr, b * s * 32, 32, 1, positions=pos, inv_freq=inv)
    _hnr('hnr_bbb_128_0110', k, kr, b * s * 8, 8, 1, positions=pos, inv_freq=inv)
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, 32, 128), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_128_128_2_000010', qr.view(b, s, 32, 128), kr.view(b, s, 8, 128), v.view(b, s, 8, 128), o,
           mask=mask, scale=128 ** -0.5, mask_strides=(s * s, 0, s))
    return _decoder_rest(x, o.view(b * s, 32 * 128), o_proj_weight, post_attention_layernorm_weight, gate_proj_weight,
                         up_proj_weight, down_proj_weight, rms_norm_eps, 'norm_bbb_001001')

# problem: 059
def run(hidden_states, position_ids, attention_mask, input_layernorm_weight, q_proj_weight, k_proj_weight, v_proj_weight,
        q_norm_weight, k_norm_weight, o_proj_weight, post_attention_layernorm_weight, gate_proj_weight, up_proj_weight,
        down_proj_weight, inv_freq, rms_norm_eps, attention_scale):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    x = _aligned(hidden_states)
    h1 = _norm('norm_bbb_001000', x, hid, rms_norm_eps, weight=_aligned(input_layernorm_weight))
    q = _linear('gemm_bb_256_00_00000', h1, q_proj_weight)
    k = _linear('gemm_bb_256_00_00000', h1, k_proj_weight)
    v = _linear('gemm_bb_256_00_00000', h1, v_proj_weight)
    pos, inv = _aligned(position_ids), _aligned(inv_freq)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_bbb_128_2111', q, qr, b * s * 40, 40, 1, rms_norm_eps, weight=_aligned(q_norm_weight), positions=pos, inv_freq=inv)
    _hnr('hnr_bbb_128_2111', k, kr, b * s * 8, 8, 1, rms_norm_eps, weight=_aligned(k_norm_weight), positions=pos, inv_freq=inv)
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, 40, 128), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_128_128_2_000010', qr.view(b, s, 40, 128), kr.view(b, s, 8, 128), v.view(b, s, 8, 128), o,
           mask=mask, scale=float(attention_scale), mask_strides=(s * s, 0, s))
    return _decoder_rest(x, o.view(b * s, 40 * 128), o_proj_weight, post_attention_layernorm_weight, gate_proj_weight,
                         up_proj_weight, down_proj_weight, rms_norm_eps, 'norm_bbb_001000')
