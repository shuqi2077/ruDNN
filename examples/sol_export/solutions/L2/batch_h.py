# L2 batch H: fp32 decoder / encoder layers (3xTF32 GEMMs + 3xTF32 attention).

# problem: 019
def run(hidden_states, input_layernorm_weight, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight,
        v_proj_bias, o_proj_weight, rope_cos, rope_sin, post_attention_layernorm_weight, gate_proj_weight, up_proj_weight,
        down_proj_weight, rms_norm_eps):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    rows = b * s
    x = _aligned(hidden_states).view(rows, hid)
    h1 = _norm('norm_fff_001000', x, hid, rms_norm_eps, weight=_aligned(input_layernorm_weight))
    xa = _tf32x3(h1, True)
    proj = []
    for w, bias in ((q_proj_weight, q_proj_bias), (k_proj_weight, k_proj_bias), (v_proj_weight, v_proj_bias)):
        o = torch.empty((rows, w.shape[0]), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_10000', xa, _tf32x3(w, False), o, rows, w.shape[0], hid, k_groups=3, bias=_aligned(bias))
        proj.append(o)
    q, k, v = proj
    kvs = rope_cos.shape[2]
    tabs = []
    for t in (rope_cos, rope_sin):
        o = torch.empty((b, s, 128), dtype=torch.float32, device=dev)
        _launch('l2_019_mrope', [_aligned(t), o], _blocks(o.numel(), 256), {'I': [o.numel(), s, kvs, b]})
        tabs.append(o)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_fff_128_0100', q, qr, rows * 28, 28, 1, cos=tabs[0], sin=tabs[1])
    _hnr('hnr_fff_128_0100', k, kr, rows * 4, 4, 1, cos=tabs[0], sin=tabs[1])
    attn = _attn_f32(qr, kr, v, b, s, s, 28, 4, 128, scale=128 ** -0.5, causal=True)
    h3 = torch.empty((rows, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_00100', attn.view(rows, hid), o_proj_weight, h3, rows, hid, hid, residual=x)
    h4 = _norm('norm_fff_001000', h3, hid, rms_norm_eps, weight=_aligned(post_attention_layernorm_weight))
    h4a = _tf32x3(h4, True)
    inter = gate_proj_weight.shape[0]
    gate = torch.empty((rows, inter), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_256_00_00000', h4a, _tf32x3(gate_proj_weight, False), gate, rows, inter, hid, k_groups=3)
    act = torch.empty_like(gate)
    _gemm('gemm_tf_256_00_00200', h4a, _tf32x3(up_proj_weight, False), act, rows, inter, hid, k_groups=3, residual=gate)
    out = torch.empty((b, s, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_00100', act, down_proj_weight, out, rows, hid, inter, residual=h3)
    return out

# problem: 063
def run(hidden_states, cos, sin, pre_sa_norm_weight, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        post_sa_norm_weight, gate_up_proj_weight, down_proj_weight, norm_eps):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    rows = b * s
    x = _aligned(hidden_states).view(rows, hid)
    h1 = _norm('norm_fff_001000', x, hid, norm_eps, weight=_aligned(pre_sa_norm_weight))
    xa = _tf32x3(h1, True)
    proj = []
    for w in (q_proj_weight, k_proj_weight, v_proj_weight):
        o = torch.empty((rows, w.shape[0]), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_00000', xa, _tf32x3(w, False), o, rows, w.shape[0], hid, k_groups=3)
        proj.append(o)
    q, k, v = proj
    c, sn = _aligned(cos), _aligned(sin)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_fff_64_0100', q, qr, rows * 16, 16, 1, cos=c, sin=sn)
    _hnr('hnr_fff_64_0100', k, kr, rows * 4, 4, 1, cos=c, sin=sn)
    attn = _attn_f32(qr, kr, v, b, s, s, 16, 4, 64, scale=1.0 / 8.0)
    h3 = torch.empty((rows, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_00100', attn.view(rows, hid), o_proj_weight, h3, rows, hid, hid, residual=x)
    h4 = _norm('norm_fff_001000', h3, hid, norm_eps, weight=_aligned(post_sa_norm_weight))
    h4a = _tf32x3(h4, True)
    inter = down_proj_weight.shape[1]
    wgu = _aligned(gate_up_proj_weight)
    gate = torch.empty((rows, inter), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_256_00_00000', h4a, _tf32x3(wgu[:inter], False), gate, rows, inter, hid, k_groups=3)
    act = torch.empty_like(gate)
    _gemm('gemm_tf_256_00_00200', h4a, _tf32x3(wgu[inter:], False), act, rows, inter, hid, k_groups=3, residual=gate)
    out = torch.empty((b, s, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_00100', act, down_proj_weight, out, rows, hid, inter, residual=h3)
    return out
