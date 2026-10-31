# Quant batch C: 013-017.

# problem: 013
def run(hidden_states, kv_a_proj_weight, kv_a_layernorm_weight, kv_b_proj_weight, rms_norm_eps):
    b, s, h = hidden_states.shape
    xq = _fq_act(hidden_states.reshape(b * s, h))
    wa = _fq_weight(kv_a_proj_weight)
    compressed = _linear('gemm_bb_256_00_00000', xq, wa[:512])
    k_pe = _linear('gemm_bb_64_00_00000', xq, wa[512:576])
    normed = _norm('norm_bbb_001001', compressed, 512, rms_norm_eps, weight=_aligned(kv_a_layernorm_weight))
    kv = _linear('gemm_bb_256_00_00000', _fq_act(normed), _fq_weight(kv_b_proj_weight))
    return kv.view(b, s, 128, 256), k_pe.view(b, s, 1, 64)

# problem: 014
def run(position_ids, mscale, mscale_all_dim):
    import math
    dev = position_ids.device
    dim, base, factor, orig = 64, 10000, 40, 4096
    exps = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    freq_extra = 1.0 / (base ** exps)
    freq_inter = 1.0 / (factor * base ** exps)
    def corr(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(corr(32)), 0), min(math.ceil(corr(1)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    inv = (freq_inter * (1 - mask) + freq_extra * mask).to(dev)
    def get_mscale(scale, m):
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
    ms = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
    n = position_ids.shape[0]
    cos = torch.empty((n, dim), dtype=torch.bfloat16, device=dev)
    sin = torch.empty_like(cos)
    _launch('q_014_yarn', [_aligned(position_ids), inv, cos, sin], ((n + 7) // 8, 1, 1), {'I': [n], 'f': [ms]})
    return cos, sin

# problem: 015
def run(hidden_states, weight, scale_x, scale_w):
    b, s, k = hidden_states.shape
    kb = k // 128
    nb = weight.shape[0] // 128
    xd = _dq_fp8(hidden_states.view(b * s, k), scale_x.reshape(b * s, kb), 1, kb, 1)
    wd = _dq_fp8(weight, scale_w, 128, 1, nb)
    return _linear('gemm_bb_256_00_00000', xd, wd).view(b, s, weight.shape[0])

# problem: 016
def run(hidden_states, q_a_proj_weight, q_a_layernorm_weight, q_b_proj_weight, kv_a_proj_weight, kv_a_layernorm_weight,
        kv_b_proj_weight, rms_norm_eps):
    b, s, h = hidden_states.shape
    xq = _fq_act(hidden_states.reshape(b * s, h))
    qa = _linear('gemm_bb_256_00_00000', xq, _fq_weight(q_a_proj_weight))
    qa_n = _norm('norm_bbb_001000', qa, qa.shape[1], rms_norm_eps, weight=_aligned(q_a_layernorm_weight))
    q = _linear('gemm_bb_256_00_00000', _fq_act(qa_n), _fq_weight(q_b_proj_weight)).view(b, s, 128, 192)
    wa = _fq_weight(kv_a_proj_weight)
    ckv = _linear('gemm_bb_256_00_00000', xq, wa[:512])
    k_pe = _linear('gemm_bb_64_00_00000', xq, wa[512:576])
    ckv_n = _norm('norm_bbb_001000', ckv, 512, rms_norm_eps, weight=_aligned(kv_a_layernorm_weight))
    kv = _linear('gemm_bb_256_00_00000', _fq_act(ckv_n), _fq_weight(kv_b_proj_weight)).view(b, s, 128, 256)
    return q[..., :128], q[..., 128:], kv[..., :128], k_pe.view(b, s, 1, 64), kv[..., 128:]

# problem: 017
def run(x, gate_proj_weight, gate_proj_weight_scales, up_proj_weight, up_proj_weight_scales, down_proj_weight,
        down_proj_weight_scales):
    xq = _fq_act(x)
    def wq(w, sc):
        return _fq_weight(w, sc, s_n=1, s_k=w.shape[0] // 128)
    gate = _linear('gemm_bb_256_00_00001', xq, wq(gate_proj_weight, gate_proj_weight_scales))
    act = _linear('gemm_bb_256_00_00201', xq, wq(up_proj_weight, up_proj_weight_scales), residual=gate)
    return _linear('gemm_bb_256_00_00000', _fq_act(act), wq(down_proj_weight, down_proj_weight_scales))
