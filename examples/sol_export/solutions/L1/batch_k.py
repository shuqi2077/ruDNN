# L1 batch K: 089-093.

# problem: 089
def run(x, group_norm_weight, group_norm_bias, query_weight, query_bias, key_weight, key_bias, value_weight, value_bias,
        proj_out_weight, proj_out_bias, eps):
    b, c, hh, ww = x.shape
    hw = hh * ww
    dev = x.device
    xc = _aligned(x)
    chunk = 16384
    chunks = (hw + chunk - 1) // chunk
    partial = torch.empty(b * c * chunks * 2, dtype=torch.float64, device=dev)
    xn = torch.empty_like(xc)
    _launch('l1_078_gn_partial', [xc, partial], (chunks, c, b), {'I': [hw, chunk]})
    _launch('l1_078_gn_apply', [xc, partial, _aligned(group_norm_weight), _aligned(group_norm_bias), xn], (chunks, c, b),
            {'I': [hw, chunk, 32], 'f': [float(eps)]})
    planes = _split3(xn, b * c * hw, b * c * hw, True).view(3 * b, c, hw)
    xa = _opb(planes, div=1, mul=1, kstep=b)
    proj = []
    for w, bias in ((query_weight, query_bias), (key_weight, key_bias), (value_weight, value_bias)):
        o = torch.empty((b, hw, c), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_10_10000', xa, _tf32x3(w, False), o, hw, c, c, batches=b, k_groups=3, ld=c, o_s1=hw * c,
              bias=_aligned(bias))
        proj.append(o)
    q, k, v = proj
    attn = _attn_f32(q.view(b * hw, c), k.view(b * hw, c), v.view(b * hw, c), b, hw, hw, 1, 1, c, scale=c ** -0.5)
    out = torch.empty((b, c, hh, ww), dtype=torch.float32, device=dev)
    attn_planes = _split3(attn, b * hw * c, b * hw * c, False).view(3 * b, hw, c)
    _gemm('gemm_tf_256_00_20101', _tf32x3(proj_out_weight, True), _opb(attn_planes, div=1, mul=1, kstep=b), out, c, hw, c,
          batches=b, k_groups=3, ld=hw, o_s1=c * hw, bias=_aligned(proj_out_bias), residual=xc)
    return out

# problem: 091
def run(grad_query, grad_key, grad_value, hidden_states, cos, sin, qkv_weight, q_norm_weight, k_norm_weight,
        query_pre_norm, key_pre_norm, q_rstd, k_rstd, eps):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    rows = b * s
    gqkv = torch.empty((b, s, 6144), dtype=torch.bfloat16, device=dev)
    grad_cos = torch.empty((b, s, 128), dtype=torch.bfloat16, device=dev)
    grad_sin = torch.empty_like(grad_cos)
    blocks = max(1, min(rows, _sm_count(dev) * 8))
    partial = torch.empty((blocks, 256), dtype=torch.float32, device=dev)
    _launch('l1_091_qkrope_bwd', [_aligned(t) for t in (grad_query, grad_key, grad_value, query_pre_norm, key_pre_norm,
                                                          q_rstd, k_rstd, cos, sin, q_norm_weight, k_norm_weight)]
            + [gqkv, grad_cos, grad_sin, partial], (blocks, 1, 1), {'I': [rows, s]})
    sums = torch.empty(256, dtype=torch.bfloat16, device=dev)
    _launch('colfin_b', [partial, sums], _blocks(256, 256), {'I': [256, blocks]})
    g2 = gqkv.view(rows, 6144)
    grad_hidden = torch.empty((b, s, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_01_00000', _op(g2), _op(_aligned(qkv_weight)), grad_hidden, rows, hid, 6144)
    grad_w = torch.empty((6144, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(g2), _op(_aligned(hidden_states).view(rows, hid)), grad_w, 6144, hid, rows)
    return grad_hidden, grad_cos, grad_sin, grad_w, sums[:128], sums[128:]

# problem: 092
def run(hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight,
        q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
    b, s, _ = hidden_states.shape
    q = _linear('gemm_bb_256_00_10000', hidden_states, q_proj_weight, bias=_aligned(q_proj_bias))
    k = _linear('gemm_bb_256_00_10000', hidden_states, k_proj_weight, bias=_aligned(k_proj_bias))
    v = _linear('gemm_bb_256_00_10000', hidden_states, v_proj_weight, bias=_aligned(v_proj_bias))
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    c, sn = _aligned(cos), _aligned(sin)
    _hnr('hnr_bbb_128_2101', q, qr, b * s * 96, 96, 1, rms_norm_eps, weight=_aligned(q_norm_weight), cos=c, sin=sn)
    _hnr('hnr_bbb_128_2101', k, kr, b * s * 8, 8, 1, rms_norm_eps, weight=_aligned(k_norm_weight), cos=c, sin=sn)
    o = torch.empty((b, s, 96, 128), dtype=torch.bfloat16, device=q.device)
    _flash('flash_bb_128_128_2_010000', qr.view(b, s, 96, 128), kr.view(b, s, 8, 128), v.view(b, s, 8, 128), o, scale=128 ** -0.5)
    return _linear('gemm_bb_256_00_00000', o.view(b, s, 96 * 128), o_proj_weight)

# problem: 093
def run(grad_topk_weights, hidden_states, weight, scores, topk_indices, topk_weights, topk_weights_normalized, denominator,
        routed_scaling_factor):
    n, hid = hidden_states.shape
    e = scores.shape[1]
    dev = hidden_states.device
    grl = torch.empty((n, e), dtype=torch.bfloat16, device=dev)
    _launch('l1_093_router_bwd', [_aligned(grad_topk_weights), _aligned(topk_weights_normalized), _aligned(topk_indices),
                                  _aligned(denominator), _aligned(scores), grl], (n, 1, 1),
            {'I': [e], 'f': [float(routed_scaling_factor)]})
    grad_hidden = torch.empty((n, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_01_00000', _op(grl), _op(_aligned(weight)), grad_hidden, n, hid, e)
    grad_weight = torch.empty((e, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(grl), _op(_aligned(hidden_states)), grad_weight, e, hid, n)
    return grad_hidden, grad_weight
