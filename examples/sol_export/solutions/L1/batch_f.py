# L1 batch F: 042-050.

# problem: 042
def run(grad_tokens_per_expert, grad_expert_mask, grad_load_balance_loss, topk_idx, expert_mask, tokens_per_expert, training):
    rows = topk_idx.shape[0]
    extra = 0.0
    if bool(training.item()):
        extra = float(grad_load_balance_loss.item()) * 256 * (1.0 / 256) / (rows * 8)
    out = torch.empty_like(grad_expert_mask)
    n = out.numel()
    _launch('l1_042_add', [_aligned(grad_expert_mask), _aligned(grad_tokens_per_expert), out], _blocks(n, 256),
            {'I': [n, 256], 'f': [extra]})
    return out

# problem: 043
def run(hidden_states, q_a_proj_weight, q_a_layernorm_weight, q_b_proj_weight, kv_a_proj_weight, rms_norm_eps):
    b, s, _ = hidden_states.shape
    dev = hidden_states.device
    ql = _linear('gemm_bb_256_00_00000', hidden_states, q_a_proj_weight)
    ql = _norm('norm_bbb_001001', ql, 1536, rms_norm_eps, weight=_aligned(q_a_layernorm_weight)).view(b * s, 1536)
    wq = _aligned(q_b_proj_weight)
    q_nope = torch.empty((b, s, 128, 128), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_00_00000', _op(ql), _Op(wq, [1536, 1, 24576], [3072, 3072], z_div=1, z_mul=192), q_nope,
          b * s, 128, 1536, batches=128, ld=128 * 128, o_s1=128)
    q_pe = torch.empty((b, s, 128, 64), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_64_00_00000', _op(ql), _Op(wq[128:], [1536, 1, 24576 - 128], [3072, 3072], z_div=1, z_mul=192), q_pe,
          b * s, 64, 1536, batches=128, ld=128 * 64, o_s1=64)
    wkv = _aligned(kv_a_proj_weight)
    compressed = _linear('gemm_bb_256_00_00000', hidden_states, wkv[:512])
    k_pe = _linear('gemm_bb_64_00_00000', hidden_states, wkv[512:])
    return q_nope, q_pe, compressed, k_pe.view(b, s, 1, 64)

# problem: 044
def run(hidden_states, expert_ids, expert_weights, gate_proj_weights, up_proj_weights, down_proj_weights):
    t, h = hidden_states.shape
    experts, inter, _ = gate_proj_weights.shape
    top_k = expert_ids.shape[1]
    groups, sorted_tokens, inverse = _moe_route(expert_ids, experts)
    xs = _moe_gather('moe_gather_b', hidden_states, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00001', xs, gate_proj_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00201', xs, up_proj_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00001', act, down_proj_weights, groups, h, inter)
    return _moe_combine('moe_combine_bb_10', y, inverse, expert_weights, t, top_k)

# problem: 045
def run(hidden_states, pwconv1_weight, pwconv1_bias, grn_weight, grn_bias, pwconv2_weight, pwconv2_bias, eps):
    b, hh, ww, c = hidden_states.shape
    dev = hidden_states.device
    rows = b * hh * ww
    hid = pwconv1_weight.shape[0]
    x1 = torch.empty((rows, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_12000', hidden_states.reshape(rows, c), pwconv1_weight, x1, rows, hid, c, bias=_aligned(pwconv1_bias))
    sumsq = torch.empty((b, hid), dtype=torch.float32, device=dev)
    _launch('grn_sumsq', [x1, sumsq], ((hid + 255) // 256, b, 1), {'I': [hh * ww, hid]})
    nx = torch.empty_like(sumsq)
    _launch('grn_scale', [sumsq, nx], (b, 1, 1), {'f': [float(eps)]})
    x2 = torch.empty_like(x1)
    _launch('grn_apply', [x1, nx, _aligned(grn_weight).view(-1), _aligned(grn_bias).view(-1), x2], _blocks(x1.numel(), 256),
            {'I': [x1.numel(), hh * ww, hid]})
    out = torch.empty((rows, c), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_128_00_10000', x2, pwconv2_weight, out, rows, c, hid, bias=_aligned(pwconv2_bias))
    return out.view(b, hh, ww, c)

# problem: 047
def run(hidden_states, cos, sin, attention_mask, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        q_norm_weight, k_norm_weight, attn_logit_softcapping, rms_norm_eps):
    b, s, _ = hidden_states.shape
    dev = hidden_states.device
    q = _linear('gemm_bb_256_00_00000', hidden_states, q_proj_weight)
    k = _linear('gemm_bb_256_00_00000', hidden_states, k_proj_weight)
    v = _linear('gemm_bb_256_00_00000', hidden_states, v_proj_weight)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    c, sn = _aligned(cos), _aligned(sin)
    _hnr('hnr_bbb_128_3101', q, qr, b * s * 24, 24, 1, rms_norm_eps, weight=_aligned(q_norm_weight), cos=c, sin=sn)
    _hnr('hnr_bbb_128_3101', k, kr, b * s * 8, 8, 1, rms_norm_eps, weight=_aligned(k_norm_weight), cos=c, sin=sn)
    n = b * s * s
    mask = torch.empty((b, s, s), dtype=torch.float32, device=dev)
    am = _aligned(attention_mask)
    _launch('bin_bbf_0', [am, am, mask], _blocks(n, 256), {'I': [1, 1, n, n, 0, 0, 0, 1, 0, 0, 0, 1], 'f': [0.0]})
    o = torch.empty((b, s, 24, 128), dtype=torch.bfloat16, device=dev)
    _flash('flash_bb_128_128_2_000110', qr.view(b, s, 24, 128), kr.view(b, s, 8, 128), v.view(b, s, 8, 128), o,
           mask=mask, scale=128 ** -0.5, cap=float(attn_logit_softcapping), mask_strides=(s * s, 0, s))
    return _linear('gemm_bb_256_00_00000', o.view(b, s, 24 * 128), o_proj_weight)

# problem: 048
def run(x, gate_proj, up_proj):
    gate = _linear('gemm_bb_256_00_00001', x, gate_proj)
    return _linear('gemm_bb_256_00_00301', x, up_proj, residual=gate)

# problem: 049
def run(query, key, scaling):
    b, h, s, d = query.shape
    kv = key.shape[1]
    ldp = (s + 7) // 8 * 8
    out = torch.empty((b, h, s, ldp), dtype=torch.bfloat16, device=query.device)
    _gemm('gemm_bb_256_00_00000', _opb(_aligned(query).view(b * h, s, d)), _opb(_aligned(key).view(b * kv, s, d), div=h // kv),
          out, s, ldp, d, batches=b * h, ld=ldp, o_s1=s * ldp, alpha=float(scaling))
    return out[..., :s]

# problem: 050
def run(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias):
    b, s, _ = hidden_states.shape
    q = _linear('gemm_bb_256_00_10002', hidden_states, q_weight, bias=_aligned(q_bias))
    k = _linear('gemm_bb_256_00_10002', hidden_states, k_weight, bias=_aligned(k_bias))
    v = _linear('gemm_bb_256_00_10002', hidden_states, v_weight, bias=_aligned(v_bias))
    return q.view(b, s, 4, 256), k.view(b, s, 1, 256), v.view(b, s, 1, 256)
