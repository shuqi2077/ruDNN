# problem: 028
def run(grad_output, hidden_states, q_weight, k_weight, v_weight, o_weight, inv_freq, scaling):
    b, seq, hidden = hidden_states.shape
    q, k, v = [_linear('gemm_bb_256_00_00000', hidden_states, w).view(b, seq, heads, 128).transpose(1, 2)
               for w, heads in ((q_weight, 64), (k_weight, 8), (v_weight, 8))]
    cos, sin = [torch.empty((1, seq, 128), dtype=torch.float32, device=hidden_states.device) for _ in range(2)]
    _launch('rope_frequency_1', [_aligned(inv_freq), cos, sin], _blocks(cos.numel(), 256), {'I': [128]})
    q = _rope_forward(q, cos, sin, True, True)
    k = _repeat_heads(_rope_forward(k, cos, sin, True, True), 64)
    v = _repeat_heads(v, 64)
    p = _softmax_raw(_affine(_matmul(q, k, transpose_b=True), scaling), True)
    pd = _affine(p, dtype=hidden_states.dtype)
    attn = _merge_heads(_matmul(pd, v))
    ga, gow, _ = _linear_backward(grad_output, attn, o_weight)
    gh = ga.view(b, seq, 64, 128).transpose(1, 2)
    gp, gv = _matmul(gh, v, transpose_b=True), _matmul(pd, gh, transpose_a=True)
    ds = _softmax_backward(gp, p, q.dtype, scaling)
    gq = _matmul(ds, k)
    gk = _sum(_matmul(q, ds, transpose_a=True).transpose(-2, -1).reshape(b, 8, 8, seq, 128), 2)
    gv = _sum(gv.reshape(b, 8, 8, seq, 128), 2)
    gx, gqw, gkw, gvw = _qkv_backward(_rope_backward(gq, cos, sin, True, True),
                                     _rope_backward(gk, cos, sin, True, True), gv, hidden_states, q_weight, k_weight, v_weight)
    return gx, gqw, gkw, gvw, gow

# problem: 034
def run(language_hidden_states, vision_hidden_states, language_position_ids, vision_grid_thw,
        q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight):
    b, sq, _ = language_hidden_states.shape
    sk = vision_hidden_states.shape[1]
    q = _linear('gemm_bb_256_00_10000', language_hidden_states, q_proj_weight, bias=_aligned(q_proj_bias)).view(b, sq, 32, 128)
    k = _linear('gemm_bb_256_00_10000', vision_hidden_states, k_proj_weight, bias=_aligned(k_proj_bias)).view(b, sk, 8, 128)
    v = _linear('gemm_bb_256_00_10000', vision_hidden_states, v_proj_weight, bias=_aligned(v_proj_bias)).view(b, sk, 8, 128)
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _launch('multimodal_rope_0', [q, _aligned(language_position_ids), qr], _blocks(q.numel(), 256), {'I': [32]})
    _launch('multimodal_rope_1', [k, _aligned(vision_grid_thw), kr], _blocks(k.numel(), 256), {'I': [8]})
    out, _ = _attn_bf16(qr, kr, v)
    return _linear('gemm_bb_256_00_00000', out.view(b, sq, 4096), o_proj_weight)

# problem: 037
def run(grad_output, hidden_states, cos, sin, q_weight, k_weight, v_weight, o_weight, q_norm_weight,
        k_norm_weight, scaling, attn_logit_softcapping, rms_norm_eps):
    b, seq, hidden = hidden_states.shape
    q, k, v = [_linear_f32(hidden_states, w).view(b, seq, heads, 128).transpose(1, 2)
               for w, heads in ((q_weight, 24), (k_weight, 8), (v_weight, 8))]
    qvar, kvar = [_sum(_mul(t, t), -1, True, mean=True) for t in (q, k)]
    qn, kn = [_mul(t, _unary('un_ff_11', _affine(var, bias=rms_norm_eps))) for t, var in ((q, qvar), (k, kvar))]
    qw, kw = _affine(q_norm_weight, bias=1), _affine(k_norm_weight, bias=1)
    qr, kr = _rope_forward(_mul(qn, qw), cos, sin), _rope_forward(_mul(kn, kw), cos, sin)
    kre, vre = _repeat_heads(kr, 24), _repeat_heads(v, 24)
    scores = _affine(_matmul(qr, kre, transpose_b=True), scaling)
    tanh = _unary('un_ff_4', _affine(scores, 1.0 / attn_logit_softcapping))
    p = _softmax_raw(_affine(tanh, attn_logit_softcapping), True)
    ga, gow, _ = _linear_backward(grad_output, _merge_heads(_matmul(p, vre)), o_weight)
    gh = ga.view(b, seq, 24, 128).transpose(1, 2)
    gs = _softmax_backward(_matmul(gh, vre, transpose_b=True), p)
    gs = _affine(_mul(gs, _unary('un_ff_17', tanh)), scaling)
    gq = _rope_backward(_matmul(gs, kre), cos, sin)
    gk = _rope_backward(_sum(_matmul(gs, qr, transpose_a=True).reshape(b, 8, 3, seq, 128), 2), cos, sin)
    gv = _sum(_matmul(p, gh, transpose_a=True).reshape(b, 8, 3, seq, 128), 2)
    gq, gqn = _rms_backward(gq, q, qvar, qn, qw, rms_norm_eps)
    gk, gkn = _rms_backward(gk, k, kvar, kn, kw, rms_norm_eps)
    gx, gqw, gkw, gvw = _qkv_backward(gq, gk, gv, hidden_states, q_weight, k_weight, v_weight)
    return gx, gqw, gkw, gvw, gow, gqn, gkn

# problem: 051
def run(hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias, in_proj_weight, in_proj_bias,
        short_conv_weight, short_conv_bias, filter_linear1_weight, filter_linear1_bias, sin_freq,
        filter_linear2_weight, filter_linear2_bias, filter_linear3_weight, filter_linear3_bias,
        filter_linear_final_weight, filter_bias, exp_mod_deltas, out_proj_weight, out_proj_bias,
        mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias, layer_norm_eps, exp_mod_shift):
    b, seq, hidden = hidden_states.shape
    length = min(seq, 32768)
    n = _layernorm_f32(hidden_states, norm1_weight, norm1_bias, layer_norm_eps)
    u = _linear_f32(n, in_proj_weight, in_proj_bias).transpose(1, 2).contiguous()
    uc = torch.empty((b, 768, length), dtype=torch.float32, device=u.device)
    _launch('depthwise1d_f32', [u, _aligned(short_conv_weight), _aligned(short_conv_bias), uc],
            _blocks(uc.numel(), 256), {'I': [seq, length, 768, short_conv_weight.shape[-1], 2]})
    x0, x1, v = uc.chunk(3, dim=1)
    z = torch.empty((1, length, 5), dtype=torch.float32, device=u.device)
    _launch('hyena_basis', [z], _blocks(z.numel(), 256), {'I': [length]})
    h = z
    for w, bias in ((filter_linear1_weight, filter_linear1_bias), (filter_linear2_weight, filter_linear2_bias), (filter_linear3_weight, filter_linear3_bias)):
        h = _unary('un_ff_8', _mul(_linear_f32(h, w, bias), sin_freq))
    h = _linear_f32(h, filter_linear_final_weight)
    filt = torch.empty_like(h)
    _launch('hyena_filter', [h, _aligned(exp_mod_deltas), _aligned(filter_bias), filt],
            _blocks(h.numel(), 256), {'I': [length], 'f': [float(exp_mod_shift)]})
    kernel = filt.transpose(0, 1).reshape(256, length)
    y = _mul(_hyena_convolve(_mul(v, x1), kernel, filter_bias), x0).transpose(1, 2)
    if length != seq:
        y = torch.cat((y, torch.zeros((b, seq - length, hidden), dtype=y.dtype, device=y.device)), dim=1)
    residual = _add(hidden_states, _linear_f32(y, out_proj_weight, out_proj_bias))
    n = _layernorm_f32(residual, norm2_weight, norm2_bias, layer_norm_eps)
    h = _unary('un_ff_1', _linear_f32(n, mlp_fc1_weight, mlp_fc1_bias))
    return _add(residual, _linear_f32(h, mlp_fc2_weight, mlp_fc2_bias))

# problem: 052
def run(v, x0, x1, k, bias, out_proj_weight, out_proj_bias):
    h = _hyena_convolve(_mul(v, x1), k[0], bias[0])
    y = _mul(h, x0).transpose(1, 2)
    return _add(_linear_f32(y, out_proj_weight), out_proj_bias)
