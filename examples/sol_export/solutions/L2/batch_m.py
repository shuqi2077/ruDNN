# problem: 001
def run(grad_output, x, x_mean, x_var, x_norm, ln_weight, qkv_weight, q, k, v,
        attn_weights, attn_output, out_weight, scale, norm_eps):
    b, seq, hidden = x.shape
    ga, gow, gob = _linear_backward(grad_output, attn_output, out_weight)
    gh = ga.view(b, seq, 16, 64).transpose(1, 2)
    gp, gv = _matmul(gh, v, transpose_b=True), _matmul(attn_weights, gh, transpose_a=True)
    gs = _softmax_backward(gp, attn_weights)
    gq = _affine(_matmul(gs, k), scale)
    gk = _matmul(gs, _affine(q, scale), transpose_a=True)
    gqkv = torch.cat(tuple(_merge_heads(g) for g in (gq, gk, gv)), dim=-1)
    gn, gqw, gqb = _linear_backward(gqkv, x_norm, qkv_weight)
    std = _unary('un_ff_10', _affine(x_var, bias=norm_eps))
    centered = _sub(x, x_mean)
    normalized = _div(centered, std)
    glnw, glnb = _sum(_mul(gn, normalized), (0, 1)), _sum(gn, (0, 1))
    gnormalized = _mul(gn, ln_weight)
    gx = _div(gnormalized, std)
    mean_grad = _sum(gx, -1, True, mean=True)
    correlation = _div(_sum(_mul(gnormalized, centered), -1, True, mean=True), _mul(std, std))
    gx = _sub(_sub(gx, mean_grad), _mul(centered, correlation))
    return _add(grad_output, gx), gqw, gqb, gow, gob, glnw, glnb

# problem: 003
def run(grad_output, hidden_states, cos, sin, q_weight, k_weight, v_weight, o_weight,
        query_states, key_states, value_states, attn_weights, attn_output, scaling):
    b, seq, _ = hidden_states.shape
    ga, gow, _ = _linear_backward(grad_output, attn_output, o_weight)
    gq, gk, gv = _attention_backward(ga.view(b, seq, 32, 128).transpose(1, 2), query_states,
                                     key_states, value_states, attn_weights, scaling, 8, True)
    gq, gk = _rope_backward(gq, cos, sin), _rope_backward(gk, cos, sin)
    gx, gqw, gkw, gvw = _qkv_backward(gq, gk, gv, hidden_states, q_weight, k_weight, v_weight)
    return gx, gqw, gkw, gvw, gow

# problem: 021
def run(grad_output, video_latents, text_embeddings, query_weight, query_bias, key_weight, key_bias,
        value_weight, value_bias, output_weight, output_bias, scale):
    b, sq, hidden = video_latents.shape
    sk = text_embeddings.shape[1]
    q = _linear_f32(video_latents, query_weight, query_bias).view(b, sq, 16, 64).transpose(1, 2)
    k = _linear_f32(text_embeddings, key_weight, key_bias).view(b, sk, 16, 64).transpose(1, 2)
    v = _linear_f32(text_embeddings, value_weight, value_bias).view(b, sk, 16, 64).transpose(1, 2)
    p = _softmax_raw(_affine(_matmul(q, k, transpose_b=True), scale))
    context = _merge_heads(_matmul(p, v))
    ga, gow, gob = _linear_backward(grad_output, context, output_weight)
    gq, gk, gv = _attention_backward(ga.view(b, sq, 16, 64).transpose(1, 2), q, k, v, p, scale)
    gx, gqw, gqb = _linear_backward(_merge_heads(gq), video_latents, query_weight)
    gtx, gkw, gkb = _linear_backward(_merge_heads(gk), text_embeddings, key_weight)
    gtv, gvw, gvb = _linear_backward(_merge_heads(gv), text_embeddings, value_weight)
    return gx, _add(gtx, gtv), gqw, gqb, gkw, gkb, gvw, gvb, gow, gob

# problem: 042
def run(grad_output, hidden_states, fc1_weight, fc1_output, gelu_output, fc2_weight,
        residual_output, normalized, var, ln_weight, eps):
    glnw = _sum(_mul(grad_output, normalized), (0, 1))
    glnb = _sum(grad_output, (0, 1))
    gn = _mul(grad_output, ln_weight)
    gnmean = _sum(gn, -1, True, mean=True)
    correlation = _sum(_mul(gn, normalized), -1, True, mean=True)
    r = _unary('un_ff_11', _affine(var, bias=eps))
    gr = _mul(r, _sub(_sub(gn, gnmean), _mul(normalized, correlation)))
    gg, gw2, gb2 = _linear_backward(gr, gelu_output, fc2_weight)
    gi = _mul(gg, _unary('un_ff_14', fc1_output))
    gx, gw1, gb1 = _linear_backward(gi, hidden_states, fc1_weight)
    return _add(gx, gr), gw1, gb1, gw2, gb2, glnw, glnb

# problem: 056
def run(grad_output, residual, attn_input, query_states, key_states, value_states, query_states_rotated,
        key_states_rotated, key_states_repeated, value_states_repeated, cos, sin, attn_weights, attn_output,
        residual2, ffn_input, gate, up, silu_up, swiglu_output, input_ln_weight, q_weight, k_weight, v_weight,
        o_weight, post_attn_ln_weight, gate_weight, up_weight, down_weight, variance1, variance2,
        hidden_states_normalized1, hidden_states_normalized2, eps):
    b, seq, hidden = grad_output.shape
    gs, gdw, _ = _linear_backward(grad_output, swiglu_output, down_weight)
    gg = _mul(gs, silu_up)
    derivative = _affine(_unary('un_ff_16', _f32(up)), dtype=up.dtype)
    gu = _mul(_mul(gs, gate), derivative)
    gfg, ggw, _ = _linear_backward(gg, ffn_input, gate_weight)
    gfu, guw, _ = _linear_backward(gu, ffn_input, up_weight)
    gr2, gpostw = _rms_backward(_add(gfg, gfu), residual2, variance2, hidden_states_normalized2, post_attn_ln_weight, eps)
    gr2 = _add(grad_output, gr2)
    ga, gow, _ = _linear_backward(gr2, attn_output, o_weight)
    gq, gk, gv = _attention_backward(ga.view(b, seq, 32, 160).transpose(1, 2), query_states_rotated,
                                     key_states_repeated, value_states_repeated, attn_weights, 160 ** -0.5, 8)
    def unrotate(g):
        a, z = g.chunk(2, dim=-1)
        rotated = torch.cat((_affine(z, -1.0), a), dim=-1)
        return _add(_mul(g, cos.unsqueeze(1)), _mul(rotated, _affine(sin.unsqueeze(1), -1.0)))
    gai, gqw, gkw, gvw = _qkv_backward(unrotate(gq), unrotate(gk), gv, attn_input, q_weight, k_weight, v_weight)
    gr1, ginw = _rms_backward(gai, residual, variance1, hidden_states_normalized1, input_ln_weight, eps)
    return _add(gr2, gr1), ginw, gqw, gkw, gvw, gow, gpostw, ggw, guw, gdw

# problem: 064
def run(grad_query, grad_key, grad_value, hidden_states, q_weight, k_weight, v_weight, cos, sin):
    return _qkv_backward(_rope_backward(grad_query, cos, sin), _rope_backward(grad_key, cos, sin),
                         grad_value, hidden_states, q_weight, k_weight, v_weight)
