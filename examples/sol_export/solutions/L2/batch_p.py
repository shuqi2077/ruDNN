# problem: 032
def run(hidden_states, timestep_embedding, encoder_hidden_states, adaln_linear_weight, adaln_linear_bias,
        to_q_weight, to_q_bias, to_k_weight, to_k_bias, to_v_weight, to_v_bias, to_k_context_weight,
        to_k_context_bias, to_v_context_weight, to_v_context_bias, to_out_weight, to_out_bias,
        pos_idx_axis0, pos_idx_axis1, pos_idx_axis2, rope_cos_axis0, rope_sin_axis0, rope_cos_axis1,
        rope_sin_axis1, rope_cos_axis2, rope_sin_axis2, is_joint_block):
    b, seq, hidden = hidden_states.shape
    activated = _mul(timestep_embedding, _unary('un_ff_3', timestep_embedding))
    sc, sh, gate, _, _, _ = _linear_f32(activated, adaln_linear_weight, adaln_linear_bias).chunk(6, dim=-1)
    n = _norm('norm_fff_100000', _aligned(hidden_states), hidden, 1e-5)
    n = _add(_mul(n, _affine(sc.unsqueeze(1), bias=1)), sh.unsqueeze(1))
    q, k, v = [_linear_f32(n, w, bias).view(b, seq, 24, 128) for w, bias in ((to_q_weight, to_q_bias), (to_k_weight, to_k_bias), (to_v_weight, to_v_bias))]
    qs, ks = [], []
    for axis, ids, ct, st in ((0, pos_idx_axis0, rope_cos_axis0, rope_sin_axis0), (1, pos_idx_axis1, rope_cos_axis1, rope_sin_axis1), (2, pos_idx_axis2, rope_cos_axis2, rope_sin_axis2)):
        tables = []
        for table in (ct, st):
            dest = torch.empty((seq, table.shape[-1]), dtype=torch.float32, device=table.device)
            _launch('gather_scalar_f32', [_aligned(table), _aligned(ids), dest], _blocks(dest.numel(), 256), {'I': [table.shape[-1]]})
            tables.append(dest[:, :21].view(1, seq, 1, 21))
        c, s = tables
        for tensor, pieces in ((q, qs), (k, ks)):
            a, z = tensor[..., axis * 42:(axis + 1) * 42].chunk(2, dim=-1)
            pieces.append(torch.cat((_sub(_mul(a, c), _mul(z, s)), _add(_mul(a, s), _mul(z, c))), dim=-1))
    q, k = torch.cat((*qs, q[..., 126:]), dim=-1), torch.cat((*ks, k[..., 126:]), dim=-1)
    if is_joint_block == 1:
        context = _norm('norm_fff_100000', _aligned(encoder_hidden_states), 4096, 1e-5)
        kc = _linear_f32(context, to_k_context_weight, to_k_context_bias).view(b, -1, 24, 128)
        vc = _linear_f32(context, to_v_context_weight, to_v_context_bias).view(b, -1, 24, 128)
        k, v = torch.cat((k, kc), dim=1), torch.cat((v, vc), dim=1)
    a = _attention_f32(q.reshape(b, seq, hidden), k.reshape(b, -1, hidden), v.reshape(b, -1, hidden), 24)
    return _add(hidden_states, _mul(gate.unsqueeze(1), _linear_f32(a, to_out_weight, to_out_bias)))

# problem: 072
def run(grad_output, hidden_states, edit_region_mask, qkv_weight, qkv_bias, out_weight, out_bias,
        edit_region_bias, within_edit_bias, cross_edit_bias, attention_mask, query, key, value,
        attention_scores, attention_probs, attention_probs_dropped, attention_output, dropout_mask, scale, dropout_p):
    b, seq, hidden = hidden_states.shape
    ga, gow, gob = _linear_backward(grad_output, attention_output, out_weight)
    ga = ga.view(b, seq, 16, 64).transpose(1, 2)
    gp, gv = _matmul(ga, value, transpose_b=True), _matmul(attention_probs_dropped, ga, transpose_a=True)
    if dropout_p > 0:
        gp = _affine(_mul(gp, dropout_mask), 1.0 / (1 - dropout_p))
    gs = _softmax_backward(gp, attention_probs)
    masked = torch.empty_like(gs)
    _launch('mask_key_f32', [gs, _aligned(attention_mask), masked], _blocks(gs.numel(), 256), {'I': [16, seq, seq]})
    eq, ek = edit_region_mask.unsqueeze(2), edit_region_mask.unsqueeze(1)
    cross = _add(_mul(eq, _affine(ek, -1, 1)), _mul(_affine(eq, -1, 1), ek)).unsqueeze(1)
    within = _mul(eq, ek).unsqueeze(1)
    gcross = _sum(_mul(masked, cross), (0, 2, 3)).view(16, 1, 1)
    gwithin = _sum(_mul(masked, within), (0, 2, 3)).view(16, 1, 1)
    gedit = torch.zeros_like(edit_region_bias)
    if seq <= 1024:
        gedit[:, :seq, :seq].copy_(_sum(masked, 0))
    ds = _affine(masked, scale)
    gq, gk = _matmul(ds, key), _matmul(ds, query, transpose_a=True)
    gqkv = torch.stack((gq, gk, gv), dim=2).permute(0, 3, 2, 1, 4).reshape(b, seq, 3 * hidden)
    gx, gqw, gqb = _linear_backward(gqkv, hidden_states, qkv_weight)
    return gx, gqw, gqb, gow, gob, gedit, gwithin, gcross

# problem: 074
def run(grad_iou_scores, grad_hyper_weights, iou_token_out, mask_tokens_out, iou_proj_in_weight, iou_proj_in_bias,
        iou_hidden_weight, iou_hidden_bias, iou_proj_out_weight, iou_proj_out_bias, iou_hidden1, iou_hidden1_relu,
        iou_hidden2, iou_hidden2_relu, hyper_proj_in_weights, hyper_proj_in_biases, hyper_hidden_weights,
        hyper_hidden_biases, hyper_proj_out_weights, hyper_proj_out_biases, hyper_hidden1_0, hyper_hidden1_1,
        hyper_hidden1_2, hyper_hidden1_3, hyper_hidden1_relu_0, hyper_hidden1_relu_1, hyper_hidden1_relu_2,
        hyper_hidden1_relu_3, hyper_hidden2_0, hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3,
        hyper_hidden2_relu_0, hyper_hidden2_relu_1, hyper_hidden2_relu_2, hyper_hidden2_relu_3):
    iou = _three_layer_relu_backward(grad_iou_scores, iou_token_out, iou_proj_in_weight, iou_hidden_weight,
                                     iou_proj_out_weight, iou_hidden1, iou_hidden1_relu, iou_hidden2, iou_hidden2_relu)
    h1 = (hyper_hidden1_0, hyper_hidden1_1, hyper_hidden1_2, hyper_hidden1_3)
    a1 = (hyper_hidden1_relu_0, hyper_hidden1_relu_1, hyper_hidden1_relu_2, hyper_hidden1_relu_3)
    h2 = (hyper_hidden2_0, hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3)
    a2 = (hyper_hidden2_relu_0, hyper_hidden2_relu_1, hyper_hidden2_relu_2, hyper_hidden2_relu_3)
    parts = [_three_layer_relu_backward(grad_hyper_weights[:, :, i], mask_tokens_out[:, :, i],
             hyper_proj_in_weights[i], hyper_hidden_weights[i], hyper_proj_out_weights[i], h1[i], a1[i], h2[i], a2[i]) for i in range(4)]
    hyper = [torch.stack([p[j] for p in parts], dim=2 if j == 0 else 0) for j in range(7)]
    return iou[0], hyper[0], *iou[1:], *hyper[1:]

# problem: 075
def run(queries, keys, query_point_embedding, key_point_embedding, self_attn_q_weight, self_attn_q_bias,
        self_attn_k_weight, self_attn_k_bias, self_attn_v_weight, self_attn_v_bias, self_attn_out_weight,
        self_attn_out_bias, layer_norm1_weight, layer_norm1_bias, cross_t2i_q_weight, cross_t2i_q_bias,
        cross_t2i_k_weight, cross_t2i_k_bias, cross_t2i_v_weight, cross_t2i_v_bias, cross_t2i_out_weight,
        cross_t2i_out_bias, layer_norm2_weight, layer_norm2_bias, mlp_lin1_weight, mlp_lin1_bias,
        mlp_lin2_weight, mlp_lin2_bias, layer_norm3_weight, layer_norm3_bias, cross_i2t_q_weight,
        cross_i2t_q_bias, cross_i2t_k_weight, cross_i2t_k_bias, cross_i2t_v_weight, cross_i2t_v_bias,
        cross_i2t_out_weight, cross_i2t_out_bias, layer_norm4_weight, layer_norm4_bias, skip_first_layer_pe):
    b, pb = queries.shape[:2]
    def attention(qi, ki, vi, qw, qb, kw, kb, vw, vb, ow, ob):
        q, k, v = [_linear_f32(x, w, bias).reshape(b * pb, x.shape[-2], -1) for x, w, bias in ((qi, qw, qb), (ki, kw, kb), (vi, vw, vb))]
        a = _attention_f32(q, k, v, 8).reshape(b, pb, qi.shape[-2], -1)
        return _linear_f32(a, ow, ob)
    qi = queries if skip_first_layer_pe else _add(queries, query_point_embedding)
    a = attention(qi, qi, queries, self_attn_q_weight, self_attn_q_bias, self_attn_k_weight, self_attn_k_bias,
                  self_attn_v_weight, self_attn_v_bias, self_attn_out_weight, self_attn_out_bias)
    queries = _layernorm_f32(_add(queries, a), layer_norm1_weight, layer_norm1_bias, 1e-6)
    a = attention(_add(queries, query_point_embedding), _add(keys, key_point_embedding), keys,
                  cross_t2i_q_weight, cross_t2i_q_bias, cross_t2i_k_weight, cross_t2i_k_bias,
                  cross_t2i_v_weight, cross_t2i_v_bias, cross_t2i_out_weight, cross_t2i_out_bias)
    queries = _layernorm_f32(_add(queries, a), layer_norm2_weight, layer_norm2_bias, 1e-6)
    mlp = _linear_f32(_unary('un_ff_5', _linear_f32(queries, mlp_lin1_weight, mlp_lin1_bias)), mlp_lin2_weight, mlp_lin2_bias)
    queries = _layernorm_f32(_add(queries, mlp), layer_norm3_weight, layer_norm3_bias, 1e-6)
    a = attention(_add(keys, key_point_embedding), _add(queries, query_point_embedding), queries,
                  cross_i2t_q_weight, cross_i2t_q_bias, cross_i2t_k_weight, cross_i2t_k_bias,
                  cross_i2t_v_weight, cross_i2t_v_bias, cross_i2t_out_weight, cross_i2t_out_bias)
    return queries, _layernorm_f32(_add(keys, a), layer_norm4_weight, layer_norm4_bias, 1e-6)

# problem: 076
def run(grad_output, hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias, rel_pos_h, rel_pos_w, scale):
    b, height, width, hidden = hidden_states.shape
    seq, dim = height * width, hidden // 12
    qkv = _linear_f32(hidden_states, qkv_weight, qkv_bias).view(b, seq, 3, 12, dim)
    q, k, v = [qkv[:, :, i].transpose(1, 2).reshape(b * 12, seq, dim) for i in range(3)]
    rh, rw = _relative_table(rel_pos_h, height), _relative_table(rel_pos_w, width)
    p = _softmax_raw(_add(_matmul(_affine(q, scale), k, transpose_b=True), _relative2d_bias(q, rh, rw, height, width)))
    attn = _merge_heads(_matmul(p, v).view(b, 12, seq, dim)).view(hidden_states.shape)
    ga, gow, gob = _linear_backward(grad_output, attn, proj_weight)
    ga = ga.view(b, seq, 12, dim).transpose(1, 2).reshape(b * 12, seq, dim)
    gs = _softmax_backward(_matmul(ga, v, transpose_b=True), p)
    gv = _matmul(p, ga, transpose_a=True)
    gq, gk = _affine(_matmul(gs, k), scale), _matmul(gs, _affine(q, scale), transpose_a=True)
    gs2 = gs.view(b * 12, height, width, height, width)
    gh, gw = _sum(gs2, -1), _sum(gs2, -2)
    q2 = q.view(b * 12, height, width, dim)
    dq_h = _matmul(gh, rh.unsqueeze(0))
    dq_w = _matmul(gw.transpose(1, 2), rw.unsqueeze(0)).transpose(1, 2)
    gq = _add(gq, _add(dq_h, dq_w).reshape(gq.shape))
    grh = _sum(_matmul(gh, q2, transpose_a=True), 0)
    grw = _sum(_matmul(gw.transpose(1, 2), q2.transpose(1, 2), transpose_a=True), 0)
    gqkv = torch.stack((gq, gk, gv), dim=1).view(b, 12, 3, seq, dim).permute(0, 3, 2, 1, 4).reshape(b, height, width, 3 * hidden)
    gx, gqw, gqb = _linear_backward(gqkv, hidden_states, qkv_weight)
    return gx, gqw, gqb, gow, gob, _relative_table_backward(grh, rel_pos_h), _relative_table_backward(grw, rel_pos_w)

# problem: 077
def run(hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias, rel_pos_h, rel_pos_w,
        layer_norm1_weight, layer_norm1_bias, layer_norm2_weight, layer_norm2_bias,
        mlp_lin1_weight, mlp_lin1_bias, mlp_lin2_weight, mlp_lin2_bias, layer_norm_eps):
    b, height, width, channels = hidden_states.shape
    hp, wp = (height + 13) // 14 * 14, (width + 13) // 14 * 14
    n = _layernorm_f32(hidden_states, layer_norm1_weight, layer_norm1_bias, layer_norm_eps)
    n = _pad_axis(_pad_axis(n, 1, hp), 2, wp)
    windows = n.view(b, hp // 14, 14, wp // 14, 14, channels).permute(0, 1, 3, 2, 4, 5).reshape(-1, 196, channels)
    count = windows.shape[0]
    qkv = _linear_f32(windows, qkv_weight, qkv_bias).view(count, 196, 3, 12, 64)
    q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
    scores = _affine(_matmul(q, k, transpose_b=True), 0.125)
    bias = _relative2d_bias(q, _relative_table(rel_pos_h, 14), _relative_table(rel_pos_w, 14), 14, 14)
    a = _merge_heads(_matmul(_softmax_raw(_add(scores, bias)), v))
    a = _linear_f32(a, proj_weight, proj_bias).view(b, hp // 14, wp // 14, 14, 14, channels)
    a = a.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, channels)[:, :height, :width]
    x = _add(hidden_states, a)
    n = _layernorm_f32(x, layer_norm2_weight, layer_norm2_bias, layer_norm_eps)
    h = _unary('un_ff_2', _linear_f32(n, mlp_lin1_weight, mlp_lin1_bias))
    return _add(x, _linear_f32(h, mlp_lin2_weight, mlp_lin2_bias))
