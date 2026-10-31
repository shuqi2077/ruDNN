# problem: 014
def run(grad_output, hidden_states, query_states, key_states, value_states, cu_seqlens,
        q_weight, k_weight, v_weight, out_weight):
    length, hidden = hidden_states.shape
    heads, dim = query_states.shape[1], query_states.shape[-1]
    go, x = _f32(grad_output), _f32(hidden_states)
    ga = _matmul(go, _f32(out_weight)).view(length, heads, dim).transpose(0, 1).unsqueeze(0)
    attn, gq, gk, gv = _varlen_backward(ga, _f32(query_states), _f32(key_states), _f32(value_states), cu_seqlens, dim ** -0.5, True)
    gow = _matmul(go, _merge_heads(attn).reshape(length, hidden), transpose_a=True)
    gob = _sum(go, 0)
    grads = [_linear_backward(_merge_heads(g).reshape(length, hidden), x, _f32(w)) for g, w in ((gq, q_weight), (gk, k_weight), (gv, v_weight))]
    gx = _add(_add(grads[0][0], grads[1][0]), grads[2][0])
    return tuple(_affine(t, dtype=torch.bfloat16) for t in (gx, grads[0][1], grads[0][2], grads[1][1], grads[1][2], grads[2][1], grads[2][2], gow, gob))

# problem: 017
def run(grad_output, hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias, cos, sin,
        cu_seqlens, attention_dropout, scaling):
    length, hidden = hidden_states.shape
    qkv = _linear_f32(hidden_states, qkv_weight, qkv_bias).view(length, 3, 28, 128)
    q, k, v = [qkv[:, i].transpose(0, 1).unsqueeze(0) for i in range(3)]
    ct, st = cos.unsqueeze(0), sin.unsqueeze(0)
    q, k = _rope_forward(q, ct, st), _rope_forward(k, ct, st)
    ga = _matmul(grad_output, proj_weight).view(length, 28, 128).transpose(0, 1).unsqueeze(0)
    attn, gq, gk, gv = _varlen_backward(ga, q, k, v, cu_seqlens, scaling)
    gow = _matmul(grad_output, _merge_heads(attn).reshape(length, hidden), transpose_a=True)
    gob = _sum(grad_output, 0)
    gq, gk = _rope_backward(gq, ct, st), _rope_backward(gk, ct, st)
    gqkv = torch.cat(tuple(_merge_heads(t).reshape(length, hidden) for t in (gq, gk, gv)), dim=-1)
    gx, gqw, gqb = _linear_backward(gqkv, hidden_states, qkv_weight)
    return gx, gqw, gqb, gow, gob

# problem: 041
def run(hidden_states, position_ids, attention_mask, q_proj_weight, k_proj_weight, v_proj_weight,
        o_proj_weight, q_norm_weight, k_norm_weight, shared_key_states, shared_value_states,
        rope_theta, softcap, rms_norm_eps, use_shared_kv):
    b, seq, hidden = hidden_states.shape
    cos, sin = [torch.empty((b, seq, 96), dtype=torch.bfloat16, device=hidden_states.device) for _ in range(2)]
    _launch('position_rope_table_b', [_aligned(position_ids), cos, sin], _blocks(cos.numel(), 256), {'I': [96], 'f': [float(rope_theta)]})
    q = _linear('gemm_bb_256_00_00000', hidden_states, q_proj_weight).view(b, seq, 32, 96)
    q = _norm('norm_bbb_001000', _aligned(q), 96, rms_norm_eps, weight=_aligned(q_norm_weight))
    q = _rope_forward(q.transpose(1, 2), cos, sin)
    if use_shared_kv:
        k, v = shared_key_states, shared_value_states
    else:
        k = _linear('gemm_bb_256_00_00000', hidden_states, k_proj_weight).view(b, seq, 8, 96)
        k = _norm('norm_bbb_001000', _aligned(k), 96, rms_norm_eps, weight=_aligned(k_norm_weight))
        k = _rope_forward(k.transpose(1, 2), cos, sin)
        v = _linear('gemm_bb_256_00_00000', hidden_states, v_proj_weight).view(b, seq, 8, 96)
        v = _norm('norm_bbb_000000', _aligned(v), 96, rms_norm_eps).transpose(1, 2)
    k, v = _repeat_heads(k, 32), _repeat_heads(v, 32)
    scores = _affine(_matmul(q, k, transpose_b=True), 1.0 / softcap)
    scores = _affine(_unary('un_bb_4', scores), softcap)
    scores = _add(scores, attention_mask)
    p = _softmax_raw(scores, dtype=torch.bfloat16)
    return _linear('gemm_bb_256_00_00000', _merge_heads(_matmul(p, v)), o_proj_weight)

# problem: 045
def run(input_features, lm_embeddings, audio_token_positions, encoder_input_weight, encoder_input_bias,
        encoder_out_weight, encoder_out_bias, encoder_out_mid_weight, encoder_out_mid_bias, learnable_queries,
        qformer_q_proj_weight, qformer_q_proj_bias, qformer_k_proj_weight, qformer_k_proj_bias,
        qformer_v_proj_weight, qformer_v_proj_bias, qformer_out_proj_weight, qformer_out_proj_bias,
        projector_weight, projector_bias):
    b, seq = input_features.shape[:2]
    h = _linear_f32(_f32(input_features), _f32(encoder_input_weight), _f32(encoder_input_bias))
    p = _softmax_raw(_linear_f32(h, _f32(encoder_out_weight), _f32(encoder_out_bias)))
    h = _add(h, _linear_f32(p, _f32(encoder_out_mid_weight), _f32(encoder_out_mid_bias)))
    blocks = (seq + 14) // 15
    h = _pad_axis(h, 1, blocks * 15).view(b * blocks, 15, 512)
    queries = _f32(learnable_queries).expand(b * blocks, -1, -1)
    q = _linear_f32(queries, _f32(qformer_q_proj_weight), _f32(qformer_q_proj_bias))
    k = _linear_f32(h, _f32(qformer_k_proj_weight), _f32(qformer_k_proj_bias))
    v = _linear_f32(h, _f32(qformer_v_proj_weight), _f32(qformer_v_proj_bias))
    a = _attention_f32(q, k, v, 16)
    a = _linear_f32(a, _f32(qformer_out_proj_weight), _f32(qformer_out_proj_bias)).view(b, blocks * 40, 1024)
    audio = _linear_f32(a, _f32(projector_weight), _f32(projector_bias))
    out = _f32(lm_embeddings).clone()
    count = audio_token_positions.shape[1]
    src = _aligned(audio[:, :count])
    for bi in range(b):
        _launch('scatterseq_f', [src[bi], _aligned(audio_token_positions[bi]), out[bi]], _blocks(src[bi].numel() // 8, 256),
                {'I': [src[bi].numel() // 8, count, out.shape[1], 4096 // 8]})
    return _affine(out, dtype=torch.bfloat16)

# problem: 058
def run(hidden_states, in_proj_weight, conv1d_weight, conv1d_bias, dt_bias, A_log, D, norm_weight,
        out_proj_weight, time_step_limit_min, time_step_limit_max, layer_norm_epsilon):
    batch, length, _ = hidden_states.shape
    projected = _linear('gemm_bb_256_00_00000', hidden_states, in_proj_weight)
    start = projected.shape[-1] - 16384 - 20480 - 256
    gate = projected[..., start:start + 16384]
    h = _aligned(projected[..., start + 16384:start + 16384 + 20480].transpose(1, 2))
    conv = torch.empty_like(h)
    _launch('depthwise1d_bf16', [h, _aligned(conv1d_weight), _aligned(conv1d_bias), conv],
            _blocks(conv.numel(), 256), {'I': [length, length, 20480, 4, 3]})
    h = _mul(conv, _unary('un_bb_3', conv)).transpose(1, 2)
    x = _f32(h[..., :16384].reshape(batch, length, 256, 64))
    b = _f32(h[..., 16384:18432].reshape(batch, length, 8, 256))
    c = _f32(h[..., 18432:].reshape(batch, length, 8, 256))
    b, c = [t.unsqueeze(2).expand(batch, length, 32, 8, 256).reshape(batch, length, 256, 256) for t in (b, c)]
    dt = _clamp(_unary('un_bb_21', _add(projected[..., -256:], dt_bias)), time_step_limit_min, time_step_limit_max)
    times = _f32(dt)
    a = _mul(_affine(_unary('un_ff_6', _f32(A_log)), -1), times)
    initial = torch.zeros((batch, 256, 64, 256), dtype=torch.float32, device=hidden_states.device)
    y, _ = _ssm_chunks(_mul(x, times.unsqueeze(-1)), a, b, c, initial, 128)
    y = _add(y, _mul(x, _f32(D).view(1, 1, 256, 1)))
    y = _affine(y.reshape(batch, length, 16384), dtype=torch.bfloat16)
    y = _norm('norm_bbb_000001', _aligned(y), 2048, layer_norm_epsilon)
    y = _mul(y, norm_weight)
    y = _mul(y, _mul(gate, _unary('un_bb_3', gate)))
    return _linear('gemm_bb_256_00_00000', y, out_proj_weight)

# problem: 078
def run(x, timestep_emb, time_emb_mlp_linear_weight, time_emb_mlp_linear_bias, resblock_in_norm_weight,
        resblock_in_norm_bias, resblock_in_conv_weight, resblock_in_conv_bias, resblock_emb_linear_weight,
        resblock_emb_linear_bias, resblock_out_norm_weight, resblock_out_norm_bias, resblock_out_conv_weight,
        resblock_out_conv_bias, resblock_skip_conv_weight, resblock_skip_conv_bias, final_norm_weight,
        final_norm_bias, final_conv_weight, final_conv_bias, eps):
    batch, seq, hidden = x.shape
    height = math.isqrt(seq)
    while seq % height:
        height -= 1
    spatial = x.view(batch, height, seq // height, hidden).permute(0, 3, 1, 2).contiguous()
    h = _unary('un_bb_0', _group_norm_bf16(spatial, resblock_in_norm_weight, resblock_in_norm_bias, 32, eps))
    h = _conv_nd(h, resblock_in_conv_weight, resblock_in_conv_bias, pad=1)
    emb = _linear('gemm_bb_256_00_10000', _unary('un_bb_0', timestep_emb), resblock_emb_linear_weight, bias=_aligned(resblock_emb_linear_bias))
    scale, shift = emb[:, :, None, None].chunk(2, dim=1)
    h = _group_norm_bf16(h, resblock_out_norm_weight, resblock_out_norm_bias, 32, eps)
    h = _unary('un_bb_0', _add(_mul(h, _affine(scale, bias=1)), shift))
    h = _conv_nd(h, resblock_out_conv_weight, resblock_out_conv_bias, pad=1)
    h = _add(_conv_nd(spatial, resblock_skip_conv_weight, resblock_skip_conv_bias), h)
    h = _unary('un_bb_0', _group_norm_bf16(h, final_norm_weight, final_norm_bias, 32, eps))
    return _conv_nd(h, final_conv_weight, final_conv_bias, pad=1)
