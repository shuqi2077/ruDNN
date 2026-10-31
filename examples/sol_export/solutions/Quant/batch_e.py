# problem: 008
def run(text_input_ids, text_embedding_weight, vision_features, vision_proj_weight_fp4, vision_proj_weight_scale,
        vision_proj_weight_global_scale, vision_proj_bias, vision_token_mask):
    b, seq = text_input_ids.shape
    count, width = vision_features.shape[1], text_embedding_weight.shape[1]
    projected = _nv4_linear(vision_features, vision_proj_weight_fp4, vision_proj_weight_scale, vision_proj_weight_global_scale)
    projected = _binary('bin_bbb_0', projected, _aligned(vision_proj_bias))
    positions = torch.empty((b, seq), dtype=torch.int32, device=vision_features.device)
    _launch('nv4_vision_positions', [_aligned(vision_token_mask), positions], (b, 1, 1), {'I': [seq, count]})
    out = torch.empty((b, seq, width), dtype=torch.bfloat16, device=vision_features.device)
    _launch('nv4_vision_merge', [_aligned(text_input_ids), positions, _aligned(text_embedding_weight), projected, out],
            _blocks(out.numel() // 8, 256), {'I': [seq, count, width // 8]})
    return out

# problem: 009
def run(x, weight_fp4, weight_scale, weight_global_scale, bias):
    b, s, c = x.shape
    y = _nv4_linear(x.reshape(b, s // 2, c * 2), weight_fp4, weight_scale, weight_global_scale, round_block=True)
    return _binary('bin_bbb_0', y, _aligned(bias))

# problem: 018
def run(attn_output, residual, weight_fp4, weight_scale, weight_global_decode):
    return _binary('bin_bbb_0', _nv4_linear(attn_output, weight_fp4, weight_scale, weight_global_decode), _aligned(residual))

# problem: 019
def run(hidden_states, cos, sin, q_weight_fp4, q_scale, q_global_decode_scale, k_weight_fp4, k_scale,
        k_global_decode_scale, v_weight_fp4, v_scale, v_global_decode_scale, o_weight_fp4, o_scale, o_global_decode_scale):
    q, k, v = _nv4_qkv(hidden_states, ((q_weight_fp4, q_scale, q_global_decode_scale),
                       (k_weight_fp4, k_scale, k_global_decode_scale), (v_weight_fp4, v_scale, v_global_decode_scale)), cos, sin)
    out, _ = _attn_bf16(q, k, v, causal=True)
    return _nv4_linear(out.flatten(2), o_weight_fp4, o_scale, o_global_decode_scale)

# problem: 020
def run(x, weight_fp4, weight_scales, weight_global_decode_scale):
    return _nv4_linear(x, weight_fp4, weight_scales, weight_global_decode_scale)

# problem: 024
def run(hidden_states, weight_fp4, weight_scale_fp8, weight_global_decode_scale):
    return _nv4_linear(hidden_states, weight_fp4, weight_scale_fp8, weight_global_decode_scale, dtype=torch.float32)

# problem: 025
def run(hidden_states, q_weight_fp4, q_scale, q_global_decode_scale, k_weight_fp4, k_scale, k_global_decode_scale,
        v_weight_fp4, v_scale, v_global_decode_scale):
    activation = _nv4_quant(hidden_states)
    return tuple(_cast('cast_fh', _nv4_project(activation, w, s, g, dtype=torch.float32), torch.float16)
                 for w, s, g in ((q_weight_fp4, q_scale, q_global_decode_scale),
                                (k_weight_fp4, k_scale, k_global_decode_scale), (v_weight_fp4, v_scale, v_global_decode_scale)))

# problem: 026
def run(hidden_states, hidden_states_scale, global_decode_scale_input, weight, weight_scale, global_decode_scale_weight):
    x = _nv4_decode(hidden_states, hidden_states_scale)
    return _nv4_project((x, _nv4_scale(global_decode_scale_input, x)), weight, weight_scale, global_decode_scale_weight,
                        dtype=torch.float32)

# problem: 027
def run(hidden_states, weight_fp4, weight_scale, weight_global_scale, e_score_correction_bias):
    logits = _nv4_linear(hidden_states, weight_fp4, weight_scale, weight_global_scale, dtype=torch.float32)
    tokens, experts = logits.shape
    ids = torch.empty((tokens, 6), dtype=torch.int64, device=logits.device)
    weights = torch.empty((tokens, 6), dtype=torch.float32, device=logits.device)
    _launch('nv4_router6', [logits, _aligned(e_score_correction_bias), ids, weights], ((tokens + 7) // 8, 1, 1),
            {'I': [tokens, experts], 'f': [2.5]})
    return ids, weights

# problem: 028
def run(hidden_states, cos, sin, q_proj_weight_fp4, q_proj_scale, q_proj_global_decode_scale, k_proj_weight_fp4,
        k_proj_scale, k_proj_global_decode_scale, v_proj_weight_fp4, v_proj_scale, v_proj_global_decode_scale,
        q_norm_weight, k_norm_weight, rms_norm_eps):
    q, k, v = _nv4_qkv(hidden_states, ((q_proj_weight_fp4, q_proj_scale, q_proj_global_decode_scale),
                       (k_proj_weight_fp4, k_proj_scale, k_proj_global_decode_scale),
                       (v_proj_weight_fp4, v_proj_scale, v_proj_global_decode_scale)), cos, sin,
                       q_norm_weight, k_norm_weight, rms_norm_eps)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

# problem: 029
def run(x, gate_weight_fp4, gate_scale, gate_global_decode_scale, up_weight_fp4, up_scale, up_global_decode_scale,
        down_weight_fp4, down_scale, down_global_decode_scale):
    return _nv4_mlp(x, (gate_weight_fp4, gate_scale, gate_global_decode_scale),
                    (up_weight_fp4, up_scale, up_global_decode_scale), (down_weight_fp4, down_scale, down_global_decode_scale))

# problem: 030
def run(hidden_states, cos, sin, attention_mask, input_layernorm_weight, q_weight_fp4, q_scale, q_global_scale,
        k_weight_fp4, k_scale, k_global_scale, v_weight_fp4, v_scale, v_global_scale, o_weight_fp4, o_scale, o_global_scale,
        q_norm_weight, k_norm_weight, rms_norm_eps):
    x = _norm('norm_bbb_001001', _aligned(hidden_states), hidden_states.shape[-1], rms_norm_eps,
              weight=_aligned(input_layernorm_weight))
    q, k, v = _nv4_qkv(x, ((q_weight_fp4, q_scale, q_global_scale), (k_weight_fp4, k_scale, k_global_scale),
                           (v_weight_fp4, v_scale, v_global_scale)), cos, sin, q_norm_weight, k_norm_weight,
                       rms_norm_eps, hf_norm=True)
    out, _ = _attn_bf16(q, k, v, mask=attention_mask)
    return _nv4_linear(out.flatten(2), o_weight_fp4, o_scale, o_global_scale)

# problem: 031
def run(query_states, key_states, value_states, attention_mask):
    b, heads, seq, d = query_states.shape
    kv = key_states.shape[1]
    q = _nv4_quant(query_states.reshape(b * heads, seq, d), round_block=True)
    k = _nv4_quant(key_states.reshape(b * kv, seq, d), round_block=True)
    scores = _nv4_bmm(q, k, group=heads // kv)
    probs = _score_softmax_bf16(scores, b, heads, seq, seq, d ** -0.5, attention_mask)
    a = _nv4_quant(probs, round_block=True)
    v = _nv4_quant(value_states.reshape(b * kv, seq, d).transpose(1, 2), round_block=True)
    out = _nv4_bmm(a, v, group=heads // kv)
    return out.view(b, heads, seq, d), probs.view(b, heads, seq, seq)

# problem: 032
def run(hidden_states, gate_weight_fp4, gate_scale, gate_global_decode, up_weight_fp4, up_scale, up_global_decode,
        down_weight_fp4, down_scale, down_global_decode):
    return _nv4_mlp(hidden_states, (gate_weight_fp4, gate_scale, gate_global_decode),
                    (up_weight_fp4, up_scale, up_global_decode), (down_weight_fp4, down_scale, down_global_decode), fp32=True)

# problem: 033
def run(hidden_states, gate_weight_fp4, gate_scale, gate_global_scale):
    logits = _nv4_linear(hidden_states, gate_weight_fp4, gate_scale, gate_global_scale, dtype=torch.float32)
    tokens, experts = logits.shape
    ids = torch.empty((tokens, 10), dtype=torch.int64, device=logits.device)
    weights = torch.empty((tokens, 10), dtype=torch.bfloat16, device=logits.device)
    _launch('softmax_topk16_fb_1', [logits, logits, ids, weights], ((tokens + 7) // 8, 1, 1),
            {'I': [tokens, experts, 10], 'f': [0.0]})
    return weights, ids, logits
