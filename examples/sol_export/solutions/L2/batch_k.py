# problem: 002
def run(hidden_states, cos, sin, attention_mask, input_layernorm_weight, q_proj_weight, k_proj_weight, v_proj_weight,
        o_proj_weight, post_attention_layernorm_weight, gate_proj_weight, up_proj_weight, down_proj_weight, rms_norm_eps):
    b, seq, hidden = hidden_states.shape
    x = _norm('norm_fff_001000', _aligned(hidden_states), hidden, rms_norm_eps, weight=_aligned(input_layernorm_weight))
    q, k, v = [_linear_f32(x, w) for w in (q_proj_weight, k_proj_weight, v_proj_weight)]
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_fff_128_0100', q, qr, b * seq * 32, 32, 1, cos=_aligned(cos), sin=_aligned(sin))
    _hnr('hnr_fff_128_0100', k, kr, b * seq * 8, 8, 1, cos=_aligned(cos), sin=_aligned(sin))
    attn = _attention_f32(qr, kr, v, 32, 8, mask=attention_mask)
    h = _binary('bin_fff_0', _linear_f32(attn, o_proj_weight), hidden_states)
    n = _norm('norm_fff_001000', h, hidden, rms_norm_eps, weight=_aligned(post_attention_layernorm_weight))
    gate = _unary('un_ff_0', _linear_f32(n, gate_proj_weight))
    up = _linear_f32(n, up_proj_weight)
    return _binary('bin_fff_0', h, _linear_f32(_binary('bin_fff_2', gate, up), down_proj_weight))

# problem: 018
def run(hidden_states, cu_seqlens, cos, sin, qkv_weight, qkv_bias, proj_weight, proj_bias):
    total = hidden_states.shape[0]
    qkv = _linear('gemm_bb_256_00_10000', hidden_states, qkv_weight, bias=_aligned(qkv_bias)).view(total, 3, 16, 72)
    q, k, v = [_aligned(qkv[:, i]).unsqueeze(0) for i in range(3)]
    q, k = _rope_pair_bf16(q, k, cos, sin, 72)
    joined = torch.empty((total, 1152), dtype=torch.bfloat16, device=hidden_states.device)
    ends = [0] + cu_seqlens.cpu().tolist()
    for begin, end in zip(ends, ends[1:]):
        if end > begin:
            result, _ = _attn_bf16(q[:, begin:end], k[:, begin:end], v[:, begin:end])
            _launch('cast_bb', [result, joined[begin:end]], _blocks(result.numel() // 8, 256), {'I': [result.numel() // 8]})
    return _linear('gemm_bb_256_00_10000', joined, proj_weight, bias=_aligned(proj_bias))

# problem: 054
def run(hidden_state, input_layernorm_weight, input_layernorm_bias, q_proj_weight, k_proj_weight, v_proj_weight,
        o_proj_weight, post_attention_layernorm_weight, post_attention_layernorm_bias, fc1_weight, fc1_bias,
        fc2_weight, fc2_bias, gate_attn, gate_ffn, norm_eps):
    b, seq, hidden = hidden_state.shape
    def norm(x, w, bias):
        n = _norm('norm_bbb_100000', _aligned(x), hidden, norm_eps)
        return _binary('bin_bbb_0', _binary('bin_bbb_2', n, _aligned(w)), _aligned(bias))
    x = norm(hidden_state, input_layernorm_weight, input_layernorm_bias)
    q, k, v = [_linear('gemm_bb_256_00_00000', x, w).view(b, seq, 16, hidden // 16)
               for w in (q_proj_weight, k_proj_weight, v_proj_weight)]
    attn, _ = _attn_bf16(q, k, v)
    attn = _linear('gemm_bb_256_00_00000', attn.flatten(2), o_proj_weight)
    h = _binary('bin_bbb_0', hidden_state, _binary('bin_bbb_2', _unary('un_bb_4', gate_attn), attn))
    n = norm(h, post_attention_layernorm_weight, post_attention_layernorm_bias)
    ff = _linear('gemm_bb_256_00_12002', n, fc1_weight, bias=_aligned(fc1_bias))
    ff = _linear('gemm_bb_256_00_10002', ff, fc2_weight, bias=_aligned(fc2_bias))
    return _binary('bin_bbb_0', h, _binary('bin_bbb_2', _unary('un_bb_4', gate_ffn), ff))

# problem: 062
def run(hidden_states, encoder_hidden_states, self_attn_norm_weight, self_attn_q_weight, self_attn_k_weight,
        self_attn_v_weight, self_attn_o_weight, cross_attn_norm_weight, cross_attn_q_weight, cross_attn_k_weight,
        cross_attn_v_weight, cross_attn_o_weight, mlp_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight,
        norm_eps, rope_theta):
    b, seq, hidden = hidden_states.shape
    x = _norm('norm_bbb_001000', _aligned(hidden_states), hidden, norm_eps, weight=_aligned(self_attn_norm_weight))
    q, k, v = [_linear('gemm_bb_256_00_00000', x, w).view(b, seq, -1, 128)
               for w in (self_attn_q_weight, self_attn_k_weight, self_attn_v_weight)]
    cos = torch.empty((b, seq, 128), dtype=torch.bfloat16, device=x.device)
    sin = torch.empty_like(cos)
    _launch('sequence_rope_b', [cos, sin], _blocks(cos.numel(), 256), {'I': [seq, 128], 'f': [float(rope_theta)]})
    q, k = _rope_pair_bf16(q, k, cos, sin)
    attn, _ = _attn_bf16(q, k, v, causal=True)
    h = _linear('gemm_bb_256_00_00101', attn.flatten(2), self_attn_o_weight, residual=hidden_states)
    x = _norm('norm_bbb_001000', h, hidden, norm_eps, weight=_aligned(cross_attn_norm_weight))
    q = _linear('gemm_bb_256_00_00000', x, cross_attn_q_weight).view(b, seq, 16, 128)
    k, v = [_linear('gemm_bb_256_00_00000', encoder_hidden_states, w).view(b, -1, 16, 128)
            for w in (cross_attn_k_weight, cross_attn_v_weight)]
    attn, _ = _attn_bf16(q, k, v)
    h = _linear('gemm_bb_256_00_00101', attn.flatten(2), cross_attn_o_weight, residual=h)
    n = _norm('norm_bbb_001000', h, hidden, norm_eps, weight=_aligned(mlp_norm_weight))
    gate = _linear('gemm_bb_256_00_00001', n, mlp_gate_weight)
    act = _linear('gemm_bb_256_00_00201', n, mlp_up_weight, residual=gate)
    return _linear('gemm_bb_256_00_00101', act, mlp_down_weight, residual=h)

# problem: 069
def run(hidden_states, encoder_hidden_states, temb, norm1_weight, norm1_bias, norm1_context_weight, norm1_context_bias,
        to_q_weight, to_q_bias, to_k_weight, to_k_bias, to_v_weight, to_v_bias, add_q_proj_weight, add_q_proj_bias,
        add_k_proj_weight, add_k_proj_bias, add_v_proj_weight, add_v_proj_bias, to_out_weight, to_out_bias,
        to_add_out_weight, to_add_out_bias, ff_linear1_weight, ff_linear1_bias, ff_linear2_weight, ff_linear2_bias,
        ff_context_linear1_weight, ff_context_linear1_bias, ff_context_linear2_weight, ff_context_linear2_bias):
    b, image_seq, dim = hidden_states.shape
    context_seq, context_dim = encoder_hidden_states.shape[1:]
    t = _unary('un_ff_0', temb)
    mod = _linear_f32(t, norm1_weight, norm1_bias)
    cmod = _linear_f32(t, norm1_context_weight, norm1_context_bias)
    def modulate(x, m, shift, scale):
        out = torch.empty_like(x)
        _launch('adaln_fff', [_aligned(x), m, out], (x.shape[0] * x.shape[1], 1, 1),
                {'I': [x.shape[2], x.shape[1], m.shape[1], shift * x.shape[2], scale * x.shape[2]], 'f': [1e-6]})
        return out
    h, c = modulate(hidden_states, mod, 0, 1), modulate(encoder_hidden_states, cmod, 0, 1)
    q, k, v = [torch.cat((_linear_f32(h, iw, ib), _linear_f32(c, cw, cb)), dim=1)
               for iw, ib, cw, cb in ((to_q_weight, to_q_bias, add_q_proj_weight, add_q_proj_bias),
                                      (to_k_weight, to_k_bias, add_k_proj_weight, add_k_proj_bias),
                                      (to_v_weight, to_v_bias, add_v_proj_weight, add_v_proj_bias))]
    a = _attention_f32(q, k, v, 24)
    h = _binary('bin_fff_0', hidden_states, _binary('bin_fff_2', mod[:, 2 * dim:3 * dim].unsqueeze(1),
                _linear_f32(a[:, :image_seq], to_out_weight, to_out_bias)))
    c = _binary('bin_fff_0', encoder_hidden_states, _binary('bin_fff_2', cmod[:, 2 * context_dim:3 * context_dim].unsqueeze(1),
                _linear_f32(a[:, image_seq:], to_add_out_weight, to_add_out_bias)))
    hf = _linear_f32(_unary('un_ff_1', _linear_f32(modulate(h, mod, 3, 4), ff_linear1_weight, ff_linear1_bias)),
                     ff_linear2_weight, ff_linear2_bias)
    cf = _linear_f32(_unary('un_ff_1', _linear_f32(modulate(c, cmod, 3, 4), ff_context_linear1_weight, ff_context_linear1_bias)),
                     ff_context_linear2_weight, ff_context_linear2_bias)
    return (_binary('bin_fff_0', c, _binary('bin_fff_2', cmod[:, 5 * context_dim:].unsqueeze(1), cf)),
            _binary('bin_fff_0', h, _binary('bin_fff_2', mod[:, 5 * dim:].unsqueeze(1), hf)))

# problem: 070
def run(hidden_states, encoder_hidden_states, norm1_weight, norm1_bias, attn1_to_q_weight, attn1_to_k_weight,
        attn1_to_v_weight, attn1_to_out_weight, attn1_to_out_bias, norm2_weight, norm2_bias, attn2_to_q_weight,
        attn2_to_k_weight, attn2_to_v_weight, attn2_to_out_weight, attn2_to_out_bias, norm3_weight, norm3_bias,
        ff_linear1_weight, ff_linear1_bias, ff_linear2_weight, ff_linear2_bias, norm_eps):
    h = hidden_states
    for nw, nb, qw, kw, vw, ow, ob, cross in (
            (norm1_weight, norm1_bias, attn1_to_q_weight, attn1_to_k_weight, attn1_to_v_weight,
             attn1_to_out_weight, attn1_to_out_bias, False),
            (norm2_weight, norm2_bias, attn2_to_q_weight, attn2_to_k_weight, attn2_to_v_weight,
             attn2_to_out_weight, attn2_to_out_bias, True)):
        n = _norm('norm_fff_101100', _aligned(h), h.shape[-1], norm_eps, weight=_aligned(nw), bias=_aligned(nb))
        source = encoder_hidden_states if cross else n
        a = _attention_f32(_linear_f32(n, qw), _linear_f32(source, kw), _linear_f32(source, vw), 160)
        h = _binary('bin_fff_0', h, _linear_f32(a, ow, ob))
    n = _norm('norm_fff_101100', h, h.shape[-1], norm_eps, weight=_aligned(norm3_weight), bias=_aligned(norm3_bias))
    both = _linear_f32(n, ff_linear1_weight, ff_linear1_bias)
    left, gate = both.chunk(2, dim=-1)
    f = _binary('bin_fff_2', left, _unary('un_ff_1', gate))
    return _binary('bin_fff_0', h, _linear_f32(f, ff_linear2_weight, ff_linear2_bias))

# problem: 073
def run(grad_output, hidden_states, weight1, bias1, weight2, bias2, intermediate, intermediate_activated):
    g = _aligned(grad_output).reshape(-1, grad_output.shape[-1])
    a = _aligned(intermediate_activated).reshape(g.shape[0], -1)
    x = _aligned(hidden_states).reshape(g.shape[0], -1)
    dw2 = _matmul_f32(g, a, transpose_a=True)
    db2 = _sum_columns_f32(g)
    gi = _matmul_f32(g, _aligned(weight2), combine=_aligned(intermediate))
    dx = _matmul_f32(gi, _aligned(weight1)).view(hidden_states.shape)
    dw1 = _matmul_f32(gi, x, transpose_a=True)
    db1 = _sum_columns_f32(gi)
    return dx, dw1, db1, dw2, db2
