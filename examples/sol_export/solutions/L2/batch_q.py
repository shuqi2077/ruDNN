# problem: 008
def run(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, norm_topk_prob):
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    logits = _linear('gemm_bb_256_00_00000', x, gate_weight)
    probs = _softmax_raw(logits, dtype=x.dtype)
    ids, weights = _scores_topk(probs, 8, norm_topk_prob, 1e-9)
    return _moe_forward(x, ids, weights, expert_gate_proj, expert_up_proj, expert_down_proj, True).view(shape)

# problem: 009
def run(hidden_states, cos, sin, attention_mask, input_layernorm_weight, q_proj_weight, q_proj_bias,
        k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, q_norm_weight, k_norm_weight,
        o_proj_weight, o_proj_bias, post_attention_layernorm_weight, router_weight, expert_gate_weights,
        expert_up_weights, expert_down_weights, rms_norm_eps):
    b, seq, hidden = hidden_states.shape
    n = _norm('norm_fff_001000', _aligned(hidden_states), hidden, rms_norm_eps, weight=_aligned(input_layernorm_weight))
    q, k, v = [_linear_f32(n, w, bias).view(b, seq, heads, 128) for w, bias, heads in
               ((q_proj_weight, q_proj_bias, 32), (k_proj_weight, k_proj_bias, 4), (v_proj_weight, v_proj_bias, 4))]
    q = _norm('norm_fff_001000', _aligned(q), 128, rms_norm_eps, weight=_aligned(q_norm_weight))
    k = _norm('norm_fff_001000', _aligned(k), 128, rms_norm_eps, weight=_aligned(k_norm_weight))
    q, k = _rope_forward(q.transpose(1, 2), cos, sin), _rope_forward(k.transpose(1, 2), cos, sin)
    a = _attention_f32(_merge_heads(q), _merge_heads(k), v.reshape(b, seq, -1), 32, 4, attention_mask)
    h = _add(hidden_states, _linear_f32(a, o_proj_weight, o_proj_bias))
    n = _norm('norm_fff_001000', _aligned(h), hidden, rms_norm_eps, weight=_aligned(post_attention_layernorm_weight)).view(-1, hidden)
    ids, weights = _scores_topk(_softmax_raw(_linear_f32(n, router_weight)))
    y = _moe_forward(n, ids, weights, expert_gate_weights, expert_up_weights, expert_down_weights)
    return _add(h, y.view(h.shape))

# problem: 011
def run(hidden_states, gate_weight, router_logits, routing_probs, selected_experts, routing_weights_sum,
        grad_routing_weights, grad_expert_mask, grad_router_logits):
    tokens, experts = routing_probs.shape
    gp = torch.empty_like(routing_probs)
    _launch('router_sparse_backward', [_aligned(routing_probs), _aligned(selected_experts), _aligned(routing_weights_sum),
            _aligned(grad_routing_weights), _aligned(grad_expert_mask), _aligned(grad_router_logits), gp],
            _blocks(gp.numel(), 256), {'I': [experts, selected_experts.shape[1]]})
    gs = _softmax_backward(gp, routing_probs, torch.bfloat16)
    return _matmul(gs, gate_weight), _matmul(gs, hidden_states, transpose_a=True)

# problem: 025
def run(grad_output, hidden_states, topk_indices, topk_weights, gate_weights, up_weights, down_weights):
    experts = gate_weights.shape[0]
    groups, tokens, inverse = _moe_route(topk_indices, experts)
    x = _moe_gather('moe_gather_f', hidden_states, tokens)
    go = _moe_gather('moe_gather_f', grad_output, tokens)
    sw = torch.empty(topk_weights.numel(), dtype=torch.float32, device=hidden_states.device)
    _launch('pair_permute', [_aligned(topk_weights), inverse, sw], _blocks(sw.numel(), 256))
    gate = _moe_gemm_f32(x, gate_weights, groups)
    activated = _unary('un_ff_0', gate)
    up = _moe_gemm_f32(x, up_weights, groups)
    mid = _mul(activated, up)
    y = _moe_gemm_f32(mid, down_weights, groups)
    gw_sorted = _sum(_mul(go, y), -1)
    gw = torch.empty_like(topk_weights)
    _launch('pair_unpermute', [gw_sorted, inverse, gw], _blocks(gw.numel(), 256))
    gy = _mul(go, sw[:, None])
    gi = _moe_gemm_f32(gy, down_weights.transpose(1, 2).contiguous(), groups)
    gg = _mul(_mul(gi, up), _unary('un_ff_16', gate))
    gu = _mul(gi, activated)
    gx = _add(_moe_gemm_f32(gg, gate_weights.transpose(1, 2).contiguous(), groups),
              _moe_gemm_f32(gu, up_weights.transpose(1, 2).contiguous(), groups))
    result = _ordered_combine(gx, inverse, topk_indices, torch.ones_like(topk_weights))
    dgate, dup, ddown = [torch.zeros_like(w) for w in (gate_weights, up_weights, down_weights)]
    offsets = groups[:experts + 1].cpu().tolist()
    for expert, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start != end:
            dgate[expert].copy_(_matmul(gg[start:end], x[start:end], transpose_a=True))
            dup[expert].copy_(_matmul(gu[start:end], x[start:end], transpose_a=True))
            ddown[expert].copy_(_matmul(gy[start:end], mid[start:end], transpose_a=True))
    return result, gw, dgate, dup, ddown

# problem: 080
def run(grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores,
        topk_indices, topk_weights, score_mask, shared_expert_gate_weight, shared_expert_up_weight,
        shared_expert_down_weight, shared_gate_output, shared_up_output, shared_activated):
    ga = _matmul(grad_output, shared_expert_down_weight)
    gdw = _affine(_matmul(_f32(grad_output), _f32(shared_activated), transpose_a=True), dtype=torch.bfloat16)
    gg0 = _mul(ga, shared_up_output)
    gu = _mul(ga, _unary('un_bb_0', shared_gate_output))
    gg = _affine(_mul(_f32(gg0), _unary('un_ff_16', _f32(shared_gate_output))), dtype=torch.bfloat16)
    guw = _affine(_matmul(_f32(gu), _f32(hidden_states), transpose_a=True), dtype=torch.bfloat16)
    ggw = _affine(_matmul(_f32(gg), _f32(hidden_states), transpose_a=True), dtype=torch.bfloat16)
    gx = _add(_matmul(gu, shared_expert_up_weight), _matmul(gg, shared_expert_gate_weight))
    k = topk_weights.shape[-1]
    gradw = _affine(_sum(_mul(_f32(grad_output), _f32(grad_output)), -1, True), 1.0 / k).expand_as(topk_weights)
    denom = _affine(_sum(topk_weights, -1, True), bias=1e-20)
    mean = _div(_sum(_mul(gradw, topk_weights), -1, True), denom)
    before = _div(_sub(gradw, mean), denom)
    gs = torch.empty_like(scores)
    _launch('scatter_scores_f32', [_aligned(before), _aligned(topk_indices), gs], _blocks(gs.numel(), 256), {'I': [scores.shape[-1], k]})
    gr = _mul(_mul(_mul(gs, score_mask), scores), _affine(scores, -1, 1))
    gx = _add(gx, _matmul(_affine(gr, dtype=torch.bfloat16), router_weight))
    return gx, _matmul(gr, _f32(hidden_states), transpose_a=True), ggw, guw, gdw

# problem: 082
def run(hidden_states, router_weight, e_score_correction_bias, expert_gate_projs, expert_up_projs,
        expert_down_projs, shared_gate_proj_weight, shared_up_proj_weight, shared_down_proj_weight,
        routed_scaling_factor, norm_topk_prob):
    logits = _linear_f32(_f32(hidden_states), _f32(router_weight))
    scores = _add(_unary('un_ff_3', logits), e_score_correction_bias)
    ids, weights = _scores_topk(scores, 8, norm_topk_prob, 1e-20, routed_scaling_factor)
    routed = _moe_forward(hidden_states, ids, weights, expert_gate_projs, expert_up_projs, expert_down_projs)
    gate = _linear('gemm_bb_256_00_00000', hidden_states, shared_gate_proj_weight)
    up = _linear('gemm_bb_256_00_00000', hidden_states, shared_up_proj_weight)
    mid = _mul(_unary('un_bb_0', gate), up)
    shared = _linear('gemm_bb_256_00_00000', mid, shared_down_proj_weight)
    return _add(routed, shared)
