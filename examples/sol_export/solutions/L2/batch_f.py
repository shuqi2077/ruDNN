# L2 batch F: 024 026 (grouped tcgen05 MoE; f32-accumulated references).

# problem: 024
def run(hidden_states, topk_indices, topk_weights, gate_proj_weights, up_proj_weights, down_proj_weights):
    t, h = hidden_states.shape
    experts, inter, _ = gate_proj_weights.shape
    top_k = topk_indices.shape[1]
    groups, sorted_tokens, inverse = _moe_route(topk_indices, experts)
    xs = _moe_gather('moe_gather_b', hidden_states, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00000', xs, gate_proj_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00200', xs, up_proj_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00000', act, down_proj_weights, groups, h, inter)
    return _moe_combine('moe_combine_bb_00', y, inverse, topk_weights, t, top_k)

# problem: 026
def run(hidden_states, routing_weights, selected_experts, gate_proj_weights, up_proj_weights, down_proj_weights):
    b, s, h = hidden_states.shape
    t = b * s
    experts, inter, _ = gate_proj_weights.shape
    top_k = selected_experts.shape[1]
    x = _aligned(hidden_states).view(t, h)
    groups, sorted_tokens, inverse = _moe_route(selected_experts, experts)
    xs = _moe_gather('moe_gather_b', x, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00000', xs, gate_proj_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00200', xs, up_proj_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00000', act, down_proj_weights, groups, h, inter)
    return _moe_combine('moe_combine_bf_10', y, inverse, routing_weights, t, top_k).view(b, s, h)

# problem: 048
def run(hidden_states, topk_idx, topk_weight, gate_weights, up_weights, down_weights):
    t, h = hidden_states.shape
    experts, inter, _ = gate_weights.shape
    top_k = topk_idx.shape[1]
    groups, sorted_tokens, inverse = _moe_route(topk_idx, experts)
    xs = _moe_gather('moe_gather_b', hidden_states, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00000', xs, gate_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00200', xs, up_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00000', act, down_weights, groups, h, inter)
    return _moe_combine('moe_combine_bb_00', y, inverse, topk_weight, t, top_k)

# problem: 081
def run(hidden_states, router_weight, expert_gate_weights, expert_up_weights, expert_down_weights, shared_gate_weight,
        shared_up_weight, shared_down_weight, e_score_correction_bias, routed_scaling_factor):
    t, h = hidden_states.shape
    dev = hidden_states.device
    experts, inter, _ = expert_gate_weights.shape
    x = _aligned(hidden_states)
    logits = torch.empty((t, experts), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_128_00_00000', _op(x), _op(_aligned(router_weight)), logits, t, experts, h)
    ids = torch.empty((t, 8), dtype=torch.int64, device=dev)
    w = torch.empty((t, 8), dtype=torch.bfloat16, device=dev)
    _launch('sigmoid_topk_b_1', [logits, _aligned(e_score_correction_bias), ids, w], ((t + 7) // 8, 1, 1),
            {'I': [t, experts, 8], 'f': [float(routed_scaling_factor)]})
    groups, sorted_tokens, inverse = _moe_route(ids, experts)
    xs = _moe_gather('moe_gather_b', x, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00001', xs, expert_gate_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00201', xs, expert_up_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00001', act, expert_down_weights, groups, h, inter)
    sg = _linear('gemm_bb_256_00_00001', x, shared_gate_weight)
    sa = _linear('gemm_bb_256_00_00201', x, shared_up_weight, residual=sg)
    shared = _linear('gemm_bb_256_00_00001', sa, shared_down_weight)
    return _moe_combine('moe_combine_bb_11', y, inverse, w, t, 8, residual=shared)
