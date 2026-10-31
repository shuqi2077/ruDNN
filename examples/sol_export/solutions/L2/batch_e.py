# L2 batch E: 013 016.

# problem: 013
def run(hidden_states, routing_weights, selected_experts, expert_gate_proj_weights, expert_up_proj_weights,
        expert_down_proj_weights, shared_expert_gate_proj_weight, shared_expert_up_proj_weight,
        shared_expert_down_proj_weight, shared_expert_gate_weight):
    t, h = hidden_states.shape
    dev = hidden_states.device
    experts, inter, _ = expert_gate_proj_weights.shape
    top_k = selected_experts.shape[1]
    x = _aligned(hidden_states)
    groups, sorted_tokens, inverse = _moe_route(selected_experts, experts)
    xs = _moe_gather('moe_gather_b', x, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00001', xs, expert_gate_proj_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00201', xs, expert_up_proj_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00001', act, expert_down_proj_weights, groups, h, inter)
    sg = _linear('gemm_bb_256_00_00001', x, shared_expert_gate_proj_weight)
    sa = _linear('gemm_bb_256_00_00201', x, shared_expert_up_proj_weight, residual=sg)
    logits = torch.empty((t, 8), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_00_00000', _op(x), _op(_aligned(shared_expert_gate_weight)), logits, t, 8, h)
    gate_w = torch.empty(t, dtype=torch.float32, device=dev)
    _launch('sigmoid_col_b', [logits, gate_w], _blocks(t, 256), {'I': [t, 8]})
    shared = torch.empty((t, h), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_00_00031', _op(sa), _op(_aligned(shared_expert_down_proj_weight)), shared, t, h, sa.shape[1],
          scale_a=gate_w)
    return _moe_combine('moe_combine_bb_11', y, inverse, routing_weights, t, top_k, residual=shared)

# problem: 016
def run(hidden_states, gate_weight, expert_gate_proj_weights, expert_up_proj_weights, expert_down_proj_weights):
    b, s, h = hidden_states.shape
    t = b * s
    dev = hidden_states.device
    experts, inter, _ = expert_gate_proj_weights.shape
    x = _aligned(hidden_states).view(t, h)
    logits = _linear('gemm_bb_128_00_00000', x, gate_weight)
    ids = torch.empty((t, 8), dtype=torch.int64, device=dev)
    w = torch.empty((t, 8), dtype=torch.bfloat16, device=dev)
    _launch('softmax_topk_bb_1', [logits, logits, ids, w], ((t + 7) // 8, 1, 1), {'I': [t, experts, 8], 'f': [0.0]})
    groups, sorted_tokens, inverse = _moe_route(ids, experts)
    xs = _moe_gather('moe_gather_b', x, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00001', xs, expert_gate_proj_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00201', xs, expert_up_proj_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00001', act, expert_down_proj_weights, groups, h, inter)
    out = _moe_combine('moe_combine_bb_10', y, inverse, w, t, 8)
    return out.view(b, s, h), logits
