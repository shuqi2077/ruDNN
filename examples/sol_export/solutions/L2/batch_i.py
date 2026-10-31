# L2 batch I: 029 061 (routing + grouped MoE + shared expert).

# problem: 029
def run(hidden_states, gate_weight, e_score_correction_bias, expert_gate_proj, expert_up_proj, expert_down_proj,
        shared_gate_proj, shared_up_proj, shared_down_proj, norm_min):
    b, s, h = hidden_states.shape
    t = b * s
    dev = hidden_states.device
    experts, inter, _ = expert_gate_proj.shape
    x = _aligned(hidden_states).view(t, h)
    sg = _linear('gemm_bb_256_00_00001', x, shared_gate_proj)
    sa = _linear('gemm_bb_256_00_00201', x, shared_up_proj, residual=sg)
    shared = _linear('gemm_bb_256_00_00001', sa, shared_down_proj)
    xf = torch.empty((t, h), dtype=torch.float32, device=dev)
    _launch('cast_bf', [x, xf], _blocks(t * h // 8, 256), {'I': [t * h // 8]})
    logits = torch.empty((t, experts), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_128_00_00000', xf, gate_weight, logits, t, experts, h)
    ids = torch.empty((t, 8), dtype=torch.int64, device=dev)
    w = torch.empty((t, 8), dtype=torch.bfloat16, device=dev)
    _launch('softmax_topk_fb_b', [logits, _aligned(e_score_correction_bias).view(-1), ids, w], ((t + 7) // 8, 1, 1),
            {'I': [t, experts, 8], 'f': [float(norm_min)]})
    groups, sorted_tokens, inverse = _moe_route(ids, experts)
    xs = _moe_gather('moe_gather_b', x, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00001', xs, expert_gate_proj, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00201', xs, expert_up_proj, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00001', act, expert_down_proj, groups, h, inter)
    out = _moe_combine('moe_combine_bb_11', y, inverse, w, t, 8, residual=shared)
    return out.view(b, s, h), logits

# problem: 061
def run(hidden_states, router_weight, expert_gate_proj, expert_up_proj, expert_down_proj, shared_gate_proj, shared_up_proj,
        shared_down_proj, shared_expert_gate_weight):
    b, s, h = hidden_states.shape
    t = b * s
    dev = hidden_states.device
    experts, inter, _ = expert_gate_proj.shape
    x = _aligned(hidden_states).view(t, h)
    logits = _linear('gemm_bb_256_00_00000', x, router_weight)
    ids = torch.empty((t, 10), dtype=torch.int64, device=dev)
    w = torch.empty((t, 10), dtype=torch.bfloat16, device=dev)
    _launch('softmax_topk16_bb_1', [logits, logits, ids, w], ((t + 7) // 8, 1, 1), {'I': [t, experts, 10], 'f': [0.0]})
    groups, sorted_tokens, inverse = _moe_route(ids, experts)
    xs = _moe_gather('moe_gather_b', x, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00001', xs, expert_gate_proj, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00201', xs, expert_up_proj, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00001', act, expert_down_proj, groups, h, inter)
    sg = _linear('gemm_bb_256_00_00001', x, shared_gate_proj)
    sa = _linear('gemm_bb_256_00_00201', x, shared_up_proj, residual=sg)
    gl = torch.empty((t, 8), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_00_00000', _op(x), _op(_aligned(shared_expert_gate_weight)), gl, t, 8, h)
    gate_w = torch.empty(t, dtype=torch.float32, device=dev)
    _launch('sigmoid_col_b', [gl, gate_w], _blocks(t, 256), {'I': [t, 8]})
    shared = torch.empty((t, h), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_00_00031', _op(sa), _op(_aligned(shared_down_proj)), shared, t, h, sa.shape[1], scale_a=gate_w)
    out = _moe_combine('moe_combine_bb_11', y, inverse, w, t, 10, residual=shared)
    return out.view(b, s, h), logits
