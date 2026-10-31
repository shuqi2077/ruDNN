# L2 batch C: 005 010 012 049.

# problem: 005
def run(grad_output, x, gate_output, up_output, activated_gate, gate_weight, up_weight, down_weight):
    b, s, h = grad_output.shape
    inter = gate_output.shape[-1]
    rows = b * s
    dev = grad_output.device
    go = _aligned(grad_output).view(rows, h)
    ggo = torch.empty((rows, inter), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_01_00001', _op(go), _op(_aligned(down_weight)), ggo, rows, inter, h)
    gated = torch.empty_like(ggo)
    gup = torch.empty_like(ggo)
    ggate = torch.empty_like(ggo)
    _launch('l2_005_ew', [ggo, _aligned(up_output), _aligned(activated_gate), _aligned(gate_output), gated, gup, ggate],
            _blocks(ggo.numel(), 256), {'I': [ggo.numel()]})
    grad_down = torch.empty((h, inter), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(go), _op(gated), grad_down, h, inter, rows)
    x2 = _aligned(x).view(rows, h)
    grad_gate_w = torch.empty((inter, h), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(ggate), _op(x2), grad_gate_w, inter, h, rows)
    grad_up_w = torch.empty((inter, h), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(gup), _op(x2), grad_up_w, inter, h, rows)
    gx = torch.empty((b, s, h), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_01_00001', _op(ggate), _op(_aligned(gate_weight)), gx, rows, h, inter)
    _gemm('gemm_bb_256_01_00101', _op(gup), _op(_aligned(up_weight)), gx, rows, h, inter, residual=gx)
    return gx, grad_gate_w, grad_up_w, grad_down

# problem: 010
def run(hidden_states, routing_weights, selected_experts, gate_proj_weights, up_proj_weights, down_proj_weights):
    t, h = hidden_states.shape
    experts, inter, _ = gate_proj_weights.shape
    top_k = selected_experts.shape[1]
    groups, sorted_tokens, inverse = _moe_route(selected_experts, experts)
    xs = _moe_gather('moe_gather_b', hidden_states, sorted_tokens)
    gate = _moe_gemm('gemm_bb_256_00g_00000', xs, gate_proj_weights, groups, inter, h)
    act = _moe_gemm('gemm_bb_256_00g_00200', xs, up_proj_weights, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_bb_256_00g_00000', act, down_proj_weights, groups, h, inter)
    return _moe_combine('moe_combine_bf_10', y, inverse, routing_weights, t, top_k)

# problem: 012
def run(hidden_states, selected_experts, routing_weights, expert_gate_weights, expert_up_weights, expert_down_weights):
    t, h = hidden_states.shape
    experts, _, inter = expert_gate_weights.shape
    top_k = selected_experts.shape[1]
    dev = hidden_states.device
    capacity = max(int((t * top_k / experts) * 1.25), 1)
    pairs = t * top_k
    keys = torch.empty(pairs, dtype=torch.int32, device=dev)
    _launch('narrow_ids', [_aligned(selected_experts), keys], _blocks(pairs, 256), {'I': [pairs]})
    chunks = (pairs + 1023) // 1024
    hist = torch.empty((chunks, 256), dtype=torch.int32, device=dev)
    _launch('sort_hist', [keys, hist], (chunks, 1, 1), {'I': [pairs]})
    base = torch.empty_like(hist)
    offsets = torch.empty(257, dtype=torch.int32, device=dev)
    _launch('sort_scan', [hist, base, offsets], (1, 1, 1), {'I': [chunks]})
    order = torch.empty(pairs, dtype=torch.int32, device=dev)
    _launch('sort_scatter', [keys, base, order], (chunks, 1, 1), {'I': [pairs]})
    groups = torch.empty(2 * experts + 2, dtype=torch.int32, device=dev)
    _launch('cap_offsets', [offsets, groups], (1, 1, 1), {'I': [experts, capacity]})
    tokens = torch.zeros(pairs, dtype=torch.int32, device=dev)
    inverse = torch.empty(pairs, dtype=torch.int32, device=dev)
    _launch('cap_place', [order, offsets, groups, tokens, inverse], _blocks(pairs, 256), {'I': [pairs, experts, capacity, top_k]})
    rows = min(pairs, experts * capacity)
    xs = _moe_gather('moe_gather_b', hidden_states, tokens[:rows])
    def grouped(name, a, w, n, k, residual=None):
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=dev)
        _gemm(name, _op(a), _opb(_aligned(w)), out, rows, n, k, batches=experts, groups=groups,
              max_tiles=(rows + 127) // 128 + experts, residual=residual)
        return out
    gate = grouped('gemm_bb_256_01g_00001', xs, expert_gate_weights, inter, h)
    act = grouped('gemm_bb_256_01g_00201', xs, expert_up_weights, inter, h, residual=gate)
    y = grouped('gemm_bb_256_01g_00001', act, expert_down_weights, h, inter)
    return _moe_combine('moe_combine_bb_10', y, inverse, routing_weights, t, top_k)

# problem: 049
def run(hidden_states, weight, expert_bias, routed_scaling_factor):
    t = hidden_states.shape[0]
    dev = hidden_states.device
    logits = torch.empty((t, 256), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_00_00000', _op(_aligned(hidden_states)), _op(_aligned(weight)), logits, t, 256, hidden_states.shape[1])
    bias = torch.empty(256, dtype=torch.float32, device=dev)
    _launch('cast_bf', [_aligned(expert_bias), bias], (1, 1, 1), {'I': [32]})
    idx = torch.empty((t, 8), dtype=torch.int64, device=dev)
    w = torch.empty((t, 8), dtype=torch.float32, device=dev)
    _launch('l2_049_route', [logits, bias, idx, w], ((t + 7) // 8, 1, 1), {'I': [t], 'f': [float(routed_scaling_factor)]})
    return idx, w
