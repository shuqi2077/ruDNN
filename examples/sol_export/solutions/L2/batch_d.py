# L2 batch D: 047 079.

# problem: 047
def run(hidden_states, topk_idx, topk_weight, expert_gate_projs, expert_up_projs, expert_down_projs):
    t, h = hidden_states.shape
    experts, inter, _ = expert_gate_projs.shape
    top_k = topk_idx.shape[1]
    groups, sorted_tokens, inverse = _moe_route(topk_idx, experts)
    xs = _moe_gather('moe_gather_h', hidden_states, sorted_tokens)
    gate = _moe_gemm('gemm_hh_256_00g_00001', xs, expert_gate_projs, groups, inter, h)
    act = _moe_gemm('gemm_hh_256_00g_00201', xs, expert_up_projs, groups, inter, h, residual=gate)
    y = _moe_gemm('gemm_hh_256_00g_00001', act, expert_down_projs, groups, h, inter)
    return _moe_combine('moe_combine_hh_10', y, inverse, topk_weight, t, top_k)

# problem: 079
def run(hidden_states, layer_norm1_weight, layer_norm1_bias, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
        v_proj_weight, v_proj_bias, out_proj_weight, out_proj_bias, layer_norm2_weight, layer_norm2_bias,
        fc1_weight, fc1_bias, fc2_weight, fc2_bias, layer_norm_eps):
    b, s, d = hidden_states.shape
    rows = b * s
    dev = hidden_states.device
    x = _aligned(hidden_states).view(rows, d)
    h1 = _norm('norm_fff_101100', x, d, layer_norm_eps, weight=_aligned(layer_norm1_weight), bias=_aligned(layer_norm1_bias))
    xa = _tf32x3(h1, True)
    proj = []
    for w, bias in ((q_proj_weight, q_proj_bias), (k_proj_weight, k_proj_bias), (v_proj_weight, v_proj_bias)):
        o = torch.empty((rows, d), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_10000', xa, _tf32x3(w, False), o, rows, d, d, k_groups=3, bias=_aligned(bias))
        proj.append(o)
    heads = 12
    attn = _attn_f32(proj[0], proj[1], proj[2], b, s, s, heads, heads, d // heads, scale=(d // heads) ** -0.5)
    h2 = torch.empty((rows, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10100', attn.view(rows, d), out_proj_weight, h2, rows, d, d, bias=_aligned(out_proj_bias), residual=x)
    h3 = _norm('norm_fff_101100', h2, d, layer_norm_eps, weight=_aligned(layer_norm2_weight), bias=_aligned(layer_norm2_bias))
    inter = fc1_weight.shape[0]
    f1 = torch.empty((rows, inter), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_17000', h3, fc1_weight, f1, rows, inter, d, bias=_aligned(fc1_bias))
    out = torch.empty((b, s, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10100', f1, fc2_weight, out, rows, d, inter, bias=_aligned(fc2_bias), residual=h2)
    return out
