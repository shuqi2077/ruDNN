# L1 batch C: tcgen05 GEMM / fused-row solutions (sol_gemm.rs, sol/l1c.rs).

# problem: 001
def run(grad_attn_output, attn_weights, attn_weights_dropped, value_states, dropout_mask, attention_dropout):
    dev = grad_attn_output.device
    b, lq, heads, d = grad_attn_output.shape
    kv, lk = value_states.shape[1], value_states.shape[2]
    group = heads // kv
    dO = _aligned(grad_attn_output)
    ldp = (lk + 7) // 8 * 8
    dp = torch.empty((b * heads, lq, ldp), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_00_00000', _opl(dO, heads), _opb(_aligned(value_states).view(b * kv, lk, d), div=group),
          dp, lq, ldp, d, batches=b * heads, ld=ldp, o_s1=lq * ldp)
    grad_scores = torch.empty((b, heads, lq, lk), dtype=torch.bfloat16, device=dev)
    p = float(attention_dropout)
    keep = _aligned(dropout_mask).view(torch.uint8)
    _launch('softmax_bwd_bf16', [dp, _aligned(attn_weights), keep, grad_scores], (b * heads * lq, 1, 1),
            {'I': [lk, ldp, int(p > 0.0)], 'f': [1.0 / (1.0 - p) if p > 0.0 else 1.0]})
    pd = _aligned(attn_weights_dropped)
    if lk % 8:
        padded = torch.empty((b * heads, lq, ldp), dtype=torch.bfloat16, device=dev)
        _launch('pad_rows_bf16', [pd, padded], _blocks(padded.numel(), 256), {'I': [padded.numel(), lk, ldp]})
        pd = padded
    pd = pd.view(b * heads, lq, ldp)
    grad_v = torch.empty((b, kv, lk, d), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_11_00000', _opb(pd, div=1, mul=group, kstep=1), _opl(dO, kv, div=1, mul=group, kstep=1),
          grad_v, lk, d, lq, batches=b * kv, k_groups=group, ld=d, o_s1=lk * d)
    return grad_scores, grad_v

# problem: 003
def run(hidden_states, weight):
    return _linear('gemm_bb_256_00_00000', hidden_states, weight)

# problem: 004
def run(grad_output, reshaped, weight):
    dev = grad_output.device
    b, s, h = grad_output.shape
    g2 = _aligned(grad_output).view(b * s, h)
    x2 = _aligned(reshaped).view(b * s, h)
    grad_weight = torch.empty((h, h), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(g2), _op(x2), grad_weight, h, h, b * s)
    out = torch.empty((b, 32, s, 64), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_01_00000', _opb(g2.view(b, s, h)), _op(_aligned(weight)), out, s, h, h, batches=b,
          ld=64, o_s1=32 * s * 64, col_split=64, col_stride=s * 64)
    return out, grad_weight

# problem: 005
def run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
    b, s, h = x.shape
    bcx = _linear('gemm_bb_256_00_10000', x, in_proj_weight, bias=in_proj_bias)
    y = torch.empty((b, s, h), dtype=x.dtype, device=x.device)
    _launch('l1_005_conv_gate', [bcx, _aligned(conv_weight), _aligned(conv_bias), y], _blocks(y.numel(), 256),
            {'I': [y.numel(), s, h]})
    return _linear('gemm_bb_256_00_10000', y, out_proj_weight, bias=out_proj_bias)

# problem: 006
def run(u, short_filter_weight, short_filter_bias):
    b, c, l = u.shape
    outs = [torch.empty((b, 256, l), dtype=torch.float32, device=u.device) for _ in range(3)]
    _launch('l1_006_hyena', [_aligned(u), _aligned(short_filter_weight), _aligned(short_filter_bias)] + outs,
            _blocks(outs[0].numel(), 256), {'I': [outs[0].numel(), l]})
    return outs[0], outs[1], outs[2]

# problem: 008
def run(final_hidden_states, expert_outputs, token_indices):
    rows, h = final_hidden_states.shape
    acc = torch.empty((rows, h), dtype=torch.float32, device=final_hidden_states.device)
    _launch('cast_bf', [_aligned(final_hidden_states), acc], _blocks(acc.numel() // 8, 256), {'I': [acc.numel() // 8]})
    n = expert_outputs.shape[0]
    _launch('index_add_bf16', [_aligned(expert_outputs), _aligned(token_indices), acc], _blocks(n * h, 256), {'I': [n * h, h]})
    out = torch.empty((rows, h), dtype=torch.bfloat16, device=acc.device)
    _launch('cast_fb', [acc, out], _blocks(acc.numel() // 8, 256), {'I': [acc.numel() // 8]})
    return out

# problem: 009
def run(grad_output, token_indices, selected_tokens, w1_output, gate_output, up_output, gated_output,
        expert_output, selected_weights, w1_weight, w2_weight, w3_weight):
    dev = grad_output.device
    rows, hidden = grad_output.shape
    n = token_indices.shape[0]
    ffn = w1_weight.shape[0]
    idx = _aligned(token_indices)
    routing = torch.zeros(rows, dtype=torch.bfloat16, device=dev)
    grad_expert = torch.empty((n, hidden), dtype=torch.bfloat16, device=dev)
    _launch('l1_009_prep', [_aligned(grad_output), idx, _aligned(expert_output), _aligned(selected_weights), routing, grad_expert],
            (n, 1, 1), {'I': [hidden // 8]})
    g = torch.empty((n, ffn), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_01_00000', _op(grad_expert), _op(_aligned(w2_weight)), g, n, ffn, hidden)
    grad_up = torch.empty((n, ffn), dtype=torch.bfloat16, device=dev)
    grad_w1o = torch.empty((n, ffn), dtype=torch.bfloat16, device=dev)
    _launch('l1_009_mid', [g, _aligned(gate_output), _aligned(up_output), _aligned(w1_output), grad_up, grad_w1o],
            _blocks(n * ffn, 256), {'I': [n * ffn]})
    sel = _aligned(selected_tokens)
    grad_w2 = torch.empty((hidden, ffn), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(grad_expert), _op(_aligned(gated_output)), grad_w2, hidden, ffn, n)
    grad_w3 = torch.empty((ffn, hidden), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(grad_up), _op(sel), grad_w3, ffn, hidden, n)
    grad_w1 = torch.empty((ffn, hidden), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(grad_w1o), _op(sel), grad_w1, ffn, hidden, n)
    gs = torch.empty((n, hidden), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_01_00000', _op(grad_w1o), _op(_aligned(w1_weight)), gs, n, hidden, ffn)
    _gemm('gemm_bf_256_01_00100', _op(grad_up), _op(_aligned(w3_weight)), gs, n, hidden, ffn, residual=gs)
    grad_hidden = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=dev)
    _launch('scatter_fb', [gs, idx, grad_hidden], _blocks(n * hidden // 8, 256), {'I': [n * hidden // 8, hidden // 8]})
    return grad_hidden, routing, grad_w1, grad_w2, grad_w3

# problem: 010
def run(hidden_states, v_proj_weight):
    b, s, h = hidden_states.shape
    n = v_proj_weight.shape[0]
    out = torch.empty((b, n // 128, s, 128), dtype=hidden_states.dtype, device=hidden_states.device)
    _gemm('gemm_bb_256_00_00000', _opb(_aligned(hidden_states)), _op(_aligned(v_proj_weight)), out, s, n, h, batches=b,
          ld=128, o_s1=n * s, col_split=128, col_stride=s * 128)
    return out
