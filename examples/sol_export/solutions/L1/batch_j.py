# L1 batch J: 079-087.

# problem: 079
def run(grad_output, ssm_output, gate, weight, bias, gate_activated, gated_output, use_silu_gate):
    b, s, hd = grad_output.shape
    ed = ssm_output.shape[2]
    dev = grad_output.device
    g2 = _aligned(grad_output).view(b * s, hd)
    grad_weight = torch.empty((hd, ed), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_11_00000', _op(g2), _op(_aligned(gated_output).view(b * s, ed)), grad_weight, hd, ed, b * s)
    grad_bias = torch.empty(hd, dtype=torch.bfloat16, device=dev)
    _colsum('colpart_b', 'colfin_b', g2, b * s, hd, grad_bias)
    gg = torch.empty((b * s, ed), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_01_00001', _op(g2), _op(_aligned(weight)), gg, b * s, ed, hd)
    grad_ssm = torch.empty((b, s, ed), dtype=torch.bfloat16, device=dev)
    grad_gate = torch.empty_like(grad_ssm)
    _launch('l1_079_ew', [gg, _aligned(gate_activated), _aligned(ssm_output), _aligned(gate), grad_ssm, grad_gate],
            _blocks(gg.numel(), 256), {'I': [gg.numel(), int(bool(use_silu_gate))]})
    return grad_ssm, grad_gate, grad_weight, grad_bias

# problem: 080
def run(hidden_states, temb, linear_weight, linear_bias, eps):
    b, s, d = hidden_states.shape
    dev = hidden_states.device
    m = linear_weight.shape[0]
    modulation = torch.empty((b, m), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10000', temb, linear_weight, modulation, b, m, d, bias=_aligned(linear_bias))
    out = torch.empty((b, s, d), dtype=torch.float32, device=dev)
    _launch('adaln_fff', [_aligned(hidden_states), modulation, out], (b * s, 1, 1), {'I': [d, s, m, d, 0], 'f': [float(eps)]})
    return out

# problem: 081
def run(image_attention_output, context_attention_output, to_out_weight, to_out_bias):
    d = to_out_weight.shape[0]
    dev = to_out_weight.device
    w = _aligned(to_out_weight)
    wsum = torch.empty((d, d), dtype=torch.float32, device=dev)
    n = d * d
    _launch('bin_fff_0', [w, w[:, d:], wsum], _blocks(n, 256), {'I': [1, d, d, n, 0, 0, 2 * d, 1, 0, 0, 2 * d, 1], 'f': [1.0]})
    wb = _tf32x3(wsum, False)
    outs = []
    for x in (image_attention_output, context_attention_output):
        bsz, s, _ = x.shape
        o = torch.empty((bsz, s, d), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_10000', _tf32x3(x.reshape(bsz * s, d), True), wb, o, bsz * s, d, d, k_groups=3, bias=_aligned(to_out_bias))
        outs.append(o)
    return outs[0], outs[1]

# problem: 082
def run(hidden_states, qkv_weight, qkv_bias, q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias,
        out_proj_weight, out_proj_bias, eps):
    b, s, dim = hidden_states.shape
    dev = hidden_states.device
    xa = _tf32x3(hidden_states.reshape(b * s, dim), True)
    w, bias = _aligned(qkv_weight), _aligned(qkv_bias)
    proj = []
    for j in range(3):
        o = torch.empty((b * s, dim), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_10000', xa, _tf32x3(w[j * dim:(j + 1) * dim], False), o, b * s, dim, dim, k_groups=3,
              bias=bias[j * dim:(j + 1) * dim])
        proj.append(o)
    q, k, v = proj
    qn, kn = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_fff_64_4000', q, qn, b * s * 24, 1, 1, eps, weight=_aligned(q_norm_weight), bias=_aligned(q_norm_bias))
    _hnr('hnr_fff_64_4000', k, kn, b * s * 24, 1, 1, eps, weight=_aligned(k_norm_weight), bias=_aligned(k_norm_bias))
    attn = _attn_f32(qn, kn, v, b, s, s, 24, 24, 64, scale=64 ** -0.5)
    out = torch.empty((b * s, dim), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10000', attn.view(b * s, dim), out_proj_weight, out, b * s, dim, dim, bias=_aligned(out_proj_bias))
    return out.view(b, s, dim)

# problem: 083
def run(attention_weights, value):
    b, h, sq, skv = attention_weights.shape
    d = value.shape[3]
    p = _aligned(attention_weights).view(b * h, sq, skv)
    if skv % 8:
        ldp = (skv + 7) // 8 * 8
        padded = torch.empty((b * h, sq, ldp), dtype=p.dtype, device=p.device)
        _launch('pad_rows_bf16', [p, padded], _blocks(padded.numel(), 256), {'I': [padded.numel(), skv, ldp]})
        p = padded
    out = torch.empty((b, sq, h * d), dtype=attention_weights.dtype, device=p.device)
    _gemm('gemm_bb_64_01_00000', _opb(p), _opb(_aligned(value).view(b * h, skv, d)), out, sq, d, skv,
          batches=b * h, ld=h * d, o_div=h, o_s1=sq * h * d, o_s2=d)
    return out

# problem: 086
def run(iou_token_out, mask_tokens_out, iou_proj_in_weight, iou_proj_in_bias, iou_hidden_weight, iou_hidden_bias,
        iou_proj_out_weight, iou_proj_out_bias,
        hyper0_proj_in_weight, hyper0_proj_in_bias, hyper0_hidden_weight, hyper0_hidden_bias, hyper0_proj_out_weight, hyper0_proj_out_bias,
        hyper1_proj_in_weight, hyper1_proj_in_bias, hyper1_hidden_weight, hyper1_hidden_bias, hyper1_proj_out_weight, hyper1_proj_out_bias,
        hyper2_proj_in_weight, hyper2_proj_in_bias, hyper2_hidden_weight, hyper2_hidden_bias, hyper2_proj_out_weight, hyper2_proj_out_bias,
        hyper3_proj_in_weight, hyper3_proj_in_bias, hyper3_hidden_weight, hyper3_hidden_bias, hyper3_proj_out_weight, hyper3_proj_out_bias):
    b, p, h = iou_token_out.shape
    rows = b * p
    dev = iou_token_out.device
    def mlp(x, w1, b1, w2, b2, w3, b3, out):
        h1 = torch.empty((rows, w1.shape[0]), dtype=torch.float32, device=dev)
        _gemm3('gemm_tf_256_00_14000', x, w1, h1, rows, w1.shape[0], x.shape[1], bias=_aligned(b1))
        h2 = torch.empty((rows, w2.shape[0]), dtype=torch.float32, device=dev)
        _gemm3('gemm_tf_256_00_14000', h1, w2, h2, rows, w2.shape[0], h1.shape[1], bias=_aligned(b2))
        n = w3.shape[0]
        n8 = (n + 7) // 8 * 8
        ld = out.stride(0)
        _gemm3('gemm_tf_64_00_10000', h2, w3, out, rows, n8, h2.shape[1], bias=_aligned(b3), ld=ld)
    iou = torch.empty((rows, 8), dtype=torch.float32, device=dev)
    mlp(iou_token_out.reshape(rows, h), iou_proj_in_weight, iou_proj_in_bias, iou_hidden_weight, iou_hidden_bias,
        iou_proj_out_weight, iou_proj_out_bias, iou)
    hyper = torch.empty((b, p, 4, 32), dtype=torch.float32, device=dev)
    hv = hyper.view(rows, 128)
    tokens = mask_tokens_out.reshape(rows, 4, h)
    params = [(hyper0_proj_in_weight, hyper0_proj_in_bias, hyper0_hidden_weight, hyper0_hidden_bias, hyper0_proj_out_weight, hyper0_proj_out_bias),
              (hyper1_proj_in_weight, hyper1_proj_in_bias, hyper1_hidden_weight, hyper1_hidden_bias, hyper1_proj_out_weight, hyper1_proj_out_bias),
              (hyper2_proj_in_weight, hyper2_proj_in_bias, hyper2_hidden_weight, hyper2_hidden_bias, hyper2_proj_out_weight, hyper2_proj_out_bias),
              (hyper3_proj_in_weight, hyper3_proj_in_bias, hyper3_hidden_weight, hyper3_hidden_bias, hyper3_proj_out_weight, hyper3_proj_out_bias)]
    for j, prm in enumerate(params):
        mlp(tokens[:, j, :], *prm, hv[:, j * 32:(j + 1) * 32])
    return iou[:, :4].reshape(b, p, 4), hyper

# problem: 087
def run(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight):
    b, s, h = grad_output.shape
    rows = b * s
    dev = grad_output.device
    table = torch.zeros((65536, h), dtype=torch.float32, device=dev)
    blocks = max(1, min(rows, _sm_count(dev) * 4))
    partial = torch.empty((blocks, h), dtype=torch.float32, device=dev)
    _launch('l1_087_rms_bwd', [_aligned(grad_output), _aligned(hidden_states_fp32), _aligned(rstd), _aligned(norm_weight),
                               _aligned(input_ids), table, partial], (blocks, 1, 1), {'I': [rows]})
    grad_w = torch.empty(h, dtype=torch.bfloat16, device=dev)
    _launch('colfin_b', [partial, grad_w], _blocks(h, 256), {'I': [h, blocks]})
    out = torch.empty((65536, h), dtype=torch.bfloat16, device=dev)
    _launch('cast_fb', [table, out], _blocks(table.numel() // 8, 256), {'I': [table.numel() // 8]})
    return out, grad_w
