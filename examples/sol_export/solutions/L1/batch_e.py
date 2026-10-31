# L1 batch E: 029-037.

# problem: 029
def run(hidden_states, attention_mask, in_proj_weight, in_proj_bias, conv1d_weight, conv1d_bias):
    b, s, _ = hidden_states.shape
    c = conv1d_weight.shape[0]
    proj = _linear('gemm_bb_256_00_10000', hidden_states, in_proj_weight, bias=_aligned(in_proj_bias))
    out = torch.empty((b, s, c), dtype=hidden_states.dtype, device=hidden_states.device)
    _launch('l1_029_conv_gate', [proj, _aligned(attention_mask), _aligned(conv1d_weight), _aligned(conv1d_bias), out],
            _blocks(out.numel(), 256), {'I': [out.numel(), s, c]})
    return out.transpose(1, 2), proj[:, :, c:].transpose(1, 2)

# problem: 030
def run(attn_output, residual, o_proj_weight):
    return _linear('gemm_bb_256_00_00101', attn_output, o_proj_weight, residual=_aligned(residual))

# problem: 031
def run(query, key):
    b, h, s, d = query.shape
    ldp = (s + 7) // 8 * 8
    out = torch.empty((b, h, s, ldp), dtype=torch.bfloat16, device=query.device)
    _gemm('gemm_bb_256_00_00000', _opb(_aligned(query).view(b * h, s, d)), _opb(_aligned(key).view(b * h, s, d)),
          out, s, ldp, d, batches=b * h, ld=ldp, o_s1=s * ldp, alpha=d ** -0.5)
    return out[..., :s]

# problem: 032
def run(attn_weights, value_states):
    b, h, s, _ = attn_weights.shape
    d = value_states.shape[3]
    p = _aligned(attn_weights).view(b * h, s, s)
    if s % 8:
        ldp = (s + 7) // 8 * 8
        padded = torch.empty((b * h, s, ldp), dtype=p.dtype, device=p.device)
        _launch('pad_rows_bf16', [p, padded], _blocks(padded.numel(), 256), {'I': [padded.numel(), s, ldp]})
        p = padded
    out = torch.empty((b, s, h * d), dtype=attn_weights.dtype, device=attn_weights.device)
    _gemm('gemm_bb_128_01_00000', _opb(p), _opb(_aligned(value_states).view(b * h, s, d)), out, s, d, s,
          batches=b * h, ld=h * d, o_div=h, o_s1=s * h * d, o_s2=d)
    return out

# problem: 034
def run(ids, theta):
    s = ids.shape[0]
    cos = torch.empty((s, 128), dtype=torch.float32, device=ids.device)
    sin = torch.empty_like(cos)
    _launch('l1_034_rope_freqs', [_aligned(ids), cos, sin], _blocks(s * 128, 256), {'I': [s * 128], 'f': [float(theta)]})
    return cos, sin

# problem: 035
def run(emb, weight, bias):
    b, k = emb.shape
    n = weight.shape[0]
    out = torch.empty((b, n), dtype=torch.float32, device=emb.device)
    _gemm3('gemm_tf_256_00_10000', emb, weight, out, b, n, k, bias=_aligned(bias))
    return out.chunk(6, dim=1)

# problem: 036
def run(hidden_states, temb, linear_weight, linear_bias, proj_out_weight, proj_out_bias, eps):
    b, s, d = hidden_states.shape
    dev = hidden_states.device
    ts = torch.empty_like(temb)
    _launch('un_ff_0', [_aligned(temb), ts], _blocks(temb.numel(), 256), {'I': [temb.numel()]})
    m = linear_weight.shape[0]
    modulation = torch.empty((b, m), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10000', ts, linear_weight, modulation, b, m, d, bias=_aligned(linear_bias))
    mod = torch.empty((b, s, d), dtype=torch.float32, device=dev)
    _launch('adaln_fff', [_aligned(hidden_states), modulation, mod], (b * s, 1, 1),
            {'I': [d, s, m, 0, d], 'f': [float(eps)]})
    n = proj_out_weight.shape[0]
    out = torch.empty((b, s, n), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_64_00_10000', mod.view(b * s, d), proj_out_weight, out, b * s, n, d, bias=_aligned(proj_out_bias))
    return out

# problem: 037
def run(hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    rows, k = x.shape
    n1 = fc1_weight.shape[0]
    h1 = torch.empty((rows, n1), dtype=torch.float32, device=x.device)
    _gemm3('gemm_tf_256_00_11000', x, fc1_weight, h1, rows, n1, k, bias=_aligned(fc1_bias))
    n2 = fc2_weight.shape[0]
    out = torch.empty((rows, n2), dtype=torch.float32, device=x.device)
    _gemm3('gemm_tf_256_00_10000', h1, fc2_weight, out, rows, n2, n1, bias=_aligned(fc2_bias))
    return out.view(*hidden_states.shape[:-1], n2)
