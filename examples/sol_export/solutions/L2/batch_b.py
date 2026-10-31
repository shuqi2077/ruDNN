# L2 batch B: 066 067 068.

# problem: 066
def run(x, time_emb, norm1_weight, norm1_bias, conv1_weight, conv1_bias, time_emb_proj_weight, time_emb_proj_bias,
        norm2_weight, norm2_bias, conv2_weight, conv2_bias, norm_eps):
    b, c, h, w = x.shape
    dev = x.device
    hw = h * w
    xc = _aligned(x)
    def gn_silu(t, weight, bias):
        chunk = 16384
        chunks = (hw + chunk - 1) // chunk
        partial = torch.empty(b * t.shape[1] * chunks * 2, dtype=torch.float64, device=dev)
        g = torch.empty_like(t)
        _launch('l1_078_gn_partial', [t, partial], (chunks, t.shape[1], b), {'I': [hw, chunk]})
        _launch('l1_078_gn_apply', [t, partial, _aligned(weight), _aligned(bias), g], (chunks, t.shape[1], b),
                {'I': [hw, chunk, 32], 'f': [float(norm_eps)]})
        _launch('un_ff_0', [g, g], _blocks(g.numel(), 256), {'I': [g.numel()]})
        return g
    def conv(t, weight, bias, bias_stride, residual, name):
        oc = weight.shape[0]
        cols, oh, ow = _im2col('im2col_f', t, 3, 3, stride=1, pad=1)
        out = torch.empty((b, oc, oh, ow), dtype=torch.float32, device=dev)
        _gemm(name, _op(_aligned(weight).view(oc, -1)), _opb(cols), out, oc, oh * ow, cols.shape[2], batches=b, ld=oh * ow,
              o_s1=oc * oh * ow, bias=bias, bias_stride=bias_stride, residual=residual)
        return out
    h1 = gn_silu(xc, norm1_weight, norm1_bias)
    ts = torch.empty_like(time_emb)
    _launch('un_ff_0', [_aligned(time_emb), ts], _blocks(ts.numel(), 256), {'I': [ts.numel()]})
    oc = conv1_weight.shape[0]
    tproj = torch.empty((b, oc), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10000', ts, time_emb_proj_weight, tproj, b, oc, ts.shape[1], bias=_aligned(time_emb_proj_bias))
    combined = torch.empty_like(tproj)
    n = combined.numel()
    _launch('bin_fff_0', [tproj, _aligned(conv1_bias), combined], _blocks(n, 256),
            {'I': [1, b, oc, n, 0, 0, oc, 1, 0, 0, 0, 1], 'f': [1.0]})
    h2 = conv(h1, conv1_weight, combined, oc, None, 'gemm_tf_256_00_20000')
    h3 = gn_silu(h2, norm2_weight, norm2_bias)
    return conv(h3, conv2_weight, _aligned(conv2_bias), 0, xc, 'gemm_tf_256_00_20100')

# problem: 067
def run(hidden_states, encoder_hidden_states, pooled_projections, timestep, proj_weight, proj_bias, pos_embed,
        timestep_linear1_weight, timestep_linear1_bias, timestep_linear2_weight, timestep_linear2_bias,
        pooled_linear1_weight, pooled_linear1_bias, pooled_linear2_weight, pooled_linear2_bias,
        context_embedder_weight, context_embedder_bias, freqs):
    b = hidden_states.shape[0]
    dev = hidden_states.device
    d = proj_weight.shape[0]
    cols, oh, ow = _im2col('im2col_f', hidden_states, 2, 2, stride=2)
    p = oh * ow
    out_h = torch.empty((b, p, d), dtype=torch.float32, device=dev)
    a = _tf32x3(cols.view(b * p, -1), True)
    a.coords[0], a.coords[1] = 1, p
    _gemm('gemm_tf_256_00_10100', a, _tf32x3(_aligned(proj_weight).view(d, -1), False), out_h, p, d, cols.shape[2],
          batches=b, k_groups=3, ld=d, o_s1=p * d, bias=_aligned(proj_bias), residual=_aligned(pos_embed), res_s1=0)
    half = freqs.shape[0]
    emb = torch.empty((b, 2 * half), dtype=torch.float32, device=dev)
    _launch('l2_031_temb', [_aligned(timestep), _aligned(freqs), emb], _blocks(emb.numel(), 256),
            {'I': [emb.numel(), half], 'f': [1.0]})
    t1 = torch.empty((b, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_13000', emb, timestep_linear1_weight, t1, b, d, 2 * half, bias=_aligned(timestep_linear1_bias))
    te = torch.empty((b, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10000', t1, timestep_linear2_weight, te, b, d, d, bias=_aligned(timestep_linear2_bias))
    p1 = torch.empty((b, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_13000', pooled_projections, pooled_linear1_weight, p1, b, d, pooled_projections.shape[1],
           bias=_aligned(pooled_linear1_bias))
    temb = torch.empty((b, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10100', p1, pooled_linear2_weight, temb, b, d, d, bias=_aligned(pooled_linear2_bias), residual=te)
    s = encoder_hidden_states.shape[1]
    ctx = torch.empty((b, s, context_embedder_weight.shape[0]), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10000', encoder_hidden_states.reshape(b * s, -1), context_embedder_weight, ctx, b * s,
           context_embedder_weight.shape[0], encoder_hidden_states.shape[2], bias=_aligned(context_embedder_bias))
    return out_h, temb, ctx

# problem: 068
def run(grad_output, hidden_states, weight1, weight2, linear1_out, tanh_inner, gelu_out):
    dim = grad_output.shape[-1]
    hid = weight1.shape[0]
    rows = grad_output.numel() // dim
    dev = grad_output.device
    go = _aligned(grad_output).view(rows, dim)
    gg = torch.empty((rows, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_01_00000', go, weight2, gg, rows, hid, dim)
    grad_w2 = torch.empty((dim, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_11_00000', go, _aligned(gelu_out).view(rows, hid), grad_w2, dim, hid, rows)
    g1 = torch.empty_like(gg)
    _launch('l2_068_gelu_bwd', [gg, _aligned(linear1_out), _aligned(tanh_inner), g1], _blocks(g1.numel(), 256), {'I': [g1.numel()]})
    gh = torch.empty((rows, dim), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_01_00000', g1, weight1, gh, rows, dim, hid)
    grad_w1 = torch.empty((hid, dim), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_11_00000', g1, _aligned(hidden_states).view(rows, dim), grad_w1, hid, dim, rows)
    return gh.view(*grad_output.shape[:-1], dim), grad_w1, grad_w2
