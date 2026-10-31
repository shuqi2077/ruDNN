# L2 batch A: 004 015 030 031 035.

# problem: 004
def run(hidden_states, residual, norm_weight, gate_proj_weight, up_proj_weight, down_proj_weight, eps):
    width = hidden_states.shape[-1]
    h = _norm('norm_bbb_011000', _aligned(hidden_states), width, eps, weight=_aligned(norm_weight),
              residual=_aligned(residual), residual_out=torch.empty_like(hidden_states))
    gate = _linear('gemm_bb_256_00_00001', h, gate_proj_weight)
    act = _linear('gemm_bb_256_00_00201', h, up_proj_weight, residual=gate)
    return _linear('gemm_bb_256_00_00000', act, down_proj_weight)

# problem: 015
def run(input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        conv_out_weight, positional_embedding, embed_scale):
    b = input_features.shape[0]
    dev = input_features.device
    def conv_nchw(x, w, bias):
        oc = w.shape[0]
        cols, oh, ow = _im2col('im2col_b', x, 3, 3, stride=2, pad=1)
        kp = cols.shape[2]
        w2 = _aligned(w).view(oc, -1)
        if w2.shape[1] != kp:
            wp = torch.empty((oc, kp), dtype=w.dtype, device=dev)
            _launch('pad_rows_bf16', [w2, wp], _blocks(wp.numel(), 256), {'I': [wp.numel(), w2.shape[1], kp]})
            w2 = wp
        out = torch.empty((b, oc, oh, ow), dtype=x.dtype, device=dev)
        _gemm('gemm_bb_256_00_22001', _op(w2), _opb(cols), out, oc, oh * ow, kp, batches=b, ld=oh * ow, o_s1=oc * oh * ow,
              bias=_aligned(bias))
        return out
    x1 = conv_nchw(input_features, conv2d1_weight, conv2d1_bias)
    x2 = conv_nchw(x1, conv2d2_weight, conv2d2_bias)
    cols, f, t = _im2col('im2col_b', x2, 3, 3, stride=2, pad=1)
    oc = conv2d3_weight.shape[0]
    x3 = torch.empty((b, f * t, oc), dtype=x2.dtype, device=dev)
    _gemm('gemm_bb_128_00_12001', _opb(cols), _op(_aligned(conv2d3_weight).view(oc, -1)), x3, f * t, oc, cols.shape[2],
          batches=b, ld=oc, o_s1=f * t * oc, bias=_aligned(conv2d3_bias))
    d = conv_out_weight.shape[0]
    wsrc = _aligned(conv_out_weight)
    wp = torch.empty((d, f, oc), dtype=wsrc.dtype, device=dev)
    n = wp.numel()
    _launch('permute_b', [wsrc, wp], _blocks(n, 256), {'I': [d, f, oc, n, 0, f * oc, 1, f]})
    y = torch.empty((b, t, d), dtype=x3.dtype, device=dev)
    _gemm('gemm_bb_256_00_00001', _opb(x3.view(b * f, t, oc), div=1, mul=f, kstep=1),
          _Op(wp, [oc, f, d], [oc * 2, f * oc * 2], y_mul=0, y_kstep=1), y, t, d, oc, batches=b, k_groups=f,
          ld=d, o_s1=t * d, alpha=float(embed_scale))
    out = torch.empty_like(y)
    cnt = out.numel()
    _launch('bin_bbb_0', [y, _aligned(positional_embedding), out], _blocks(cnt, 256),
            {'I': [b, t, d, cnt, 0, t * d, d, 1, 0, 0, d, 1], 'f': [1.0]})
    return out

# problem: 030
def run(hidden_states, encoder_hidden_states, process_weight):
    d = process_weight.shape[0]
    wb = _tf32x3(process_weight, False)
    outs = []
    for x in (encoder_hidden_states, hidden_states):
        bsz, s, _ = x.shape
        o = torch.empty((bsz, s, d), dtype=torch.float32, device=x.device)
        _gemm('gemm_tf_256_00_00000', _tf32x3(x.reshape(bsz * s, d), True), wb, o, bsz * s, d, d, k_groups=3)
        outs.append(o)
    return outs[0], outs[1]

# problem: 031
def run(timestep, pooled_projections, freqs, timestep_linear1_weight, timestep_linear1_bias, timestep_linear2_weight,
        timestep_linear2_bias, text_embedder_weight, text_embedder_bias):
    b = timestep.shape[0]
    half = freqs.shape[0]
    dev = timestep.device
    emb = torch.empty((b, 2 * half), dtype=torch.float32, device=dev)
    _launch('l2_031_temb', [_aligned(timestep), _aligned(freqs), emb], _blocks(emb.numel(), 256),
            {'I': [emb.numel(), half], 'f': [1000.0]})
    d = timestep_linear1_weight.shape[0]
    x = torch.empty((b, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_13000', emb, timestep_linear1_weight, x, b, d, 2 * half, bias=_aligned(timestep_linear1_bias))
    te = torch.empty((b, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10000', x, timestep_linear2_weight, te, b, d, d, bias=_aligned(timestep_linear2_bias))
    out = torch.empty((b, d), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_10100', pooled_projections, text_embedder_weight, out, b, d, pooled_projections.shape[1],
           bias=_aligned(text_embedder_bias), residual=te)
    return out

# problem: 035
def run(x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias, pwconv1_weight, pwconv1_bias, grn_weight, grn_bias,
        pwconv2_weight, pwconv2_bias, eps, layer_norm_eps):
    b, c, h, w = x.shape
    dev = x.device
    p = h * w
    xc = _aligned(x)
    dw = torch.empty((b, h, w, c), dtype=torch.float32, device=dev)
    _launch('l2_035_dwconv', [xc, _aligned(dwconv_weight), _aligned(dwconv_bias), dw], _blocks(dw.numel(), 256),
            {'I': [dw.numel(), c, h, w]})
    ln = _norm('norm_fff_101100', dw, c, layer_norm_eps, weight=_aligned(layernorm_weight), bias=_aligned(layernorm_bias))
    e4 = pwconv1_weight.shape[0]
    x1 = torch.empty((b * p, e4), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_12000', ln.view(b * p, c), pwconv1_weight, x1, b * p, e4, c, bias=_aligned(pwconv1_bias))
    sumsq = torch.empty((b, e4), dtype=torch.float32, device=dev)
    _launch('grn_sumsq', [x1, sumsq], ((e4 + 255) // 256, b, 1), {'I': [p, e4]})
    nx = torch.empty_like(sumsq)
    _launch('grn_scale_n', [sumsq, nx], (b, 1, 1), {'I': [e4], 'f': [float(eps)]})
    x2 = torch.empty_like(x1)
    _launch('grn_apply', [x1, nx, _aligned(grn_weight).view(-1), _aligned(grn_bias).view(-1), x2], _blocks(x1.numel(), 256),
            {'I': [x1.numel(), p, e4]})
    out = torch.empty((b, c, h, w), dtype=torch.float32, device=dev)
    planes = _split3(x2, b * p * e4, b * p * e4, False).view(3 * b, p, e4)
    _gemm('gemm_tf_256_00_20100', _tf32x3(pwconv2_weight, True), _opb(planes, div=1, mul=1, kstep=b), out, c, p, e4,
          batches=b, k_groups=3, ld=p, o_s1=c * p, bias=_aligned(pwconv2_bias), residual=xc)
    return out
