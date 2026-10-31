# problem: 036
def run(grad_output, residual, x_dwconv, x_nhwc, mean, var, x_normalized, x_ln, x_expanded, x_gelu,
        global_features, gf_mean, norm_features, x_grn_scaled, x_grn, dwconv_weight, layernorm_weight,
        pwconv1_weight, grn_weight, pwconv2_weight, drop_mask, drop_path_prob, eps):
    b, c, height, width = grad_output.shape
    g = grad_output
    if drop_path_prob > 0:
        g = _affine(_mul(g, drop_mask), 1.0 / (1 - drop_path_prob))
    gp = g.permute(0, 2, 3, 1)
    gg, gw2, gb2 = _linear_backward(gp, x_grn, pwconv2_weight)
    gscaled = _mul(gg, grn_weight)
    ggrnw = _sum(_mul(gg, x_grn_scaled), (0, 1, 2), True)
    ggrnb = _sum(gg, (0, 1, 2), True)
    ggx = _add(gg, _mul(gscaled, norm_features))
    gn = _sum(_mul(gscaled, x_gelu), (1, 2), True)
    denominator = _affine(gf_mean, bias=eps)
    gf = _div(gn, denominator)
    gm = _div(_affine(_mul(gn, global_features), -1), _mul(denominator, denominator))
    gf = _add(gf, _affine(gm, 1.0 / global_features.shape[-1]))
    ggx = _add(ggx, _div(_mul(x_gelu, gf), _affine(global_features, bias=eps)))
    ge = _mul(ggx, _unary('un_ff_14', x_expanded))
    gln, gw1, gb1 = _linear_backward(ge, x_ln, pwconv1_weight)
    glnw, glnb = _sum(_mul(gln, x_normalized), (0, 1, 2)), _sum(gln, (0, 1, 2))
    gn = _mul(gln, layernorm_weight)
    variance = _affine(var, bias=eps)
    std = _unary('un_ff_10', variance)
    centered = _sub(x_nhwc, mean)
    gx = _div(gn, std)
    gv = _div(_affine(_sum(_mul(gn, centered), -1, True), -1), _mul(_affine(variance, 2), std))
    gm = _affine(_sum(_div(gn, std), -1, True), -1)
    gm = _add(gm, _mul(gv, _affine(_sum(centered, -1, True), -2.0 / c)))
    gx = _add(_add(gx, _mul(gv, _affine(centered, 2.0 / c))), _affine(gm, 1.0 / c))
    gd = _aligned(gx.permute(0, 3, 1, 2))
    ginput, gdw = torch.empty_like(residual), torch.empty_like(dwconv_weight)
    _launch('dwconv2d_input_grad', [gd, _aligned(dwconv_weight), ginput], _blocks(ginput.numel(), 256), {'I': [c, height, width]})
    _launch('dwconv2d_weight_grad', [_aligned(residual), gd, gdw], (c * 49, 1, 1), {'I': [c, height, width]})
    return _add(ginput, grad_output), gdw, _sum(gd, (0, 2, 3)), glnw, glnb, gw1, gb1, ggrnw, ggrnb, gw2, gb2

# problem: 040
def run(grad_corrected, hidden_states, activated, prediction_coef_weight, correction_coef_weight,
        router_weight, norm_weight, altup_active_idx, rms_norm_eps):
    h, act, g = _f32(hidden_states), _f32(activated), _f32(grad_corrected)
    batch, seq, hidden = act.shape
    rw, nw, pw, cw = [_f32(t) for t in (router_weight, norm_weight, prediction_coef_weight, correction_coef_weight)]
    def route(x):
        variance = _sum(_mul(x, x), -1, True, mean=True)
        r = _unary('un_ff_11', _affine(variance, bias=rms_norm_eps))
        n = _mul(x, r)
        scaled = _affine(_mul(n, nw), 1.0 / hidden)
        return _unary('un_ff_4', _linear_f32(scaled, rw)), n, scaled, r
    mp, np, sp, rp = route(h[altup_active_idx])
    mc, nc, sc, rc = route(act)
    coef = _linear_f32(mp, pw).view(batch, seq, 3, 3).transpose(-2, -1)
    hp = h.permute(1, 2, 3, 0)
    pred = _add(_matmul(hp, coef).permute(3, 0, 1, 2), h)
    innovation = _sub(act, pred[altup_active_idx])
    cc = _affine(_linear_f32(mc, cw), bias=1)
    grad_cc = _sum(_mul(g, innovation.unsqueeze(0)), -1).permute(1, 2, 0)
    gcw = _matmul(grad_cc.reshape(-1, 3), mc.reshape(-1, 3), transpose_a=True)
    gmc = _matmul(grad_cc, cw)
    gi = _sum(_mul(g, cc.permute(2, 0, 1).unsqueeze(-1)), 0)
    gp = g.clone()
    gp[altup_active_idx].copy_(_sub(gp[altup_active_idx], gi))
    def route_backward(gm, modalities, x, normalized, scaled, r):
        gr = _mul(gm, _unary('un_ff_17', modalities))
        grw = _matmul(gr.reshape(-1, 3), scaled.reshape(-1, hidden), transpose_a=True)
        gn = _affine(_matmul(gr, rw), 1.0 / hidden)
        gnw = _sum(_mul(gn, normalized), (0, 1))
        gn = _mul(gn, nw)
        average = _sum(_mul(gn, x), -1, True, mean=True)
        gx = _sub(_mul(gn, r), _mul(_mul(x, _mul(_mul(r, r), r)), average))
        return gx, grw, gnw
    ga, grwc, gnwc = route_backward(gmc, mc, act, nc, sc, rc)
    ga = _add(gi, ga)
    gpp = gp.permute(1, 2, 3, 0)
    gh = _add(gp, _matmul(gpp, coef, transpose_b=True).permute(3, 0, 1, 2))
    gcoef = _matmul(hp, gpp, transpose_a=True).transpose(-2, -1).reshape(batch, seq, 9)
    gpw = _matmul(gcoef.reshape(-1, 9), mp.reshape(-1, 3), transpose_a=True)
    gmp = _matmul(gcoef, pw)
    gap, grwp, gnwp = route_backward(gmp, mp, h[altup_active_idx], np, sp, rp)
    gh[altup_active_idx].copy_(_add(gh[altup_active_idx], gap))
    return _affine(gh, dtype=torch.bfloat16), _affine(ga, dtype=torch.bfloat16), gpw, gcw, _add(grwp, grwc), _add(gnwp, gnwc)

# problem: 065
def run(hidden_states, router_indices, routing_weights, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
    experts = gate_up_proj.shape[0]
    groups, tokens, inverse = _moe_route(router_indices, experts + 1)
    x = _moe_gather('moe_gather_f', hidden_states, tokens)
    out = torch.zeros((tokens.numel(), hidden_states.shape[1]), dtype=torch.float32, device=hidden_states.device)
    weights = torch.empty_like(router_indices, dtype=torch.float32)
    _launch('expert_pair_weights', [_aligned(routing_weights), _aligned(router_indices), weights],
            _blocks(weights.numel(), 256), {'I': [experts, router_indices.shape[1]]})
    offsets = groups[:experts + 2].cpu().tolist()
    for expert, (start, end) in enumerate(zip(offsets[:experts], offsets[1:experts + 1])):
        if start != end:
            gu = _add(_matmul(x[start:end], gate_up_proj[expert]), gate_up_proj_bias[expert])
            gate, up = _clamp(gu[..., ::2], upper=7), _clamp(gu[..., 1::2], -7, 7)
            glu = _mul(gate, _unary('un_ff_3', _affine(gate, 1.702)))
            mid = _mul(_affine(up, bias=1), glu)
            out[start:end].copy_(_add(_matmul(mid, down_proj[expert]), down_proj_bias[expert]))
    return _ordered_combine(out, inverse, router_indices, weights)

# problem: 071
def run(predicted, target, source, edit_mask, extractor0_conv1_weight, extractor0_conv1_bias,
        extractor0_conv2_weight, extractor0_conv2_bias, extractor1_conv1_weight, extractor1_conv1_bias,
        extractor1_conv2_weight, extractor1_conv2_bias, extractor2_conv1_weight, extractor2_conv1_bias,
        extractor2_conv2_weight, extractor2_conv2_bias, scale_weights, pixel_loss_weight,
        perceptual_loss_weight, edit_region_weight_multiplier):
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=edit_mask.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=edit_mask.device).view(1, 1, 3, 3)
    ex, ey = _conv2d_f32(edit_mask, sx, pad=1), _conv2d_f32(edit_mask, sy, pad=1)
    magnitude = _unary('un_ff_10', _affine(_add(_mul(ex, ex), _mul(ey, ey)), bias=1e-8))
    spatial = _add(_affine(edit_mask, edit_region_weight_multiplier - 1, 1), _affine(magnitude, 0.5))
    pixel = _sum(_mul(_unary('un_ff_12', _sub(predicted, target)), spatial), mean=True)
    blocks = [(extractor0_conv1_weight, extractor0_conv1_bias, extractor0_conv2_weight, extractor0_conv2_bias),
              (extractor1_conv1_weight, extractor1_conv1_bias, extractor1_conv2_weight, extractor1_conv2_bias),
              (extractor2_conv1_weight, extractor2_conv1_bias, extractor2_conv2_weight, extractor2_conv2_bias)]
    p, t = predicted, target
    perceptual = torch.zeros((), dtype=torch.float32, device=predicted.device)
    for index, (w1, b1, w2, b2) in enumerate(blocks):
        def extract(x):
            return _maxpool2(_unary('un_ff_5', _conv2d_f32(_unary('un_ff_5', _conv2d_f32(x, w1, b1, pad=1)), w2, b2, pad=1)))
        p, t = extract(p), extract(t)
        sw = _bilinear2(spatial, p.shape[2], p.shape[3])
        loss = _sum(_mul(_unary('un_ff_12', _sub(p, t)), sw), mean=True)
        perceptual = _add(perceptual, _mul(loss, _unary('un_ff_12', scale_weights[index])))
    preservation = _sum(_mul(_unary('un_ff_12', _sub(predicted, source)), _affine(edit_mask, -1, 1)), mean=True)
    return _add(_add(_affine(pixel, pixel_loss_weight), _affine(perceptual, perceptual_loss_weight)), _affine(preservation, 0.3))
