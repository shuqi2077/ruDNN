# L1 batch I: 070-077.

# problem: 070
def run(hidden_states, A_cumsum, B, C):
    b, nc, cs, h, p = hidden_states.shape
    g = B.shape[3]
    dev = hidden_states.device
    bc = b * nc
    G = torch.empty((bc * g, cs, cs), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_128_00_00000', _opl(_aligned(C).view(bc, cs, g, C.shape[4]), g), _opl(_aligned(B).view(bc, cs, g, B.shape[4]), g),
          G, cs, cs, C.shape[4], batches=bc * g, ld=cs, o_s1=cs * cs)
    prefix = torch.empty((b * h * nc, cs), dtype=torch.float32, device=dev)
    _launch('l1_070_prefix', [_aligned(A_cumsum), prefix], _blocks(b * h * nc, 256), {'I': [b * h * nc]})
    M = torch.empty((bc * h, cs, cs), dtype=torch.bfloat16, device=dev)
    _launch('l1_070_mask', [G, prefix, M], _blocks(M.numel(), 256), {'I': [M.numel(), nc]})
    Y = torch.empty((b, nc, cs, h, p), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_01_00000', _opb(M), _opl(_aligned(hidden_states).view(bc, cs, h, p), h), Y, cs, p, cs,
          batches=bc * h, ld=h * p, o_div=h, o_s1=cs * h * p, o_s2=p)
    return Y

# problem: 072
def run(grad_query_states, grad_key_states, grad_value_states, decoder_hidden_states, encoder_hidden_states,
        q_weight, k_weight, v_weight):
    b, h, sd, d = grad_query_states.shape
    se = grad_key_states.shape[2]
    dev = grad_query_states.device
    gq = torch.empty((b, sd, h, d), dtype=torch.float32, device=dev)
    n = gq.numel()
    _launch('permute_f', [_aligned(grad_query_states), gq], _blocks(n, 256), {'I': [sd, h, d, n, h * sd * d, d, sd * d, 1]})
    gq2 = gq.view(b * sd, h * d)
    dec2 = decoder_hidden_states.reshape(b * sd, -1)
    hid = dec2.shape[1]
    gdec = torch.empty((b * sd, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_01_00000', gq2, q_weight, gdec, b * sd, hid, h * d)
    gwq = torch.empty((h * d, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_11_00000', gq2, dec2, gwq, h * d, hid, b * sd)
    kvh = k_weight.shape[0] // d
    proj = []
    for gsrc in (grad_key_states, grad_value_states):
        o = torch.empty((b * se, kvh * d), dtype=torch.float32, device=dev)
        _launch('l1_072_group_sum', [_aligned(gsrc), o], _blocks(o.numel(), 256), {'I': [o.numel(), kvh, h // kvh, se, d]})
        proj.append(o)
    gk2, gv2 = proj
    enc2 = encoder_hidden_states.reshape(b * se, -1)
    ch = enc2.shape[1]
    genc = torch.empty((b * se, ch), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_01_00000', gk2, k_weight, genc, b * se, ch, kvh * d)
    _gemm3('gemm_tf_256_01_00100', gv2, v_weight, genc, b * se, ch, kvh * d, residual=genc)
    gwk = torch.empty((kvh * d, ch), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_11_00000', gk2, enc2, gwk, kvh * d, ch, b * se)
    gwv = torch.empty((kvh * d, ch), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_11_00000', gv2, enc2, gwv, kvh * d, ch, b * se)
    return gdec.view(b, sd, hid), genc.view(b, se, ch), gwq, gwk, gwv

# problem: 073
def run(encoder_hidden_states, norm_weight, k_proj_weight, v_proj_weight, eps):
    b, s, h = encoder_hidden_states.shape
    normed = _norm('norm_hhh_001000', _aligned(encoder_hidden_states), h, eps, weight=_aligned(norm_weight))
    outs = []
    for w in (k_proj_weight, v_proj_weight):
        o = torch.empty((b, 2, s, 64), dtype=torch.float16, device=normed.device)
        _gemm('gemm_hh_128_00_00000', _opb(normed), _op(_aligned(w)), o, s, 128, h, batches=b, ld=64, o_s1=2 * s * 64,
              col_split=64, col_stride=s * 64)
        outs.append(o)
    return outs[0], outs[1]

# problem: 074
def run(hidden_states, gate_up_weight, down_weight):
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    rows, k = x.shape
    inter = down_weight.shape[1]
    xa = _tf32x3(x, True)
    w = _aligned(gate_up_weight)
    gate = torch.empty((rows, inter), dtype=torch.float32, device=x.device)
    _gemm('gemm_tf_256_00_00000', xa, _tf32x3(w[:inter], False), gate, rows, inter, k, k_groups=3)
    act = torch.empty_like(gate)
    _gemm('gemm_tf_256_00_00200', xa, _tf32x3(w[inter:], False), act, rows, inter, k, k_groups=3, residual=gate)
    out = torch.empty((rows, down_weight.shape[0]), dtype=torch.float32, device=x.device)
    _gemm3('gemm_tf_256_00_00000', act, down_weight, out, rows, down_weight.shape[0], inter)
    return out.view(*hidden_states.shape[:-1], down_weight.shape[0])

# problem: 075
def run(hidden_states, position_ids, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, inv_freq, is_causal):
    b, s, hid = hidden_states.shape
    dev = hidden_states.device
    xa = _tf32x3(hidden_states.reshape(b * s, hid), True)
    proj = []
    for w in (q_proj_weight, k_proj_weight, v_proj_weight):
        o = torch.empty((b * s, w.shape[0]), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_00000', xa, _tf32x3(w, False), o, b * s, w.shape[0], hid, k_groups=3)
        proj.append(o)
    q, k, v = proj
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    pos, inv = _aligned(position_ids), _aligned(inv_freq)
    _hnr('hnr_fff_64_0110', q, qr, b * s * 16, 16, 1, positions=pos, inv_freq=inv)
    _hnr('hnr_fff_64_0110', k, kr, b * s * 4, 4, 1, positions=pos, inv_freq=inv)
    ldp = (s + 7) // 8 * 8
    qs = _split3(qr, 1024, 1024, True)
    ks = _split3(kr, 256, 256, False)
    scores = torch.empty((b * 16, s, ldp), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_256_00_00000',
          _Op(qs, [64, 48, b * s], [256, 3 * 1024 * 4], z_div=16, z_mul=s, y_mod=16, y_div=1, y_mul=1, y_kstep=16),
          _Op(ks, [64, 12, b * s], [256, 3 * 256 * 4], z_div=16, z_mul=s, y_mod=16, y_div=4, y_mul=1, y_kstep=4),
          scores, s, ldp, 64, batches=b * 16, k_groups=3, ld=ldp, o_s1=s * ldp)
    probs = torch.empty((b, 16, s, s), dtype=torch.float32, device=dev)
    _launch('softmax_ff_01' if bool(is_causal) else 'softmax_ff_00', [scores, scores, scores, probs], (b * 16 * s, 1, 1),
            {'I': [s, ldp, s, _BIG, 1, s, 0], 'f': [1.0]})
    ld4 = (s + 3) // 4 * 4
    ps = _split3(probs, s, ld4, True)
    vs = _split3(v, 256, 256, False)
    attn = torch.empty((b, s, 16, 64), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_64_01_00000', _Op(ps, [ld4, 3, b * 16 * s], [ld4 * 4, 3 * ld4 * 4], z_div=1, z_mul=s, y_kstep=1),
          _Op(vs, [64, 12, b * s], [256, 3 * 256 * 4], z_div=16, z_mul=s, y_mod=16, y_div=4, y_mul=1, y_kstep=4),
          attn, s, 64, s, batches=b * 16, k_groups=3, ld=1024, o_div=16, o_s1=s * 1024, o_s2=64)
    out = torch.empty((b * s, hid), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_00_00000', attn.view(b * s, 1024), o_proj_weight, out, b * s, hid, 1024)
    return out.view(b, s, hid), probs

# problem: 076
def run(hidden_states, routing_weights, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, alpha, limit):
    b, s, h = hidden_states.shape
    t = b * s
    e, _, i2 = gate_up_proj.shape
    inter = i2 // 2
    dev = hidden_states.device
    xa = _tf32x3(hidden_states.reshape(t, h), True)
    wsrc, dsrc = _aligned(gate_up_proj), _aligned(down_proj)
    wgu = torch.empty((e * 3, h, i2), dtype=torch.float32, device=dev)
    wd = torch.empty((e * 3, inter, h), dtype=torch.float32, device=dev)
    for j in range(e):
        _launch('tf32_split_b', [wsrc[j], wgu[3 * j:3 * j + 3]], _blocks(3 * h * i2, 256),
                {'I': [3 * h * i2, h * i2, h * i2, h * i2]})
        _launch('tf32_split_b', [dsrc[j], wd[3 * j:3 * j + 3]], _blocks(3 * inter * h, 256),
                {'I': [3 * inter * h, inter * h, inter * h, inter * h]})
    gu = torch.empty((e, t, i2), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_256_01_10000', xa, _opb(wgu, div=1, mul=3, kstep=1), gu, t, i2, h, batches=e, k_groups=3, ld=i2,
          o_s1=t * i2, bias=_aligned(gate_up_proj_bias), bias_stride=i2)
    rw = _aligned(routing_weights)
    gated = torch.empty((e, t, inter), dtype=torch.float32, device=dev)
    _launch('l1_076_glu', [gu, rw, gated], _blocks(gated.numel(), 256),
            {'I': [gated.numel(), t, inter, e], 'f': [float(alpha), float(limit)]})
    ga = _split3(gated, t * inter, t * inter, True).view(e * 3, t, inter)
    bias_term = torch.empty((t, h), dtype=torch.float32, device=dev)
    _gemm3('gemm_tf_256_01_00000', rw, down_proj_bias, bias_term, t, h, e)
    out = torch.empty((t, h), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_256_01_00100', _opb(ga, div=1, mul=0, kstep=1), _opb(wd, div=1, mul=0, kstep=1), out, t, h, inter,
          k_groups=3 * e, residual=bias_term)
    return out.view(b, s, h)

# problem: 077
def run(hidden_states, weight):
    x2 = hidden_states.reshape(-1, hidden_states.shape[-1])
    rows, k = x2.shape
    n = weight.shape[0]
    n8 = (n + 7) // 8 * 8
    out = torch.empty((rows, n8), dtype=hidden_states.dtype, device=x2.device)
    _gemm('gemm_hh_256_00_00000', _op(x2), _op(_aligned(weight)), out, rows, n8, k)
    return out.view(*hidden_states.shape[:-1], n8)[..., :n]
