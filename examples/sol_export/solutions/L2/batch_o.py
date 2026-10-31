# problem: 043
def run(hidden_states, A, B, C, D, initial_states):
    batch, length, heads, dim = hidden_states.shape
    x, a = _f32(hidden_states), _f32(A).transpose(1, 2)
    b, c = [_f32(t).expand(batch, length, heads, 256) for t in (B, C)]
    y, final = _ssm_chunks(x, a, b, c, _f32(initial_states), 256)
    y = _add(y, _mul(x, _f32(D).view(1, 1, heads, 1)))
    return _affine(y.reshape(batch, length, heads * dim), dtype=torch.bfloat16), _affine(final, dtype=torch.bfloat16)

# problem: 044
def run(hidden_states, dt, A_log, B, C):
    batch, chunks, size, heads, dim = hidden_states.shape
    a = _affine(_unary('un_ff_6', _f32(A_log)), -1.0)
    times = _unary('un_ff_21', _f32(dt)).permute(0, 3, 1, 2)
    discrete = _mul(a.view(1, heads, 1, 1), times)
    prefix, lower = _segment_exp(discrete)
    groups, sd = B.shape[-2:]
    bc, cc = [_f32(t).unsqueeze(4).expand(batch, chunks, size, groups, heads // groups, sd).reshape(batch, chunks, size, heads, sd).permute(0, 1, 3, 2, 4) for t in (B, C)]
    g = _matmul(cc, bc, transpose_b=True).permute(0, 1, 3, 4, 2)
    m = _mul(g, lower.permute(0, 2, 3, 4, 1))
    decay = _unary('un_ff_6', _sub(prefix[..., -1:], prefix))
    return _affine(lower, dtype=torch.bfloat16), _affine(g, dtype=torch.bfloat16), _affine(m, dtype=torch.bfloat16), prefix, decay

# problem: 060
def run(query, key, value, g, beta, scale):
    dtype = query.dtype
    batch, kh, length, kd = key.shape
    vh, vd = value.shape[1], value.shape[-1]
    def normalize(x):
        square = _mul(x, x)
        summed = _sum(square, -1, True)
        inv = _affine(_unary('un_ff_11', _f32(_affine(summed, bias=1e-6))), dtype=dtype)
        return _f32(_mul(x, inv))
    q, k = _affine(normalize(query), scale), normalize(key)
    v, g, beta = _f32(value), _f32(g), _f32(beta)
    padded = (length + 63) // 64 * 64
    q, k = [_pad_axis(t, 2, padded).unsqueeze(1).expand(batch, vh // kh, kh, padded, kd).reshape(batch, vh, padded, kd) for t in (q, k)]
    v, g, beta = _pad_axis(v, 2, padded), _pad_axis(g, 2, padded), _pad_axis(beta, 2, padded)
    chunks = padded // 64
    qc, kc = [t.view(batch, vh, chunks, 64, kd) for t in (q, k)]
    vb = _mul(v, beta.unsqueeze(-1)).view(batch, vh, chunks, 64, vd)
    kb = _mul(k, beta.unsqueeze(-1)).view(batch, vh, chunks, 64, kd)
    gc = g.view(batch, vh, chunks, 64)
    prefix, _ = _segment_exp(gc)
    gram = _aligned(_matmul(kb, kc, transpose_b=True))
    triangular, inverse = torch.empty_like(gram), torch.empty_like(gram)
    _launch('delta_decay_1', [gram, prefix, triangular], _blocks(gram.numel(), 256))
    _launch('delta_inverse_f32', [triangular, inverse], (gram.numel() // 4096, 1, 1))
    transformed = _matmul(inverse, vb)
    cumulative_keys = _matmul(inverse, _mul(kb, _unary('un_ff_6', gc).unsqueeze(-1)))
    recurrent = torch.zeros((batch, vh, kd, vd), dtype=torch.float32, device=value.device)
    output = torch.empty((batch, vh, chunks, 64, vd), dtype=torch.float32, device=value.device)
    for ci in range(chunks):
        qq, kk, gg = qc[:, :, ci], kc[:, :, ci], gc[:, :, ci]
        raw = _aligned(_matmul(qq, kk, transpose_b=True))
        intra = torch.empty_like(raw)
        _launch('delta_decay_0', [raw, _aligned(prefix[:, :, ci]), intra], _blocks(raw.numel(), 256))
        new_v = _sub(transformed[:, :, ci], _matmul(cumulative_keys[:, :, ci], recurrent))
        inter = _matmul(_mul(qq, _unary('un_ff_6', gg).unsqueeze(-1)), recurrent)
        output[:, :, ci].copy_(_add(inter, _matmul(intra, new_v)))
        decayed = _mul(recurrent, _unary('un_ff_6', gg[..., -1:]).unsqueeze(-1))
        token_decay = _unary('un_ff_6', _sub(gg[..., -1:], gg)).unsqueeze(-1)
        recurrent = _add(decayed, _matmul(_mul(kk, token_decay), new_v, transpose_a=True))
    return _affine(output.reshape(batch, vh, padded, vd)[:, :, :length].transpose(1, 2), dtype=dtype)
