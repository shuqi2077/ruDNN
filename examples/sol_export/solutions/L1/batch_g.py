# L1 batch G: 051-057.

# problem: 051
def run(grad_query, grad_key, grad_value, hidden_states, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight,
        query_transposed, key_transposed, q_rstd, k_rstd, q_normed, k_normed, rms_norm_eps):
    dev = hidden_states.device
    b, s, hid = hidden_states.shape
    def norm_bwd(g, n, r, w, heads):
        rows = b * heads * s
        out = torch.empty((b, s, heads * 256), dtype=torch.bfloat16, device=dev)
        blocks = max(1, min((rows + 7) // 8, _sm_count(dev) * 4))
        partial = torch.empty((blocks, 256), dtype=torch.float32, device=dev)
        _launch('l1_051_qknorm_bwd', [_aligned(g), _aligned(n), _aligned(r), _aligned(w), out, partial], (blocks, 1, 1),
                {'I': [rows, heads, s]})
        gw = torch.empty(256, dtype=torch.float32, device=dev)
        _launch('colfin_f', [partial, gw], _blocks(256, 256), {'I': [256, blocks]})
        return out, gw
    gq, gwq = norm_bwd(grad_query, q_normed, q_rstd, q_norm_weight, 4)
    gk, gwk = norm_bwd(grad_key, k_normed, k_rstd, k_norm_weight, 1)
    gv = _aligned(grad_value).view(b * s, 256)
    gq2, gk2 = gq.view(b * s, 1024), gk.view(b * s, 256)
    gh = torch.empty((b * s, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_01_00001', _op(gq2), _op(_aligned(q_weight)), gh, b * s, hid, 1024)
    _gemm('gemm_bb_128_01_00101', _op(gk2), _op(_aligned(k_weight)), gh, b * s, hid, 256, residual=gh)
    _gemm('gemm_bb_128_01_00101', _op(gv), _op(_aligned(v_weight)), gh, b * s, hid, 256, residual=gh)
    hs = _aligned(hidden_states).view(b * s, hid)
    grad_wq = torch.empty((1024, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_11_00000', _op(gq2), _op(hs), grad_wq, 1024, hid, b * s)
    grad_wk = torch.empty((256, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_11_00000', _op(gk2), _op(hs), grad_wk, 256, hid, b * s)
    grad_wv = torch.empty((256, hid), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_128_11_00000', _op(gv), _op(hs), grad_wv, 256, hid, b * s)
    return gh.view(b, s, hid), grad_wq, grad_wk, grad_wv, gwq, gwk

# problem: 052
def run(hidden_states, unembed_proj_1, unembed_proj_2, epsilon):
    _, b, s, h = hidden_states.shape
    hs = _aligned(hidden_states)
    dev = hs.device
    p1 = torch.empty((b * s, h), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_00_00000', _op(hs[1].reshape(b * s, h)), _op(_aligned(unembed_proj_1)), p1, b * s, h, h)
    p2 = torch.empty((b * s, h), dtype=torch.float32, device=dev)
    _gemm('gemm_bf_256_00_00000', _op(hs[2].reshape(b * s, h)), _op(_aligned(unembed_proj_2)), p2, b * s, h, h)
    out = torch.empty((b, s, h), dtype=torch.bfloat16, device=dev)
    _launch('l1_052_collapse', [hs[0], p1, p2, out], (b * s, 1, 1), {'I': [h], 'f': [float(epsilon)]})
    return out

# problem: 053
def run(inputs, target_sparsity):
    p = float(target_sparsity)
    if p == 0.0:
        return inputs
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    bb = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p32 = float(torch.tensor(p, dtype=torch.float32))
    if p32 < 0.02425:
        q = math.sqrt(-2.0 * math.log(p32))
        m = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((dd[0]*q + dd[1])*q + dd[2])*q + dd[3])*q + 1.0)
    elif p32 <= 1.0 - 0.02425:
        q = p32 - 0.5
        r = q * q
        m = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / (((((bb[0]*r + bb[1])*r + bb[2])*r + bb[3])*r + bb[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p32))
        m = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((dd[0]*q + dd[1])*q + dd[2])*q + dd[3])*q + 1.0)
    width = inputs.shape[-1]
    rows = inputs.numel() // width
    out = torch.empty_like(inputs)
    _launch('l1_053_sparse', [_aligned(inputs), out], (rows, 1, 1), {'I': [width], 'f': [m]})
    return out

# problem: 054
def run(hidden_states, q_proj_weight, k_proj_weight, v_proj_weight, q_norm_weight, k_norm_weight, eps):
    b, s, h = hidden_states.shape
    dev = hidden_states.device
    xa = _tf32x3(hidden_states.reshape(b * s, h), True)
    outs = []
    for w in (q_proj_weight, k_proj_weight, v_proj_weight):
        o = torch.empty((b * s, w.shape[0]), dtype=torch.float32, device=dev)
        _gemm('gemm_tf_256_00_00000', xa, _tf32x3(w, False), o, b * s, w.shape[0], h, k_groups=3)
        outs.append(o)
    q, k, v = outs
    qn, kn = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_fff_128_2000', q, qn, b * s * 8, 1, 1, eps, weight=_aligned(q_norm_weight))
    _hnr('hnr_fff_128_2000', k, kn, b * s * 8, 1, 1, eps, weight=_aligned(k_norm_weight))
    return qn.view(b, s, 8, 128), kn.view(b, s, 8, 128), v.view(b, s, 8, 128)

# problem: 055
def run(hidden_states, q_weight, k_weight, v_weight):
    b, s, h = hidden_states.shape
    x = _aligned(hidden_states)
    outs = []
    for w in (q_weight, k_weight, v_weight):
        n = w.shape[0]
        o = torch.empty((b, n // 128, s, 128), dtype=x.dtype, device=x.device)
        _gemm('gemm_bb_256_00_00000', _opb(x), _op(_aligned(w)), o, s, n, h, batches=b, ld=128, o_s1=n * s,
              col_split=128, col_stride=s * 128)
        outs.append(o)
    return outs[0], outs[1], outs[2]

# problem: 057
def run(input_ids, hidden_states, word_embeddings, enorm_weight, hnorm_weight, eh_proj_weight, rms_norm_eps):
    b, s, h = hidden_states.shape
    dev = hidden_states.device
    ids = torch.empty_like(input_ids)
    _launch('shift_ids', [_aligned(input_ids), ids], _blocks(b * s, 256), {'I': [b * s, s]})
    buf = torch.empty((2, b * s, h), dtype=torch.bfloat16, device=dev)
    _launch('gather_b', [_aligned(word_embeddings), ids, buf[0]], _blocks(b * s * h // 8, 256), {'I': [b * s * h // 8, h // 8]})
    _norm('norm_bbb_001001', buf[0], h, rms_norm_eps, weight=_aligned(enorm_weight), out=buf[0])
    _norm('norm_bbb_001001', _aligned(hidden_states), h, rms_norm_eps, weight=_aligned(hnorm_weight), out=buf[1])
    w = _aligned(eh_proj_weight)
    out = torch.empty((b, s, h), dtype=torch.bfloat16, device=dev)
    _gemm('gemm_bb_256_00_00000', _opb(buf, div=1, mul=0, kstep=1),
          _Op(w, [h, 2, h], [h * 2, 2 * h * 2], y_div=1, y_mul=0, y_kstep=1), out, b * s, h, h, k_groups=2)
    return out
