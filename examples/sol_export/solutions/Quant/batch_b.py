# Quant batch B: 002 010 011 012.

# problem: 002
def run(hidden_states, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, q_norm_weight, k_norm_weight, rms_norm_eps):
    b, s, h = hidden_states.shape
    xq = _fq_act(hidden_states.reshape(b * s, h))
    outs = []
    for w, bias, heads, nw in ((q_weight, q_bias, 60, q_norm_weight), (k_weight, k_bias, 8, k_norm_weight), (v_weight, v_bias, 8, None)):
        y = _linear('gemm_bb_256_00_10000', xq, _fq_weight(w), bias=_aligned(bias))
        if nw is not None:
            yn = torch.empty_like(y)
            _hnr('hnr_bbb_128_1000', y, yn, b * s * heads, 1, 1, rms_norm_eps, weight=_aligned(nw))
            y = yn
        outs.append(y.view(b, s, heads, 128).transpose(1, 2))
    return outs[0], outs[1], outs[2]

# problem: 010
def run(attn_output, o_proj_weight):
    return _linear('gemm_bb_256_00_00000', _fq_act(attn_output), _fq_weight(o_proj_weight))

# problem: 011
def run(hidden_states, weight, e_score_correction_bias, scale_x, scale_w, routed_scaling_factor):
    t = hidden_states.shape[0]
    dev = hidden_states.device
    nb = weight.shape[0] // 128
    logits_bf = _linear('gemm_bb_256_00_00000', _fq_act(hidden_states, scale_x), _fq_weight(weight, scale_w, s_n=1, s_k=nb))
    logits = torch.empty((t, 256), dtype=torch.float32, device=dev)
    _launch('cast_bf', [logits_bf, logits], _blocks(t * 32, 256), {'I': [t * 32]})
    bias = torch.empty(256, dtype=torch.float32, device=dev)
    _launch('cast_bf', [_aligned(e_score_correction_bias), bias], (1, 1, 1), {'I': [32]})
    idx = torch.empty((t, 8), dtype=torch.int64, device=dev)
    w = torch.empty((t, 8), dtype=torch.float32, device=dev)
    _launch('l2_049_route', [logits, bias, idx, w], ((t + 7) // 8, 1, 1), {'I': [t], 'f': [float(routed_scaling_factor)]})
    return idx, w

# problem: 012
def run(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight):
    xq = _fq_act(hidden_states)
    gate = _linear('gemm_bb_256_00_00001', xq, _fq_weight(gate_proj_weight))
    act = _linear('gemm_bb_256_00_00201', xq, _fq_weight(up_proj_weight), residual=gate)
    return _linear('gemm_bb_256_00_00000', _fq_act(act), _fq_weight(down_proj_weight))
