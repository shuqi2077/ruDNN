# Quant batch A: FP8 blockwise (1x128 activations, 128x128 weights) solutions:
# RUDA fake-quant kernels reproduce the reference quantize/dequantize numerics,
# then the tcgen05 bf16 GEMM accumulates in f32 with fused bias / gating.

# problem: 001
def run(attn_output, o_proj_weight, o_proj_bias):
    return _linear('gemm_bb_256_00_10000', _fq_act(attn_output), _fq_weight(o_proj_weight), bias=_aligned(o_proj_bias))

# problem: 003
def run(x, scale_x, gate_proj_weight, scale_gate, up_proj_weight, scale_up):
    kb = x.shape[1] // 128
    xd = _dq_fp8(x, scale_x, 1, kb, 1)
    gd = _dq_fp8(gate_proj_weight, scale_gate, 128, kb, 1)
    ud = _dq_fp8(up_proj_weight, scale_up, 128, kb, 1)
    gate = _linear('gemm_bb_256_00_00001', xd, gd)
    return _linear('gemm_bb_256_00_00201', xd, ud, residual=gate)

# problem: 004
def run(hidden_states, routing_weight, gate_up_weight, down_weight):
    t, h = hidden_states.shape
    inter = down_weight.shape[1]
    xq = _fq_act(hidden_states)
    wq = _fq_weight(gate_up_weight)
    gate = _linear('gemm_bb_256_00_00001', xq, wq[:inter])
    act = _linear('gemm_bb_256_00_00201', xq, wq[inter:], residual=gate)
    y = _linear('gemm_bb_256_00_00001', _fq_act(act), _fq_weight(down_weight))
    out = torch.empty_like(y)
    n = out.numel()
    _launch('bin_bbb_2', [y, _aligned(routing_weight), out], _blocks(n, 256),
            {'I': [1, t, h, n, 0, 0, h, 1, 0, 0, 1, 0], 'f': [1.0]})
    return out

# problem: 005
def run(hidden_states, gate_weight, scale_hidden, scale_weight):
    kb = hidden_states.shape[1] // 128
    xd = _dq_fp8(hidden_states, scale_hidden, 1, kb, 1)
    wd = _dq_fp8(gate_weight, scale_weight, 128, kb, 1)
    n = wd.shape[0]
    n8 = (n + 7) // 8 * 8
    out = torch.empty((xd.shape[0], n8), dtype=torch.bfloat16, device=xd.device)
    _gemm('gemm_bb_128_00_00000', _op(xd), _op(wd), out, xd.shape[0], n8, xd.shape[1])
    return out[:, :64]

# problem: 006
def run(attn_output, weight, scale_attn, scale_weight):
    nb = weight.shape[0] // 128
    return _linear('gemm_bb_256_00_00000', _fq_act(attn_output, scale_attn), _fq_weight(weight, scale_weight, s_n=1, s_k=nb))

# problem: 007
def run(hidden_states, qkv_weight, qkv_bias):
    s = hidden_states.shape[0]
    qkv = _linear('gemm_bb_256_00_10000', _fq_act(hidden_states), _fq_weight(qkv_weight), bias=_aligned(qkv_bias))
    v = qkv.view(s, 3, 16, 96)
    return v[:, 0], v[:, 1], v[:, 2]
