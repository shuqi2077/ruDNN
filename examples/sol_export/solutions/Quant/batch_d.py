# Quant batch D: 021-023.

# problem: 021
def run(hidden_states, weight):
    return _linear('gemm_bb_256_00_00000', _fq_act(hidden_states), _fq_weight(weight))

# problem: 022
def run(x, w, scale_x, scale_w):
    kb = x.shape[1] // 128
    return _linear('gemm_bb_256_00_00000', _dq_fp8(x, scale_x, 1, kb, 1), _dq_fp8(w, scale_w, 128, kb, 1))

# problem: 023
def run(hidden_states, B, dt_proj_weight, dt_bias, A_log, time_step_limit_min, time_step_limit_max):
    b, l, h = hidden_states.shape
    g, n = B.shape[2], B.shape[3]
    dev = hidden_states.device
    dt_proj = _linear('gemm_bb_128_00_10000', _fq_act(hidden_states.reshape(b * l, h)), _fq_weight(dt_proj_weight),
                      bias=_aligned(dt_bias))
    dt = torch.empty((b, l, h), dtype=torch.bfloat16, device=dev)
    da = torch.empty((b, l, h), dtype=torch.float32, device=dev)
    db = torch.empty((b, l, h, n), dtype=torch.float32, device=dev)
    _launch('q_023_ssm', [dt_proj, _aligned(A_log), _aligned(B), dt, da, db], _blocks(db.numel(), 256),
            {'I': [db.numel(), h, g, n], 'f': [float(time_step_limit_min), float(time_step_limit_max)]})
    return (dt.unsqueeze(-1).expand(-1, -1, -1, 128), da.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 128, n),
            db.unsqueeze(3).expand(-1, -1, -1, 128, -1))
