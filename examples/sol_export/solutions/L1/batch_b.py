# L1 batch B launch glue (kernels in ruDNN/src/standalone/sol.rs).

# problem: 056
def run(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max):
    rows = grad_output.numel() // 40
    grad_dt = torch.empty_like(grad_output)
    blocks = max(1, min((rows + 7) // 8, _sm_count(grad_output.device) * 6))
    partial = torch.empty((blocks, 40), dtype=torch.float32, device=grad_output.device)
    grad_bias = torch.empty(40, dtype=torch.bfloat16, device=grad_output.device)
    _launch('l1_056_dt_bwd', [grad_output.contiguous(), dt_with_bias.contiguous(), dt_activated.contiguous(), grad_dt, partial],
            (blocks, 1, 1), {'f': [float(time_step_min), float(time_step_max)]})
    _launch('col_sum_bf16', [partial, grad_bias], (1, 1, 1))
    return grad_dt, grad_bias

# problem: 059
def run(scores):
    tokens = scores.shape[0]
    masked = torch.empty_like(scores)
    group_mask = torch.empty((tokens, 8), dtype=scores.dtype, device=scores.device)
    _launch('l1_059_group_mask', [scores.contiguous(), masked, group_mask], ((tokens + 7) // 8, 1, 1))
    return masked, group_mask

# problem: 061
def run(grad_output, gate, hidden_states, mask):
    grad_residual = torch.empty_like(grad_output)
    grad_hidden = torch.empty_like(grad_output)
    vectors = grad_output.numel() // 8
    blocks = max(1, min((vectors + 255) // 256, _sm_count(grad_output.device) * 8))
    partial = torch.empty(blocks, dtype=torch.float32, device=grad_output.device)
    grad_gate = torch.empty((), dtype=torch.bfloat16, device=grad_output.device)
    gate = gate.reshape(1).contiguous()
    _launch('l1_061_gate_bwd', [grad_output.contiguous(), gate, hidden_states.contiguous(), mask.contiguous(),
                                grad_residual, grad_hidden, partial], (blocks, 1, 1))
    _launch('l1_061_gate_finish', [partial, gate, grad_gate.view(1)], (1, 1, 1))
    return grad_residual, grad_hidden, grad_gate

# problem: 062
def run(grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position):
    batch, heads, cache_length, _ = grad_key_cache.shape
    fresh = key_states.shape[2]
    device = grad_key_cache.device
    grad_key_cache = grad_key_cache.contiguous()
    grad_value_cache = grad_value_cache.contiguous()
    positions = cache_position.contiguous()
    flags = torch.empty(cache_length, dtype=torch.int32, device=device)
    key_input = torch.empty_like(grad_key_cache)
    value_input = torch.empty_like(grad_value_cache)
    grad_key = torch.empty_like(key_states)
    grad_value = torch.empty(key_states.shape, dtype=grad_value_cache.dtype, device=device)
    grad_cos = torch.empty((batch, fresh, 128), dtype=torch.bfloat16, device=device)
    grad_sin = torch.empty_like(grad_cos)
    _launch('position_flags', [positions, flags], _blocks(cache_length, 256))
    vectors = grad_key_cache.numel() // 8
    blocks = max(1, min((vectors + 255) // 256, _sm_count(device) * 16))
    _launch('l1_062_cache_copy', [grad_key_cache, grad_value_cache, flags, key_input, value_input], (blocks, 1, 1))
    if fresh:
        _launch('l1_062_kv_rope_bwd', [grad_key_cache, grad_value_cache, key_states.contiguous(), cos.contiguous(), sin.contiguous(),
                                       positions, grad_key, grad_value, grad_cos, grad_sin], (fresh, batch, 1),
                {'I': [cache_length]})
    return grad_key, grad_value, grad_cos, grad_sin, key_input, value_input

# problem: 066
def run(grad_output, p_attn, mask, dropout_mask, p_dropout):
    batch, heads, seq, _ = grad_output.shape
    out = torch.empty_like(grad_output)
    use = 1 if p_dropout > 0.0 else 0
    # PyTorch divides by a CPU scalar as a multiply by its float32 reciprocal.
    inverse = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(1.0 - p_dropout, dtype=torch.float32)) if use else 1.0
    _launch('l1_066_softmax_bwd', [grad_output.contiguous(), p_attn.contiguous(), mask.contiguous().view(torch.uint8),
                                   dropout_mask.contiguous().view(torch.uint8), out],
            (batch * heads * seq, 1, 1), {'I': [seq, heads, use], 'f': [inverse]})
    return out

# problem: 068
def run(grad_q_embed, grad_k_embed, q, k, cos, sin):
    batch, heads, seq, _ = q.shape
    kv_heads = k.shape[1]
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_cos = torch.empty_like(cos)
    grad_sin = torch.empty_like(sin)
    _launch('l1_068_rope_gqa_bwd', [grad_q_embed.contiguous(), grad_k_embed.contiguous(), q.contiguous(), k.contiguous(),
                                    cos.contiguous(), sin.contiguous(), grad_q, grad_k, grad_cos, grad_sin],
            (batch * seq, 1, 1), {'I': [seq, heads, kv_heads]})
    return grad_q, grad_k, grad_cos, grad_sin

# problem: 071
def run(key_states, value_states, cos, sin, key_cache, value_cache):
    batch, heads, fresh, dim = key_states.shape
    current = key_cache.shape[2]
    shape = (batch, heads, current + fresh, dim)
    key_out = torch.empty(shape, dtype=key_states.dtype, device=key_states.device)
    value_out = torch.empty(shape, dtype=value_states.dtype, device=key_states.device)
    vectors = key_out.numel() // 8
    blocks = max(1, min((vectors + 255) // 256, _sm_count(key_states.device) * 16))
    _launch('l1_071_kv_append', [key_states.contiguous(), value_states.contiguous(), cos.contiguous(), sin.contiguous(),
                                 key_cache.contiguous(), value_cache.contiguous(), key_out, value_out],
            (blocks, 1, 1), {'I': [current, fresh, heads]})
    return key_out, value_out

# problem: 078
def run(x, weight, bias, eps):
    batch, channels, height, width = x.shape
    spatial = height * width
    chunk = 16384
    chunks = (spatial + chunk - 1) // chunk
    x = x.contiguous()
    partial = torch.empty(batch * channels * chunks * 2, dtype=torch.float64, device=x.device)
    out = torch.empty_like(x)
    _launch('l1_078_gn_partial', [x, partial], (chunks, channels, batch), {'I': [spatial, chunk]})
    _launch('l1_078_gn_apply', [x, partial, weight.contiguous(), bias.contiguous(), out], (chunks, channels, batch),
            {'I': [spatial, chunk, 32], 'f': [float(eps)]})
    return out

# problem: 084
def run(grad_output, x, sigmoid_x):
    out = torch.empty_like(grad_output)
    _launch('l1_084_silu_bwd', [grad_output.contiguous(), x.contiguous(), sigmoid_x.contiguous(), out],
            _blocks(out.numel(), 1024))
    return out

# problem: 085
def run(x):
    rows = x.numel() // x.shape[-1]
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    _launch('l1_085_geglu', [x.contiguous(), out], _blocks(out.numel() // 4, 256))
    return out

# problem: 088
def run(query, key, cos, sin):
    seq = query.shape[2]
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    vectors = query.numel() // 4
    blocks = max(1, min((vectors + 255) // 256, _sm_count(query.device) * 16))
    _launch('l1_088_rope_apply', [query.contiguous(), key.contiguous(), cos.contiguous(), sin.contiguous(), query_out, key_out],
            (blocks, 2, 1), {'I': [seq]})
    return query_out, key_out

# problem: 090
def run(grad_cos, grad_sin, idx_theta):
    out = torch.empty(idx_theta.shape, dtype=torch.float32, device=idx_theta.device)
    _launch('l1_090_theta_bwd', [grad_cos.contiguous(), grad_sin.contiguous(), idx_theta.contiguous(), out],
            _blocks(out.numel() // 4, 256))
    return out

# problem: 094
def run(time_decay, key, time_first, value, max_state, num_state, den_state):
    batch, seq, hidden = key.shape
    out = torch.empty_like(key)
    max_out = torch.empty_like(max_state)
    num_out = torch.empty_like(num_state)
    den_out = torch.empty_like(den_state)
    _launch('l1_094_time_decay', [time_decay.contiguous(), key.contiguous(), time_first.contiguous(), value.contiguous(),
                                  max_state.contiguous(), num_state.contiguous(), den_state.contiguous(),
                                  out, max_out, num_out, den_out], _blocks(batch * hidden, 128), {'I': [seq]})
    return out, max_out, num_out, den_out
