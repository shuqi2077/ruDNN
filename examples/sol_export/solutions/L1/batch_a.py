# L1 batch A: per-problem launch glue for RUDA kernels in ruDNN/src/standalone/sol.rs.
# Each `# problem: NNN` section becomes the body of NNN/submission.py
# (after the launcher prelude and the PTX bundle of the kernels it names).

# problem: 011
def run(position_ids, inv_freq, attention_scaling):
    batch, seq = position_ids.shape
    out = torch.empty((batch, seq, 128, 2), dtype=torch.bfloat16, device=position_ids.device)
    _launch('l1_011_rope_table', [position_ids.contiguous(), inv_freq.contiguous(), out],
            _blocks(batch * seq * 16, 256), {'f': [float(attention_scaling)]})
    return out

# problem: 012
def run(freqs, attention_scaling):
    batch, seq, half = freqs.shape
    cos = torch.empty((batch, seq, 2 * half), dtype=torch.bfloat16, device=freqs.device)
    sin = torch.empty_like(cos)
    _launch('l1_012_cos_sin', [freqs.contiguous(), cos, sin], _blocks(batch * seq * 16, 256),
            {'f': [float(attention_scaling)]})
    return cos, sin

# problem: 013
def run(grad_output, x, normalized, rstd, weight):
    hidden = grad_output.shape[-1]
    rows = grad_output.numel() // hidden
    grad_x = torch.empty(grad_output.shape, dtype=torch.bfloat16, device=grad_output.device)
    grad_residual = torch.empty_like(grad_x)
    blocks = max(1, min(rows, _sm_count(grad_output.device) * 6))
    partial = torch.empty((blocks, hidden), dtype=torch.float32, device=grad_output.device)
    grad_weight = torch.empty(hidden, dtype=torch.float32, device=grad_output.device)
    _launch('l1_013_rms_bwd', [grad_output.contiguous(), normalized.contiguous(), rstd.contiguous(), weight.contiguous(),
                               grad_x, grad_residual, partial], (blocks, 1, 1))
    _launch('col_sum', [partial, grad_weight], _blocks(hidden, 256))
    return grad_x, grad_residual, grad_weight

# problem: 014
def run(grad_cos, grad_sin, emb, inv_freq_expanded, attention_scaling):
    batch, seq, _ = grad_cos.shape
    out = torch.empty((batch, seq), dtype=torch.float32, device=grad_cos.device)
    _launch('l1_014_rope_pos_grad', [grad_cos.contiguous(), grad_sin.contiguous(), emb.contiguous(),
                                     inv_freq_expanded.contiguous(), out],
            _blocks(batch * seq, 8), {'I': [seq], 'f': [float(attention_scaling)]})
    return out

# problem: 016
def run(rope_theta):
    out = torch.empty(64, dtype=torch.float32, device='cuda')
    _launch('l1_016_inv_freq', [out], (1, 1, 1), {'f': [float(rope_theta)]})
    return out

# problem: 019
def run(grad_q_embed, grad_k_embed, q, k, embeddings):
    seq = q.shape[0]
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_embeddings = torch.empty_like(embeddings)
    _launch('l1_019_rope_vision_bwd', [grad_q_embed.contiguous(), grad_k_embed.contiguous(), q.contiguous(), k.contiguous(),
                                       embeddings.contiguous(), grad_q, grad_k, grad_embeddings], (seq, 1, 1))
    return grad_q, grad_k, grad_embeddings

# problem: 024
def run(grad_cos, grad_sin, pos_ids, inv_freq, emb):
    patches = grad_cos.shape[0]
    blocks = max(1, min((patches + 15) // 16, _sm_count(grad_cos.device) * 4))
    partial = torch.empty((blocks, 18), dtype=torch.float32, device=grad_cos.device)
    out = torch.empty(inv_freq.shape, dtype=inv_freq.dtype, device=grad_cos.device)
    _launch('l1_024_partial', [grad_cos.contiguous(), grad_sin.contiguous(), pos_ids.contiguous(), emb.contiguous(), partial],
            (blocks, 1, 1))
    _launch('col_sum', [partial, out], (1, 1, 1))
    return out

# problem: 025
def run(x):
    out = torch.empty_like(x)
    _launch('l1_025_gelu', [x.contiguous(), out], _blocks(x.numel() // 4, 256))
    return out

# problem: 028
def run(batch_size_scalar, seq_length_scalar, past_key_values_length_scalar):
    batch = int(batch_size_scalar)
    target = int(seq_length_scalar)
    past = int(past_key_values_length_scalar)
    source = target + past
    total = batch * 64 * target * source
    full = torch.empty(total // 4, dtype=torch.int32, device='cuda')
    sliding = torch.empty_like(full)
    vectors = total // 16
    grid = max(1, min((vectors + 255) // 256, _sm_count(full.device) * 32))
    _launch('l1_028_masks', [full, sliding], (grid, 1, 1), {'I': [target, source, past]})
    shape = (batch, 64, target, source)
    return full.view(torch.bool).view(shape), sliding.view(torch.bool).view(shape)

# problem: 033
def run(sublayer_output, residual, weight, eps):
    out = torch.empty_like(sublayer_output)
    rows = sublayer_output.numel() // 4096
    _launch('rms_post_4096', [sublayer_output.contiguous(), residual.contiguous(), weight.contiguous(), out],
            (rows, 1, 1), {'f': [float(eps)]})
    return out

# problem: 038
def run(query, key, weight_q, weight_k, eps):
    rows = query.numel() // 128
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    _launch('l1_038_head_rms', [query.contiguous(), key.contiguous(), weight_q.contiguous(), weight_k.contiguous(),
                                query_out, key_out], ((rows + 7) // 8, 2, 1), {'f': [float(eps)]})
    return query_out, key_out

# problem: 039
def run(grad_output, x, weight, mean, std):
    samples, channels, height, width = x.shape
    spatial = height * width
    planes = samples * channels
    chunk = 32768
    chunks = (spatial + chunk - 1) // chunk
    device = x.device
    partial = torch.empty(planes * chunks * 3, dtype=torch.float32, device=device)
    coefficients = torch.empty(planes * 3, dtype=torch.float32, device=device)
    grad_weight = torch.empty(channels, dtype=torch.float32, device=device)
    grad_bias = torch.empty(channels, dtype=torch.float32, device=device)
    grad_input = torch.empty_like(x)
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    mean = mean.contiguous()
    _launch('l1_039_partial', [grad_output, x, mean, partial], (chunks, planes, 1), {'I': [spatial, chunk]})
    _launch('l1_039_finalize', [partial, weight.contiguous(), std.contiguous(), coefficients, grad_weight, grad_bias],
            (channels, 1, 1), {'I': [samples, chunks, spatial]})
    _launch('l1_039_apply', [grad_output, x, mean, coefficients, grad_input], (chunks, planes, 1), {'I': [spatial, chunk]})
    return grad_input, grad_weight, grad_bias

# problem: 046
def run(attn_weights):
    columns = attn_weights.shape[-1]
    rows = attn_weights.numel() // columns
    out = torch.empty_like(attn_weights)
    _launch('l1_046_softcap_softmax', [attn_weights.contiguous(), out], ((rows + 7) // 8, 1, 1),
            {'I': [columns], 'f': [30.0]})
    return out

# problem: 069
def run(hidden_states, residual, weight, eps):
    out = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // 8192
    _launch('rms_pre_8192', [hidden_states.contiguous(), residual.contiguous(), weight.contiguous(), out],
            (rows, 1, 1), {'f': [float(eps)]})
    return out
