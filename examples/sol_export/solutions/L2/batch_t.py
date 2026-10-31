# problem: 006
def run(input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask):
    b, seq = input_ids.shape
    tokens, ranks = torch.empty_like(input_ids), torch.empty_like(input_ids)
    counts = torch.empty((b, 3), dtype=torch.int64, device=input_ids.device)
    positions = torch.empty((3, b, seq), dtype=torch.int64, device=input_ids.device)
    delta = torch.empty((b, 1), dtype=torch.int64, device=input_ids.device)
    _launch('mrope_compact', [_aligned(input_ids), _aligned(attention_mask), tokens, ranks, counts], (b, 1, 1), {'I': [seq]})
    _launch('mrope_positions', [tokens, ranks, counts, _aligned(image_grid_thw), _aligned(video_grid_thw),
            _aligned(second_per_grid_ts), positions, delta], (b, 1, 1), {'I': [seq]})
    return positions, delta

# problem: 038
def run(hidden_states, mask, q_proj_weight, k_proj_weight, v_proj_weight, pos_proj_weight,
        per_dim_scale, inv_timescales, attention_logits_soft_cap):
    b, length, _ = hidden_states.shape
    blocks = (length + 31) // 32
    q, k, v = [_linear_f32(hidden_states, w).view(b, length, 8, 64) for w in (q_proj_weight, k_proj_weight, v_proj_weight)]
    q = _mul(_affine(q, 0.125 / math.log(2.0)), _unary('un_ff_21', per_dim_scale).view(1, 1, 1, 64))
    qb = _pad_axis(q, 1, blocks * 32).view(b, blocks, 32, 8, 64).permute(0, 3, 1, 2, 4)
    def context(t):
        padded = torch.cat((torch.zeros((b, 127, 8, 64), dtype=t.dtype, device=t.device), t,
                            torch.zeros((b, 31, 8, 64), dtype=t.dtype, device=t.device)), dim=1)
        return padded.unfold(1, 160, 32).movedim(-1, 2).permute(0, 3, 1, 2, 4)
    kb, vb = context(k), context(v)
    timing = torch.empty((1, 128, inv_timescales.numel() * 2), dtype=torch.float32, device=hidden_states.device)
    _launch('audio_relative_timing', [_aligned(inv_timescales), timing], _blocks(timing.numel(), 256))
    pos = _linear_f32(timing, pos_proj_weight).view(128, 8, 64).permute(1, 0, 2)
    ac = _aligned(_matmul(qb, kb, transpose_b=True))
    bd = _aligned(_matmul(qb.reshape(b, 8, blocks * 32, 64), pos, transpose_b=True))
    logits = torch.empty_like(ac)
    _launch('audio_local_logits', [ac, bd, _aligned(mask).view(torch.uint8), logits], _blocks(logits.numel(), 256),
            {'I': [length, blocks], 'f': [float(attention_logits_soft_cap)]})
    result = _matmul(_softmax_raw(logits), vb).permute(0, 2, 3, 1, 4).reshape(b, blocks * 32, 8, 64)
    return result[:, :length]

# problem: 046
def run(hidden_states, pre_norm_weight, pre_norm_bias, to_q_weight, to_kv_weight, to_out_weight,
        to_out_bias, rel_pos_emb_weight, scale):
    b, length, _ = hidden_states.shape
    blocks = (length + 511) // 512
    n = _norm('norm_bbb_101100', _aligned(hidden_states), 1024, 1e-5, weight=_aligned(pre_norm_weight), bias=_aligned(pre_norm_bias))
    n = _pad_axis(n, 1, blocks * 512)
    q = _linear('gemm_bb_256_00_00000', n, to_q_weight).view(b, blocks, 512, 8, 128).transpose(2, 3)
    k, v = _linear('gemm_bb_256_00_00000', n, to_kv_weight).chunk(2, dim=-1)
    k, v = [t.reshape(b, blocks, 512, 8, 128).transpose(2, 3) for t in (k, v)]
    posdot = _aligned(_matmul(_f32(q), _f32(rel_pos_emb_weight), transpose_b=True))
    pos = torch.empty((b, blocks, 8, 512, 512), dtype=torch.bfloat16, device=q.device)
    _launch('conformer_relative_position', [posdot, pos], _blocks(pos.numel(), 256),
            {'I': [blocks, rel_pos_emb_weight.shape[0], length % 512], 'f': [float(scale)]})
    root_scale = math.sqrt(scale)
    scores = _matmul(_affine(_f32(q), root_scale), _affine(_f32(k), root_scale), transpose_b=True)
    p = _softmax_raw(_add(scores, _f32(pos)))
    out = _affine(_matmul(p, _f32(v)), dtype=torch.bfloat16).transpose(2, 3).reshape(b, blocks * 512, 1024)[:, :length]
    return _linear('gemm_bb_256_00_10000', out, to_out_weight, bias=_aligned(to_out_bias))
