# problem: 002
def run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
    h = _conv2d_f32(x, conv1_weight, pad=1)
    h = _unary('un_ff_0', _group_norm_f32(h, norm1_weight, norm1_bias, 32, eps))
    h = _conv2d_f32(h, conv2_weight, pad=1)
    h = _unary('un_ff_0', _group_norm_f32(h, norm2_weight, norm2_bias, 32, eps))
    return _binary('bin_fff_0', h, _aligned(x))

# problem: 021
def run(hidden_states, cu_seqlens, cos, sin, qkv_weight, qkv_bias, proj_weight, proj_bias):
    return _varlen_f32(hidden_states, cu_seqlens, cos, sin, qkv_weight, qkv_bias, proj_weight, proj_bias, 16, 80)

# problem: 007
def run(x):
    length = x.shape[-1] * 2
    spectrum = _fft_real(x, length)
    bins = length // 2 + 1
    re = torch.empty((*x.shape[:-1], bins), dtype=torch.float32, device=x.device)
    im = torch.empty_like(re)
    _launch('fft_real_split', [spectrum, re, im], _blocks(re.numel(), 256), {'I': [length, bins], 'f': [1.0 / length]})
    return re, im

# problem: 023
def run(grid_thw, pos_embed_weight, inv_freq):
    grids = grid_thw.cpu().tolist()
    tokens = sum(t * h * w for t, h, w in grids)
    hidden, dim = pos_embed_weight.shape[1], inv_freq.numel() * 4
    out = torch.empty((tokens, hidden), dtype=pos_embed_weight.dtype, device=pos_embed_weight.device)
    cos = torch.empty((tokens, dim), dtype=torch.float32, device=pos_embed_weight.device)
    sin = torch.empty_like(cos)
    offset = 0
    for t, h, w in grids:
        _launch('vision_grid_embedding', [_aligned(pos_embed_weight), _aligned(inv_freq), out, cos, sin],
                _blocks(t * h * w * hidden, 256), {'I': [t, h, w, hidden, offset]})
        offset += t * h * w
    return out, cos, sin

# problem: 026
def run(video, patch_projection_weight, patch_projection_bias, spatial_pos_embedding, temporal_pos_embedding,
        norm_weight, norm_bias, eps):
    b, frames, channels, h, w = video.shape
    hidden, _, kt, kh, kw = patch_projection_weight.shape
    assert kt == 1
    cols, oh, ow = _im2col('im2col_h', video.reshape(b * frames, channels, h, w), kh, kw, stride=kh)
    patches = _linear('gemm_hh_256_00_10000', cols, _aligned(patch_projection_weight).view(hidden, -1),
                      bias=_aligned(patch_projection_bias)).view(b, frames, oh * ow, hidden)
    patches = _binary('bin_hhh_0', patches, spatial_pos_embedding.unsqueeze(1))
    patches = _binary('bin_hhh_0', patches, temporal_pos_embedding).view(b, frames * oh * ow, hidden)
    out = torch.empty_like(patches)
    _launch('norm_arithmetic_h', [patches, _aligned(norm_weight), _aligned(norm_bias), out],
            (b * frames * oh * ow, 1, 1), {'I': [hidden], 'f': [float(eps)]})
    return out

# problem: 027
def run(hidden_states, qkv_weight, qkv_bias, out_weight, out_bias, temporal_freqs, spatial_freqs, scale):
    b, frames, patches, hidden = hidden_states.shape
    seq = frames * patches
    side = __import__('math').isqrt(patches)
    if side * side < patches:
        side += 1
    qkv = _linear_f32(hidden_states.reshape(b, seq, hidden), qkv_weight, qkv_bias)
    q = torch.empty((b * seq, hidden), dtype=torch.float32, device=hidden_states.device)
    k, v = torch.empty_like(q), torch.empty_like(q)
    _launch('video_rope_f32', [qkv, _aligned(temporal_freqs), _aligned(spatial_freqs), q, k, v],
            _blocks(q.numel(), 256), {'I': [seq, patches, side]})
    out = _attn_f32(q, k, v, b, seq, seq, 16, 16, 64, scale=float(scale))
    return _linear_f32(out.reshape(b, seq, hidden), out_weight, out_bias).view(b, frames, patches, hidden)

# problem: 040
def run(x, conv_in_weight, conv_in_bias, conv_out_weight, conv_out_bias):
    h = _conv2d_f32(x, conv_in_weight, conv_in_bias, pad=1)
    return _binary('bin_fff_0', _conv2d_f32(h, conv_out_weight, conv_out_bias, pad=1), _aligned(x))

# problem: 041
def run(x, weight, bias):
    return _conv2d_f32(x, weight, bias, stride=2, pad=1)
