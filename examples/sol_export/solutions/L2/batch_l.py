# problem: 022
def run(video_latents, text_embeddings, temporal_norm_weight, temporal_norm_bias, temporal_qkv_weight,
        temporal_qkv_bias, temporal_out_proj_weight, temporal_out_proj_bias, spatial_norm_weight,
        spatial_norm_bias, spatial_qkv_weight, spatial_qkv_bias, spatial_out_proj_weight, spatial_out_proj_bias,
        cross_attn_norm_weight, cross_attn_norm_bias, cross_attn_q_weight, cross_attn_q_bias,
        cross_attn_kv_weight, cross_attn_kv_bias, cross_attn_out_proj_weight, cross_attn_out_proj_bias,
        ffn_norm_weight, ffn_norm_bias, ffn_fc1_weight, ffn_fc1_bias, ffn_fc2_weight, ffn_fc2_bias,
        num_frames_scalar, num_spatial_tokens_scalar):
    b, seq, hidden = video_latents.shape
    frames, patches = int(num_frames_scalar), int(num_spatial_tokens_scalar)
    n = _layernorm_f32(video_latents, temporal_norm_weight, temporal_norm_bias)
    qkv = _linear_f32(n, temporal_qkv_weight, temporal_qkv_bias).reshape(b, frames, patches, 3, hidden)
    q, k, v = [qkv[:, :, :, i].permute(0, 2, 1, 3).reshape(b * patches, frames, hidden) for i in range(3)]
    a = _attention_f32(q, k, v, 16).reshape(b, patches, frames, hidden).transpose(1, 2).reshape(b, seq, hidden)
    x = _binary('bin_fff_0', video_latents, _linear_f32(a, temporal_out_proj_weight, temporal_out_proj_bias))
    n = _layernorm_f32(x, spatial_norm_weight, spatial_norm_bias).reshape(b * frames, patches, hidden)
    q, k, v = _linear_f32(n, spatial_qkv_weight, spatial_qkv_bias).chunk(3, dim=-1)
    a = _attention_f32(q, k, v, 16).reshape(b, seq, hidden)
    x = _binary('bin_fff_0', x, _linear_f32(a, spatial_out_proj_weight, spatial_out_proj_bias))
    n = _layernorm_f32(x, cross_attn_norm_weight, cross_attn_norm_bias)
    q = _linear_f32(n, cross_attn_q_weight, cross_attn_q_bias)
    k, v = _linear_f32(text_embeddings, cross_attn_kv_weight, cross_attn_kv_bias).chunk(2, dim=-1)
    a = _attention_f32(q, k, v, 16)
    x = _binary('bin_fff_0', x, _linear_f32(a, cross_attn_out_proj_weight, cross_attn_out_proj_bias))
    n = _layernorm_f32(x, ffn_norm_weight, ffn_norm_bias)
    h = _unary('un_ff_2', _linear_f32(n, ffn_fc1_weight, ffn_fc1_bias))
    return _binary('bin_fff_0', x, _linear_f32(h, ffn_fc2_weight, ffn_fc2_bias))

# problem: 023
def run(x, input_conv_weight, input_conv_bias, down1_gn_weight, down1_gn_bias, down1_conv_weight,
        down1_conv_bias, down1_res_weight, down1_res_bias, down2_gn_weight, down2_gn_bias, down2_conv_weight,
        down2_conv_bias, down2_res_weight, down2_res_bias, down3_gn_weight, down3_gn_bias, down3_conv_weight,
        down3_conv_bias, down3_res_weight, down3_res_bias, latent_gn_weight, latent_gn_bias,
        latent_conv_weight, latent_conv_bias, eps):
    h = _conv_nd(x, input_conv_weight, input_conv_bias, pad=1)
    blocks = [(down1_gn_weight, down1_gn_bias, down1_conv_weight, down1_conv_bias, down1_res_weight, down1_res_bias, 8, 2),
              (down2_gn_weight, down2_gn_bias, down2_conv_weight, down2_conv_bias, down2_res_weight, down2_res_bias, 16, 2),
              (down3_gn_weight, down3_gn_bias, down3_conv_weight, down3_conv_bias, down3_res_weight, down3_res_bias, 32, (1, 2, 2))]
    for nw, nb, cw, cb, rw, rb, groups, stride in blocks:
        residual = _conv_nd(h, rw, rb, stride=stride)
        n = _unary('un_ff_0', _group_norm_f32(h, nw, nb, groups, eps))
        h = _binary('bin_fff_0', residual, _conv_nd(n, cw, cb, stride=stride, pad=1))
    h = _unary('un_ff_0', _group_norm_f32(h, latent_gn_weight, latent_gn_bias, 32, eps))
    return _conv_nd(h, latent_conv_weight, latent_conv_bias, pad=1)

# problem: 033
def run(x, encoder_conv0_weight, encoder_conv0_bias, downsample_conv0_weight, downsample_conv0_bias,
        encoder_conv1_weight, encoder_conv1_bias, downsample_conv1_weight, downsample_conv1_bias,
        encoder_conv2_weight, encoder_conv2_bias, bottleneck_weight, bottleneck_bias, upsample_conv0_weight,
        upsample_conv0_bias, decoder_conv0_weight, decoder_conv0_bias, upsample_conv1_weight, upsample_conv1_bias,
        decoder_conv1_weight, decoder_conv1_bias, output_conv_weight, output_conv_bias):
    e0 = _conv2d_f32(x, encoder_conv0_weight, encoder_conv0_bias, pad=1)
    e1 = _conv2d_f32(_conv2d_f32(e0, downsample_conv0_weight, downsample_conv0_bias, stride=2, pad=1), encoder_conv1_weight, encoder_conv1_bias, pad=1)
    e2 = _conv2d_f32(_conv2d_f32(e1, downsample_conv1_weight, downsample_conv1_bias, stride=2, pad=1), encoder_conv2_weight, encoder_conv2_bias, pad=1)
    h = _conv2d_f32(e2, bottleneck_weight, bottleneck_bias, pad=1)
    h = _conv_nd(h, upsample_conv0_weight, upsample_conv0_bias, stride=2, transpose=True)
    h = _conv2d_f32(torch.cat((h, e1), dim=1), decoder_conv0_weight, decoder_conv0_bias, pad=1)
    h = _conv_nd(h, upsample_conv1_weight, upsample_conv1_bias, stride=2, transpose=True)
    h = _conv2d_f32(torch.cat((h, e0), dim=1), decoder_conv1_weight, decoder_conv1_bias, pad=1)
    return _conv2d_f32(h, output_conv_weight, output_conv_bias, pad=1)

# problem: 050
def run(hidden_states, temb, resnet1_norm1_weight, resnet1_norm1_bias, resnet1_conv1_weight, resnet1_conv1_bias,
        resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias, resnet1_norm2_weight, resnet1_norm2_bias,
        resnet1_conv2_weight, resnet1_conv2_bias, attn_group_norm_weight, attn_group_norm_bias,
        attn_to_q_weight, attn_to_q_bias, attn_to_k_weight, attn_to_k_bias, attn_to_v_weight, attn_to_v_bias,
        attn_to_out_weight, attn_to_out_bias, resnet2_norm1_weight, resnet2_norm1_bias, resnet2_conv1_weight,
        resnet2_conv1_bias, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias, resnet2_norm2_weight,
        resnet2_norm2_bias, resnet2_conv2_weight, resnet2_conv2_bias, eps):
    x = _resnet2d(hidden_states, temb, resnet1_norm1_weight, resnet1_norm1_bias, resnet1_conv1_weight,
                 resnet1_conv1_bias, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
                 resnet1_norm2_weight, resnet1_norm2_bias, resnet1_conv2_weight, resnet1_conv2_bias, eps)
    b, channels, height, width = x.shape
    n = _group_norm_f32(x, attn_group_norm_weight, attn_group_norm_bias, 32, eps).view(b, channels, -1).transpose(1, 2)
    q, k, v = [_linear_f32(n, w, bias) for w, bias in ((attn_to_q_weight, attn_to_q_bias), (attn_to_k_weight, attn_to_k_bias), (attn_to_v_weight, attn_to_v_bias))]
    a = _linear_f32(_attention_f32(q, k, v, 1), attn_to_out_weight, attn_to_out_bias).transpose(1, 2).reshape(x.shape)
    x = _binary('bin_fff_0', x, a)
    return _resnet2d(x, temb, resnet2_norm1_weight, resnet2_norm1_bias, resnet2_conv1_weight,
                    resnet2_conv1_bias, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
                    resnet2_norm2_weight, resnet2_norm2_bias, resnet2_conv2_weight, resnet2_conv2_bias, eps)

# problem: 055
def run(input_features, conv1_weight, conv1_bias, conv2_weight, conv2_bias, embed_positions_weight,
        self_attn_layer_norm_weight, self_attn_layer_norm_bias, q_proj_weight, q_proj_bias, k_proj_weight,
        v_proj_weight, v_proj_bias, out_proj_weight, out_proj_bias, final_layer_norm_weight,
        final_layer_norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    x = _unary('un_bb_2', _conv_nd(input_features, conv1_weight, conv1_bias, pad=1))
    x = _unary('un_bb_2', _conv_nd(x, conv2_weight, conv2_bias, stride=2, pad=1)).transpose(1, 2)
    x = _binary('bin_bbb_0', x, embed_positions_weight)
    b, seq, hidden = x.shape
    n = _norm('norm_bbb_101100', _aligned(x), hidden, 1e-5, weight=_aligned(self_attn_layer_norm_weight), bias=_aligned(self_attn_layer_norm_bias))
    q = _linear('gemm_bb_256_00_10000', n, q_proj_weight, bias=_aligned(q_proj_bias))
    scale = torch.tensor(0.0625, dtype=torch.bfloat16, device=x.device)
    q = _binary('bin_bbb_2', q, scale).view(b, seq, 20, 256)
    k = _linear('gemm_bb_256_00_00000', n, k_proj_weight).view(b, seq, 20, 256)
    v = _linear('gemm_bb_256_00_10000', n, v_proj_weight, bias=_aligned(v_proj_bias)).view(b, seq, 20, 256)
    a, _ = _attn_bf16(q, k, v, scale=1.0)
    x = _binary('bin_bbb_0', x, _linear('gemm_bb_256_00_10000', a.view(b, seq, hidden), out_proj_weight, bias=_aligned(out_proj_bias)))
    n = _norm('norm_bbb_101100', _aligned(x), hidden, 1e-5, weight=_aligned(final_layer_norm_weight), bias=_aligned(final_layer_norm_bias))
    h = _unary('un_bb_2', _linear('gemm_bb_256_00_10000', n, fc1_weight, bias=_aligned(fc1_bias)))
    return _binary('bin_bbb_0', x, _linear('gemm_bb_256_00_10000', h, fc2_weight, bias=_aligned(fc2_bias)))

# problem: 057
def run(x, x_mask, reverse, transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight,
        transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias, transform_1_conv0_weight,
        transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight,
        transform_1_conv2_bias, transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight,
        transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias, transform_3_conv0_weight,
        transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight,
        transform_3_conv2_bias):
    transforms = [(transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
                  (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
                  (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
                  (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias)]
    for w0, b0, w1, b1, w2, b2 in reversed(transforms) if reverse else transforms:
        a, b = x.chunk(2, dim=1)
        pad = w0.shape[-1] // 2
        h = _unary('un_ff_5', _conv_nd(a, w0, b0, pad=pad))
        h = _unary('un_ff_5', _conv_nd(h, w1, b1, pad=pad))
        h = _binary('bin_fff_2', _conv_nd(h, w2, b2, pad=pad), x_mask)
        b = _binary('bin_fff_1' if reverse else 'bin_fff_0', b, h)
        x = _binary('bin_fff_2', torch.cat((a, b), dim=1), x_mask)
    return x
