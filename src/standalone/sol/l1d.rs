//! L1 batch E kernels (029, 034).
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

use super::common::rnd;

/// L1/029: out[b, s, c] = F(F(h * F(sigmoid(h))) * mask[b, s]) with
/// h = F(conv(F(proj[:, :, c] * mask)) + bias), causal depthwise k = 4 over s.
/// proj rows are 2C wide (conv half first). Output layout (B, S, C).
#[ruda(launch)]
pub fn mamba_conv_gate<F: Float>(proj: &Array<F>, mask: &Array<F>, weight: &Array<F>, bias: &Array<F>, out: &mut Array<F>,
    count: u32, seq: u32, channels: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let c_count = channels as usize;
        let c = i % c_count;
        let row = i / c_count;
        let s = row % seq as usize;
        let mut acc = 0.0f32;
        #[unroll]
        for k in 0usize..4 {
            if s + k >= 3 {
                let r = row + k - 3;
                let v = rnd::<F>(f32::cast_from(proj[r * 2 * c_count + c]) * f32::cast_from(mask[r]));
                acc += f32::cast_from(weight[c * 4 + k]) * v;
            }
        }
        let h = rnd::<F>(acc + f32::cast_from(bias[c]));
        let sg = rnd::<F>(1.0f32 / (1.0f32 + (-h).exp()));
        out[i] = F::cast_from(rnd::<F>(h * sg) * f32::cast_from(mask[row]));
    }
}

/// L1/034: Flux multi-axis RoPE tables (S, 128) for axes dims [16, 56, 56],
/// repeat-interleaved cos / sin of pos[s, axis] / theta^(f / half).
#[ruda(launch)]
pub fn flux_rope_freqs(ids: &Array<f32>, cos: &mut Array<f32>, sin: &mut Array<f32>, count: u32, theta: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let j = i % 128;
        let s = i / 128;
        // axis 0: dims [0, 16) (half 8); axis 1: [16, 72); axis 2: [72, 128) (half 28)
        let mut exponent = f32::cast_from(j / 2) / 8.0f32;
        let mut position = ids[s * 3];
        if j >= 72 {
            exponent = f32::cast_from((j - 72) / 2) / 28.0f32;
            position = ids[s * 3 + 2];
        } else if j >= 16 {
            exponent = f32::cast_from((j - 16) / 2) / 28.0f32;
            position = ids[s * 3 + 1];
        }
        let band = 1.0f32 / f32::powf(theta, exponent);
        let angle = position * band;
        cos[i] = angle.cos();
        sin[i] = angle.sin();
    }
}

/// L1/042: out[r, c] = x[r, c] + (v[c] + s).
#[ruda(launch)]
pub fn add_row_vector(x: &Array<f32>, v: &Array<f32>, out: &mut Array<f32>, count: u32, columns: u32, s: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize { out[i] = x[i] + (v[i % columns as usize] + s); }
}

/// GRN pass 1: sumsq[b, c] = sum over spatial of x[b, p, c]^2; grid (C / 256, B).
#[ruda(launch)]
pub fn grn_sumsq(x: &Array<f32>, out: &mut Array<f32>, spatial: u32, channels: u32) {
    let c = RUDA_POS_X as usize * 256 + UNIT_POS as usize;
    let b = RUDA_POS_Y as usize;
    if c < channels as usize {
        let mut total = 0.0f32;
        for p in 0..spatial as usize {
            let v = x[(b * spatial as usize + p) * channels as usize + c];
            total += v * v;
        }
        out[b * channels as usize + c] = total;
    }
}

/// GRN pass 2 (one block of 512 threads per batch, C = 512): nx = g / (mean_c g + eps), g = sqrt(sumsq).
#[ruda(launch)]
pub fn grn_scale(sumsq: &Array<f32>, out: &mut Array<f32>, eps: f32) {
    let b = RUDA_POS_X as usize;
    let c = UNIT_POS as usize;
    let mut scratch = SharedMemory::<f32>::new(16usize);
    let g = sumsq[b * 512 + c].sqrt();
    let total = crate::standalone::block_sum(g, &mut scratch, 512usize);
    out[b * 512 + c] = g / (total / 512.0f32 + eps);
}

/// GRN pass 3: y = w * (x * nx[b, c]) + bias + x.
#[ruda(launch)]
pub fn grn_apply(x: &Array<f32>, nx: &Array<f32>, weight: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>,
    count: u32, spatial: u32, channels: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let c = i % channels as usize;
        let b = i / (channels as usize * spatial as usize);
        let v = x[i];
        out[i] = weight[c] * (v * nx[b * channels as usize + c]) + bias[c] + v;
    }
}

/// L1/051: per-head (1 + w) RMSNorm backward for head_dim 256, one warp per row
/// (lane owns 8 contiguous columns), rows in (b, h, s) order. Writes
/// gx = F(rstd * (gn - mean(gn * n) * n)), gn = F(g * (1 + w)), to the (b, s, h)
/// layout and per-block column partials of sum(g * n).
#[ruda(launch)]
pub fn qknorm_bwd<F: Float>(grad: &Array<Vector<F, Const<8>>>, normed: &Array<Vector<F, Const<8>>>, rstd: &Array<f32>,
    weight: &Array<Vector<F, Const<8>>>, out: &mut Array<Vector<F, Const<8>>>, partial: &mut Array<f32>,
    rows: u32, heads: u32, seq: u32, #[define(F)] _f: StorageType,
) {
    let warp = UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    let w = Vector::<f32, Const<8>>::cast_from(weight[lane]);
    let mut acc = Vector::<f32, Const<8>>::new(0.0f32);
    let mut row = RUDA_POS_X as usize * 8 + warp;
    let stride = RUDA_COUNT_X as usize * 8;
    while row < rows as usize {
        let g = Vector::<f32, Const<8>>::cast_from(grad[row * 32 + lane]);
        let n = Vector::<f32, Const<8>>::cast_from(normed[row * 32 + lane]);
        let mut gn = Vector::<f32, Const<8>>::empty();
        let mut local = 0.0f32;
        #[unroll]
        for d in 0usize..8 {
            acc[d] = acc[d] + g[d] * n[d];
            gn[d] = rnd::<F>(g[d] * (1.0f32 + w[d]));
            local += gn[d] * n[d];
        }
        let mean = plane_sum(local) / 256.0f32;
        let r = rstd[row];
        let mut result = Vector::<F, Const<8>>::empty();
        #[unroll]
        for d in 0usize..8 { result[d] = F::cast_from(r * (gn[d] - mean * n[d])); }
        let s = row % seq as usize;
        let h = (row / seq as usize) % heads as usize;
        let b = row / (seq as usize * heads as usize);
        out[((b * seq as usize + s) * heads as usize + h) * 32 + lane] = result;
        row += stride;
    }
    let mut scratch = SharedMemory::<f32>::new(2048usize);
    #[unroll]
    for d in 0usize..8 { scratch[warp * 256 + lane * 8 + d] = acc[d]; }
    sync_ruda();
    let c = UNIT_POS as usize;
    let mut total = 0.0f32;
    #[unroll]
    for k in 0usize..8 { total += scratch[k * 256 + c]; }
    partial[RUDA_POS_X as usize * 256 + c] = total;
}

/// L1/052: out = F((h0 + p1 * (tm / m1) + p2 * (tm / m2)) / 3), tm = rms(h0),
/// m_i = sqrt(max(mean(p_i^2), eps)). 256 threads per row.
#[ruda(launch)]
pub fn altup_collapse<F: Float>(h0: &Array<F>, p1: &Array<f32>, p2: &Array<f32>, out: &mut Array<F>, width: u32, eps: f32,
    #[define(F)] _f: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = width as usize;
    let base = row * n;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut j = thread;
    while j < n {
        let a = f32::cast_from(h0[base + j]);
        let b = p1[base + j];
        let c = p2[base + j];
        s0 += a * a; s1 += b * b; s2 += c * c;
        j += 256;
    }
    let target = (crate::standalone::block_sum(s0, &mut scratch, 256usize) / n as f32).sqrt();
    let m1 = f32::max(crate::standalone::block_sum(s1, &mut scratch, 256usize) / n as f32, eps).sqrt();
    let m2 = f32::max(crate::standalone::block_sum(s2, &mut scratch, 256usize) / n as f32, eps).sqrt();
    let r1 = target / m1;
    let r2 = target / m2;
    let mut j2 = thread;
    while j2 < n {
        let v = f32::cast_from(h0[base + j2]) + p1[base + j2] * r1 + p2[base + j2] * r2;
        out[base + j2] = F::cast_from(v / 3.0f32);
        j2 += 256;
    }
}

/// L1/053: out = F(relu(x - (mean + std * multiplier))) per row (population std).
#[ruda(launch)]
pub fn gauss_sparse<F: Float>(x: &Array<F>, out: &mut Array<F>, width: u32, multiplier: f32, #[define(F)] _f: StorageType) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let n = width as usize;
    let base = row * n;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut s = 0.0f32;
    let mut j = thread;
    while j < n { s += f32::cast_from(x[base + j]); j += 256; }
    let mean = crate::standalone::block_sum(s, &mut scratch, 256usize) / n as f32;
    let mut q = 0.0f32;
    let mut j2 = thread;
    while j2 < n { let c = f32::cast_from(x[base + j2]) - mean; q += c * c; j2 += 256; }
    let std = (crate::standalone::block_sum(q, &mut scratch, 256usize) / n as f32).sqrt();
    let cutoff = mean + std * multiplier;
    let mut j3 = thread;
    while j3 < n {
        out[base + j3] = F::cast_from(f32::max(f32::cast_from(x[base + j3]) - cutoff, 0.0f32));
        j3 += 256;
    }
}

/// L1/057: out[b, s] = ids[b, s + 1] (0 for the last position).
#[ruda(launch)]
pub fn shift_ids(ids: &Array<i64>, out: &mut Array<i64>, count: u32, seq: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let s = i % seq as usize;
        let mut v = 0i64;
        if s + 1 < seq as usize { v = ids[i + 1]; }
        out[i] = v;
    }
}

/// L1/070: prefix[r, i] = sum_{k <= i} a[r, k] (f32) over rows of 128 (one thread per row).
#[ruda(launch)]
pub fn ssd_prefix<F: Float>(a: &Array<F>, prefix: &mut Array<f32>, rows: u32, #[define(F)] _f: StorageType) {
    let r = ABSOLUTE_POS as usize;
    if r < rows as usize {
        let mut s = 0.0f32;
        for i in 0..128usize {
            s += f32::cast_from(a[r * 128 + i]);
            prefix[r * 128 + i] = s;
        }
    }
}

/// L1/070: M[t, l, s] = F(G[tg, l, s] * exp(P[l] - P[s])) for s <= l (else 0), t = (bc * 32 + h),
/// tg = bc * 8 + h / 4; P from ssd_prefix over (b, h, c) rows of A (layout (B, H, NC, 128)).
#[ruda(launch)]
pub fn ssd_mask<F: Float>(g: &Array<f32>, prefix: &Array<f32>, out: &mut Array<F>, count: u32, chunks: u32,
    #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let s = i % 128;
        let l = (i / 128) % 128;
        let t = i / (128 * 128);
        let h = t % 32;
        let bc = t / 32;
        let c = bc % chunks as usize;
        let b = bc / chunks as usize;
        let prow = ((b * 32 + h) * chunks as usize + c) * 128;
        let mut v = 0.0f32;
        if s <= l {
            v = g[((bc * 8 + h / 4) * 128 + l) * 128 + s] * (prefix[prow + l] - prefix[prow + s]).exp();
        }
        out[i] = F::cast_from(v);
    }
}

/// L1/072: out[b, s, kv*D + d] = sum_j g[b, kv*G + j, s, d] (f32).
#[ruda(launch)]
pub fn group_sum_transpose(g: &Array<f32>, out: &mut Array<f32>, count: u32, kv_heads: u32, groups: u32, seq: u32, dim: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let d = i % dim as usize;
        let kv = (i / dim as usize) % kv_heads as usize;
        let s = (i / (dim as usize * kv_heads as usize)) % seq as usize;
        let b = i / (dim as usize * kv_heads as usize * seq as usize);
        let heads = kv_heads as usize * groups as usize;
        let mut total = 0.0f32;
        for j in 0..groups as usize {
            total += g[((b * heads + kv * groups as usize + j) * seq as usize + s) * dim as usize + d];
        }
        out[i] = total;
    }
}

/// L1/076: gated[e, t, i] = rw[t, e] * ((clamp(up, -L, L) + 1) * (g * sigmoid(g * alpha))),
/// g = min(gate, L); gate / up are the even / odd columns of gate_up (E, T, 2I).
#[ruda(launch)]
pub fn oss_glu(gate_up: &Array<f32>, rw: &Array<f32>, out: &mut Array<f32>, count: u32, tokens: u32, inter: u32,
    experts: u32, alpha: f32, limit: f32,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < count as usize {
        let i = idx % inter as usize;
        let t = (idx / inter as usize) % tokens as usize;
        let e = idx / (inter as usize * tokens as usize);
        let base = (e * tokens as usize + t) * 2 * inter as usize + 2 * i;
        let g = f32::min(gate_up[base], limit);
        let u = f32::max(f32::min(gate_up[base + 1], limit), -limit);
        let glu = g * (1.0f32 / (1.0f32 + (-(g * alpha)).exp()));
        out[idx] = rw[t * experts as usize + e] * ((u + 1.0f32) * glu);
    }
}

/// L1/079: grad_ssm = F(g * gate_act), grad_gate = F(g * ssm) [* bf16 silu'(gate)].
#[ruda(launch)]
pub fn gate_mul_bwd<F: Float>(g: &Array<F>, gate_act: &Array<F>, ssm: &Array<F>, gate: &Array<F>,
    grad_ssm: &mut Array<F>, grad_gate: &mut Array<F>, count: u32, use_silu: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let gg = f32::cast_from(g[i]);
        grad_ssm[i] = F::cast_from(gg * f32::cast_from(gate_act[i]));
        let mut gga = rnd::<F>(gg * f32::cast_from(ssm[i]));
        if use_silu != 0u32 {
            let x = f32::cast_from(gate[i]);
            let s = rnd::<F>(1.0f32 / (1.0f32 + (-x).exp()));
            let one_minus = rnd::<F>(1.0f32 - s);
            let inner = rnd::<F>(1.0f32 + rnd::<F>(x * one_minus));
            let grad = rnd::<F>(s * inner);
            gga = gga * grad;
        }
        grad_gate[i] = F::cast_from(gga);
    }
}

/// L1/087: RMSNorm backward rows (hidden 4096, 256 threads x 16 columns) scattered
/// into the f32 embedding gradient table with atomics, plus per-block column
/// partials of sum(g * h * rstd). Grid-stride over rows.
#[ruda(launch)]
pub fn embed_rms_bwd<F: Float>(grad: &Array<F>, hidden: &Array<f32>, rstd: &Array<f32>, weight: &Array<F>, ids: &Array<i64>,
    table: &mut Array<Atomic<f32>>, partial: &mut Array<f32>, rows: u32, #[define(F)] _f: StorageType,
) {
    let thread = UNIT_POS as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut acc = Array::<f32>::new(16usize);
    let mut w = Array::<f32>::new(16usize);
    #[unroll]
    for j in 0usize..16 { acc[j] = 0.0f32; w[j] = f32::cast_from(weight[thread * 16 + j]); }
    let mut row = RUDA_POS_X as usize;
    while row < rows as usize {
        let r = rstd[row];
        let mut gw = Array::<f32>::new(16usize);
        let mut hn = Array::<f32>::new(16usize);
        let mut local = 0.0f32;
        #[unroll]
        for j in 0usize..16 {
            let c = row * 4096 + thread * 16 + j;
            let g = f32::cast_from(grad[c]);
            hn[j] = hidden[c] * r;
            acc[j] += g * hn[j];
            gw[j] = g * w[j];
            local += gw[j] * hn[j];
        }
        let mean = crate::standalone::block_sum(local, &mut scratch, 256usize) / 4096.0f32;
        let target = ids[row] as usize * 4096 + thread * 16;
        #[unroll]
        for j in 0usize..16 { table[target + j].fetch_add(r * (gw[j] - mean * hn[j])); }
        row += RUDA_COUNT_X as usize;
    }
    #[unroll]
    for j in 0usize..16 { partial[RUDA_POS_X as usize * 4096 + thread * 16 + j] = acc[j]; }
}

/// L1/091: per (b, s) row block (128 threads = head dim), for 32 q + 8 k heads:
/// RMSNorm backward, RoPE backward (bf16 arithmetic), grad cos / sin sums over
/// heads, and the concatenated (B, S, 6144) grad_qkv (v copied). Grid-stride over
/// rows; per-block column partials of sum(g * x * rstd) for q and k weights.
#[ruda(launch)]
pub fn qkrope_bwd<F: Float>(gq: &Array<F>, gk: &Array<F>, gv: &Array<F>, xq: &Array<F>, xk: &Array<F>,
    rq: &Array<F>, rk: &Array<F>, cos: &Array<F>, sin: &Array<F>, wq: &Array<F>, wk: &Array<F>,
    gqkv: &mut Array<F>, grad_cos: &mut Array<F>, grad_sin: &mut Array<F>, partial: &mut Array<f32>,
    rows: u32, seq: u32, #[define(F)] _f: StorageType,
) {
    let d = UNIT_POS as usize;
    let mut scratch = SharedMemory::<f32>::new(4usize);
    let mut sg = SharedMemory::<f32>::new(128usize);
    let mut sx = SharedMemory::<f32>::new(128usize);
    let mut so = SharedMemory::<f32>::new(128usize);
    let mut pq = 0.0f32;
    let mut pk = 0.0f32;
    let mut row = RUDA_POS_X as usize;
    let partner = (d + 64) % 128;
    while row < rows as usize {
        let b = row / seq as usize;
        let s = row % seq as usize;
        let c = f32::cast_from(cos[row * 128 + d]);
        let sn = f32::cast_from(sin[row * 128 + d]);
        let mut cos_q = 0.0f32;
        let mut sin_q = 0.0f32;
        let mut cos_k = 0.0f32;
        let mut sin_k = 0.0f32;
        for h in 0..40usize {
            let mut g = 0.0f32;
            let mut x = 0.0f32;
            let mut r = 0.0f32;
            let mut w = 0.0f32;
            if h < 32 {
                let idx = ((b * 32 + h) * seq as usize + s) * 128 + d;
                g = f32::cast_from(gq[idx]); x = f32::cast_from(xq[idx]);
                r = f32::cast_from(rq[(b * 32 + h) * seq as usize + s]); w = f32::cast_from(wq[d]);
                pq += g * (x * r);
            } else {
                let idx = ((b * 8 + h - 32) * seq as usize + s) * 128 + d;
                g = f32::cast_from(gk[idx]); x = f32::cast_from(xk[idx]);
                r = f32::cast_from(rk[(b * 8 + h - 32) * seq as usize + s]); w = f32::cast_from(wk[d]);
                pk += g * (x * r);
            }
            let total = crate::standalone::block_sum(g * w * x, &mut scratch, 128usize);
            let gx = rnd::<F>(g * w * r + total * ((-(r * r * r)) * x / 128.0f32));
            sg[d] = gx;
            sx[d] = x;
            sync_ruda();
            let mut inv_g = -sg[partner];
            let mut inv_x = sx[partner];
            if d < 64 { inv_g = sg[partner]; inv_x = -sx[partner]; }
            let grad_rope = rnd::<F>(rnd::<F>(gx * c) + rnd::<F>(inv_g * sn));
            let original = rnd::<F>(rnd::<F>(x * c) + rnd::<F>(inv_x * sn));
            so[d] = original;
            sync_ruda();
            let mut rotated = so[partner];
            if d < 64 { rotated = -so[partner]; }
            let pc = rnd::<F>(gx * original);
            let ps = rnd::<F>(gx * rotated);
            if h < 32 {
                cos_q += pc; sin_q += ps;
                gqkv[row * 6144 + h * 128 + d] = F::cast_from(grad_rope);
            } else {
                cos_k += pc; sin_k += ps;
                gqkv[row * 6144 + 4096 + (h - 32) * 128 + d] = F::cast_from(grad_rope);
            }
            sync_ruda();
        }
        for h in 0..8usize {
            gqkv[row * 6144 + 5120 + h * 128 + d] = gv[((b * 8 + h) * seq as usize + s) * 128 + d];
        }
        grad_cos[row * 128 + d] = F::cast_from(rnd::<F>(cos_q) + rnd::<F>(cos_k));
        grad_sin[row * 128 + d] = F::cast_from(rnd::<F>(sin_q) + rnd::<F>(sin_k));
        row += RUDA_COUNT_X as usize;
    }
    partial[RUDA_POS_X as usize * 256 + d] = pq;
    partial[RUDA_POS_X as usize * 256 + 128 + d] = pk;
}

/// L1/093: grouped top-k routing backward rows (E <= 192 experts, top-8):
/// grl[n, e] = F(F(gs[e] * s) * F(1 - s)), gs[e] = F(F(gtn - sum) / denom) where routed.
#[ruda(launch)]
pub fn router_bwd<F: Float>(grad: &Array<F>, normalized: &Array<F>, index: &Array<i64>, denominator: &Array<F>,
    scores: &Array<F>, out: &mut Array<F>, experts: u32, factor: f32, #[define(F)] _f: StorageType,
) {
    let n = RUDA_POS_X as usize;
    let e = UNIT_POS as usize;
    if e < experts as usize {
        let mut sum = 0.0f32;
        #[unroll]
        for k in 0usize..8 {
            let gtn = rnd::<F>(f32::cast_from(grad[n * 8 + k]) * factor);
            sum += rnd::<F>(gtn * f32::cast_from(normalized[n * 8 + k]));
        }
        let gs = rnd::<F>(sum);
        let den = f32::cast_from(denominator[n]);
        let mut value = 0.0f32;
        #[unroll]
        for k in 0usize..8 {
            if index[n * 8 + k] as usize == e {
                let gtn = rnd::<F>(f32::cast_from(grad[n * 8 + k]) * factor);
                value += rnd::<F>(rnd::<F>(gtn - gs) / den);
            }
        }
        value = rnd::<F>(value);
        let s = f32::cast_from(scores[n * experts as usize + e]);
        out[n * experts as usize + e] = F::cast_from(rnd::<F>(value * s) * rnd::<F>(1.0f32 - s));
    }
}
