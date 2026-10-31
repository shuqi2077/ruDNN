//! L2 batch A kernels (031, 035).
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

/// L2/031: emb[b, j] = cos(t*1000 * f[j]) for j < half, sin(t*1000 * f[j - half]) otherwise.
#[ruda(launch)]
pub fn timestep_embed(t: &Array<f32>, freqs: &Array<f32>, out: &mut Array<f32>, count: u32, half: u32, scale: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let h = half as usize;
        let j = i % (2 * h);
        let b = i / (2 * h);
        let mut scaled = t[b];
        if scale != 1.0f32 { scaled = t[b] * scale; }
        if j < h { out[i] = (scaled * freqs[j]).cos(); } else { out[i] = (scaled * freqs[j - h]).sin(); }
    }
}

/// L2/035: depthwise 7x7 conv (pad 3) of NCHW input with bias, written NHWC.
#[ruda(launch)]
pub fn dwconv7_nhwc(x: &Array<f32>, weight: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>, count: u32,
    channels: u32, height: u32, width: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let w = width as usize;
        let h = height as usize;
        let c_count = channels as usize;
        let xx = i % w;
        let y = (i / w) % h;
        let c = (i / (w * h)) % c_count;
        let b = i / (w * h * c_count);
        let base = (b * c_count + c) * h * w;
        let mut acc = bias[c];
        for dy in 0..7usize {
            let yy = y as i32 + dy as i32 - 3;
            if yy >= 0 && yy < h as i32 {
                #[unroll]
                for dx in 0usize..7 {
                    let xs = xx as i32 + dx as i32 - 3;
                    if xs >= 0 && xs < w as i32 {
                        acc += weight[c * 49 + dy * 7 + dx] * x[base + yy as usize * w + xs as usize];
                    }
                }
            }
        }
        out[((b * h + y) * w + xx) * c_count + c] = acc;
    }
}

/// GRN scale for any channel count (one block of 256 threads per batch):
/// nx = g / (mean_c g + eps), g = sqrt(sumsq).
#[ruda(launch)]
pub fn grn_scale_n(sumsq: &Array<f32>, out: &mut Array<f32>, channels: u32, eps: f32) {
    let b = RUDA_POS_X as usize;
    let t = UNIT_POS as usize;
    let n = channels as usize;
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut local = 0.0f32;
    let mut c = t;
    while c < n { local += sumsq[b * n + c].sqrt(); c += 256; }
    let mean = crate::standalone::block_sum(local, &mut scratch, 256usize) / n as f32;
    let mut c2 = t;
    while c2 < n { out[b * n + c2] = sumsq[b * n + c2].sqrt() / (mean + eps); c2 += 256; }
}

/// L2/068: out = g * (0.5 * (1 + t) + 0.5 * x * (1 - t^2) * 0.79788 * (1 + 0.134145 * x^2)).
#[ruda(launch)]
pub fn gelu_bwd_given(g: &Array<f32>, x: &Array<f32>, t: &Array<f32>, out: &mut Array<f32>, count: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let xv = x[i];
        let tv = t[i];
        let part1 = 0.5f32 * (1.0f32 + tv);
        let sech = 1.0f32 - tv * tv;
        let d = 0.7978845608028654f32 * (1.0f32 + 0.134145f32 * xv * xv);
        out[i] = g[i] * (part1 + 0.5f32 * xv * sech * d);
    }
}

#[ruda]
fn r16<F: Float>(v: f32) -> f32 { f32::cast_from(F::cast_from(v)) }

/// L2/005: from ggo = F(dO @ Wd): gated = F(act * up), gup = F(ggo * act),
/// ggate = F(F(ggo * up) * silu'(gate)) (silu' in f32).
#[ruda(launch)]
pub fn swiglu_bwd_full<F: Float>(ggo: &Array<F>, up: &Array<F>, act: &Array<F>, gate: &Array<F>,
    gated: &mut Array<F>, grad_up: &mut Array<F>, grad_gate: &mut Array<F>, count: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let g = f32::cast_from(ggo[i]);
        let u = f32::cast_from(up[i]);
        let a = f32::cast_from(act[i]);
        let x = f32::cast_from(gate[i]);
        gated[i] = F::cast_from(a * u);
        grad_up[i] = F::cast_from(g * a);
        let s = 1.0f32 / (1.0f32 + (-x).exp());
        grad_gate[i] = F::cast_from(r16::<F>(g * u) * (s * (1.0f32 + x * (1.0f32 - s))));
    }
}

/// L2/049: DeepSeek-V3 group-limited top-k routing, one warp per token (256
/// experts = 8 groups x 32 lanes). Writes the 8 selected experts in descending
/// masked-score order and weights = s / (sum s + 1e-20) * factor.
#[ruda(launch)]
pub fn group_topk_route(logits: &Array<f32>, bias: &Array<f32>, index: &mut Array<i64>, weights: &mut Array<f32>,
    tokens: u32, factor: f32,
) {
    let token = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    if token < tokens as usize {
        let mut score = Array::<f32>::new(8usize);
        let mut routed = Array::<f32>::new(8usize);
        let mut group_score = Array::<f32>::new(8usize);
        #[unroll]
        for g in 0usize..8 {
            let e = g * 32 + lane;
            let s = 1.0f32 / (1.0f32 + (-logits[token * 256 + e]).exp());
            score[g] = s;
            routed[g] = s + bias[e];
            let top1 = plane_max(routed[g]);
            let first = plane_min(select(routed[g] == top1, lane as u32, 64u32));
            let mut rest = routed[g];
            if lane as u32 == first { rest = f32::new(f32::NEG_INFINITY); }
            let top2 = plane_max(rest);
            group_score[g] = top1 + top2;
        }
        // top-4 groups (ties -> lower group index)
        let mut keep = Array::<u32>::new(8usize);
        #[unroll]
        for g in 0usize..8 { keep[g] = 0u32; }
        #[unroll]
        for _ in 0usize..4 {
            let mut best = 0usize;
            let mut best_value = f32::new(f32::NEG_INFINITY);
            #[unroll]
            for g in 0usize..8 {
                if keep[g] == 0u32 && (group_score[g] > best_value || best_value == f32::new(f32::NEG_INFINITY)) {
                    if group_score[g] > best_value { best_value = group_score[g]; best = g; }
                }
            }
            keep[best] = 1u32;
        }
        let mut masked = Array::<f32>::new(8usize);
        #[unroll]
        for g in 0usize..8 {
            masked[g] = routed[g];
            if keep[g] == 0u32 { masked[g] = f32::new(-3.4028234663852886e38f32); }
        }
        let mut chosen = Array::<f32>::new(8usize);
        let mut chosen_index = Array::<u32>::new(8usize);
        let mut total = 0.0f32;
        #[unroll]
        for k in 0usize..8 {
            // lane-local best, then warp argmax (lowest expert index on ties)
            let mut v = f32::new(f32::NEG_INFINITY);
            let mut idx = 1024u32;
            #[unroll]
            for g in 0usize..8 {
                if masked[g] > v { v = masked[g]; idx = (g * 32 + lane) as u32; }
            }
            let best = plane_max(v);
            let winner = plane_min(select(v == best, idx, 1024u32));
            #[unroll]
            for g in 0usize..8 {
                if (g * 32 + lane) as u32 == winner { masked[g] = f32::new(f32::NEG_INFINITY); }
            }
            let g_w = (winner / 32u32) as usize;
            let mut s = 0.0f32;
            #[unroll]
            for g in 0usize..8 { if g == g_w { s = score[g]; } }
            let picked = plane_max(select(lane as u32 == winner % 32u32, s, f32::new(f32::NEG_INFINITY)));
            chosen[k] = picked;
            chosen_index[k] = winner;
            total += picked;
        }
        if lane == 0 {
            #[unroll]
            for k in 0usize..8 {
                index[token * 8 + k] = chosen_index[k] as i64;
                weights[token * 8 + k] = chosen[k] / (total + 1e-20f32) * factor;
            }
        }
    }
}

/// L2/013: s[t] = f32(F(sigmoid(f32(logits[t * ld])))).
#[ruda(launch)]
pub fn sigmoid_column<F: Float>(logits: &Array<F>, out: &mut Array<f32>, count: u32, ld: u32, #[define(F)] _f: StorageType) {
    let t = ABSOLUTE_POS as usize;
    if t < count as usize {
        let x = f32::cast_from(logits[t * ld as usize]);
        out[t] = r16::<F>(1.0f32 / (1.0f32 + (-x).exp()));
    }
}

/// Softmax top-k routing (E <= 256 experts as 8 per lane, k <= 8), one warp per
/// token: probabilities in f32 from F logits, top-k descending (lowest index on
/// ties), optional renormalization; weights stored as W, ids as i64.
#[ruda(launch)]
pub fn softmax_topk<F: Float, W: Float>(logits: &Array<F>, bias: &Array<f32>, index: &mut Array<i64>, weights: &mut Array<W>,
    tokens: u32, experts: u32, top_k: u32, norm_min: f32, #[define(F)] _f: StorageType, #[define(W)] _w: StorageType,
    #[comptime] renorm: bool, #[comptime] per_lane: usize, #[comptime] biased: bool,
) {
    let token = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    if token < tokens as usize {
        let e_count = experts as usize;
        let mut p = Array::<f32>::new(per_lane);
        let mut m = f32::new(f32::NEG_INFINITY);
        #[unroll]
        for j in 0usize..per_lane {
            let e = j * 32 + lane;
            let mut v = f32::new(f32::NEG_INFINITY);
            if e < e_count { v = f32::cast_from(logits[token * e_count + e]); }
            p[j] = v;
            m = f32::max(m, v);
        }
        let mx = plane_max(m);
        let mut s = 0.0f32;
        #[unroll]
        for j in 0usize..per_lane {
            let e = j * 32 + lane;
            let mut v = 0.0f32;
            if e < e_count { v = (p[j] - mx).exp(); }
            p[j] = v;
            s += v;
        }
        let total = plane_sum(s);
        #[unroll]
        for j in 0usize..per_lane {
            p[j] = p[j] / total;
            if comptime!(biased) {
                let e = j * 32 + lane;
                if e < e_count { p[j] = p[j] + bias[e]; }
            }
        }
        let mut chosen = Array::<f32>::new(16usize);
        let mut chosen_index = Array::<u32>::new(16usize);
        let mut picked_sum = 0.0f32;
        for k in 0..top_k as usize {
            let mut v = f32::new(f32::NEG_INFINITY);
            let mut idx = 4096u32;
            #[unroll]
            for j in 0usize..per_lane {
                if (j * 32 + lane) < e_count && p[j] > v { v = p[j]; idx = (j * 32 + lane) as u32; }
            }
            let best = plane_max(v);
            let winner = plane_min(select(v == best, idx, 4096u32));
            #[unroll]
            for j in 0usize..per_lane { if (j * 32 + lane) as u32 == winner { p[j] = f32::new(f32::NEG_INFINITY); } }
            chosen[k] = best;
            chosen_index[k] = winner;
            picked_sum += best;
        }
        if comptime!(biased) { picked_sum = f32::max(picked_sum, norm_min); }
        if lane == 0 {
            for k in 0..top_k as usize {
                let mut w = chosen[k];
                if comptime!(renorm) { w = w / picked_sum; }
                index[token * top_k as usize + k] = chosen_index[k] as i64;
                weights[token * top_k as usize + k] = W::cast_from(w);
            }
        }
    }
}

/// Sigmoid + bias top-k routing (E <= 256, k <= 8), one warp per token: choice
/// scores s = sigmoid(logit) + bias, top-k descending; weights = s_k / (sum + 1e-20)
/// * factor taken from the biased scores (`biased`) or the raw sigmoid.
#[ruda(launch)]
pub fn sigmoid_topk<W: Float>(logits: &Array<f32>, bias: &Array<f32>, index: &mut Array<i64>, weights: &mut Array<W>,
    tokens: u32, experts: u32, top_k: u32, factor: f32, #[define(W)] _w: StorageType, #[comptime] biased: bool,
) {
    let token = RUDA_POS_X as usize * 8 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    if token < tokens as usize {
        let e_count = experts as usize;
        let mut choice = Array::<f32>::new(8usize);
        let mut raw = Array::<f32>::new(8usize);
        #[unroll]
        for j in 0usize..8 {
            let e = j * 32 + lane;
            let mut c = f32::new(f32::NEG_INFINITY);
            let mut r = 0.0f32;
            if e < e_count {
                r = 1.0f32 / (1.0f32 + (-logits[token * e_count + e]).exp());
                c = r + bias[e];
            }
            choice[j] = c;
            raw[j] = r;
        }
        let mut picked = Array::<f32>::new(8usize);
        let mut picked_index = Array::<u32>::new(8usize);
        let mut total = 0.0f32;
        for k in 0..top_k as usize {
            let mut v = f32::new(f32::NEG_INFINITY);
            let mut idx = 1024u32;
            let mut rv = 0.0f32;
            #[unroll]
            for j in 0usize..8 {
                if choice[j] > v { v = choice[j]; idx = (j * 32 + lane) as u32; rv = raw[j]; }
            }
            let best = plane_max(v);
            let winner = plane_min(select(v == best, idx, 1024u32));
            let mut mine = f32::new(f32::NEG_INFINITY);
            if idx == winner { mine = rv; if comptime!(biased) { mine = v; } }
            let w = plane_max(mine);
            #[unroll]
            for j in 0usize..8 { if (j * 32 + lane) as u32 == winner { choice[j] = f32::new(f32::NEG_INFINITY); } }
            picked[k] = w;
            picked_index[k] = winner;
            total += w;
        }
        if lane == 0 {
            for k in 0..top_k as usize {
                index[token * top_k as usize + k] = picked_index[k] as i64;
                weights[token * top_k as usize + k] = W::cast_from(picked[k] / (total + 1e-20f32) * factor);
            }
        }
    }
}

/// L2/019: multimodal RoPE tables: out[b, s, d] = tables[sec(d), b, s, d] with
/// sections [0, 32) -> 0, [32, 80) -> 1, [80, 128) -> 2; tables (3, B, KV, 128).
#[ruda(launch)]
pub fn mrope_combine<F: Float>(tables: &Array<F>, out: &mut Array<F>, count: u32, seq: u32, kv_seq: u32, batch: u32,
    #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let d = i % 128;
        let s = (i / 128) % seq as usize;
        let b = i / (128 * seq as usize);
        let mut sec = 2usize;
        if d < 32usize { sec = 0usize; } else if d < 80usize { sec = 1usize; }
        out[i] = tables[((sec * batch as usize + b) * kv_seq as usize + s) * 128 + d];
    }
}

/// cos / sin tables (rows, dim) = F(cos / sin(pos[row] * inv[d % (dim / 2)]) * factor).
#[ruda(launch)]
pub fn rope_table_factor<F: Float>(positions: &Array<i64>, inv_freq: &Array<f32>, cos: &mut Array<F>, sin: &mut Array<F>,
    count: u32, dim: u32, factor: f32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < count as usize {
        let d = i % dim as usize;
        let row = i / dim as usize;
        let angle = f32::cast_from(positions[row]) * inv_freq[d % (dim as usize / 2)];
        cos[i] = F::cast_from(angle.cos() * factor);
        sin[i] = F::cast_from(angle.sin() * factor);
    }
}
