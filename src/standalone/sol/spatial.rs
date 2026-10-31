use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};
use super::super::block_sum;

#[ruda(launch)]
pub fn depthwise2d_input_grad(grad: &Array<f32>, weight: &Array<f32>, out: &mut Array<f32>, channels: u32, height: u32, width: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let x = (i % width as usize) as i32;
        let y = (i / width as usize % height as usize) as i32;
        let bc = i / (height * width) as usize;
        let c = bc % channels as usize;
        let mut sum = 0.0f32;
        for ky in 0usize..7 {
            for kx in 0usize..7 {
                let oy = y + 3 - ky as i32;
                let ox = x + 3 - kx as i32;
                if oy >= 0 && oy < height as i32 && ox >= 0 && ox < width as i32 {
                    sum += grad[(bc * height as usize + oy as usize) * width as usize + ox as usize] * weight[c * 49 + ky * 7 + kx];
                }
            }
        }
        out[i] = sum;
    }
}

#[ruda(launch)]
pub fn depthwise2d_weight_grad(x: &Array<f32>, grad: &Array<f32>, out: &mut Array<f32>, channels: u32, height: u32, width: u32) {
    let item = RUDA_POS_X as usize;
    let c = item / 49;
    let ky = (item / 7 % 7) as i32;
    let kx = (item % 7) as i32;
    let spatial = (height * width) as usize;
    let batches = x.len() / (channels as usize * spatial);
    let mut scratch = SharedMemory::<f32>::new(8usize);
    let mut sum = 0.0f32;
    let mut p = UNIT_POS as usize;
    while p < batches * spatial {
        let batch = p / spatial;
        let y = (p % spatial / width as usize) as i32 + ky - 3;
        let xx = (p % width as usize) as i32 + kx - 3;
        if y >= 0 && y < height as i32 && xx >= 0 && xx < width as i32 {
            sum += x[((batch * channels as usize + c) * height as usize + y as usize) * width as usize + xx as usize]
                * grad[(batch * channels as usize + c) * spatial + p % spatial];
        }
        p += 256;
    }
    let total = block_sum(sum, &mut scratch, 256usize);
    if UNIT_POS == 0u32 { out[item] = total; }
}

#[ruda(launch)]
pub fn pool2(x: &Array<f32>, out: &mut Array<f32>, height: u32, width: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let oh = height as usize / 2;
        let ow = width as usize / 2;
        let bc = i / (oh * ow);
        let y = i / ow % oh * 2;
        let xx = i % ow * 2;
        let base = (bc * height as usize + y) * width as usize + xx;
        out[i] = f32::max(f32::max(x[base], x[base + 1]), f32::max(x[base + width as usize], x[base + width as usize + 1]));
    }
}

#[ruda(launch)]
pub fn resize2(x: &Array<f32>, out: &mut Array<f32>, height: u32, width: u32, oh: u32, ow: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let bc = i / (oh * ow) as usize;
        let y = i / ow as usize % oh as usize;
        let xx = i % ow as usize;
        let sy = f32::max(0.0f32, (y as f32 + 0.5f32) * height as f32 / oh as f32 - 0.5f32);
        let sx = f32::max(0.0f32, (xx as f32 + 0.5f32) * width as f32 / ow as f32 - 0.5f32);
        let y0 = sy as usize; let x0 = sx as usize;
        let y1 = usize::min(y0 + 1, height as usize - 1); let x1 = usize::min(x0 + 1, width as usize - 1);
        let dy = sy - y0 as f32; let dx = sx - x0 as f32;
        let a = x[(bc * height as usize + y0) * width as usize + x0] * (1.0f32 - dx) + x[(bc * height as usize + y0) * width as usize + x1] * dx;
        let b = x[(bc * height as usize + y1) * width as usize + x0] * (1.0f32 - dx) + x[(bc * height as usize + y1) * width as usize + x1] * dx;
        out[i] = a * (1.0f32 - dy) + b * dy;
    }
}

#[ruda(launch)]
pub fn columns<F: Float>(x: &Array<F>, out: &mut Array<F>,
    channels: u32, depth: u32, height: u32, width: u32, od: u32, oh: u32, ow: u32,
    kd: u32, kh: u32, kw: u32, sd: u32, sh: u32, sw: u32, pd: u32, ph: u32, pw: u32,
    kp: u32, #[comptime] transpose: bool, #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let column = i % kp as usize;
        let position = i / kp as usize;
        let c = column / (kd * kh * kw) as usize;
        let z = (column / (kh * kw) as usize % kd as usize) as i32;
        let y = (column / kw as usize % kh as usize) as i32;
        let xx = (column % kw as usize) as i32;
        let oz = (position / (oh * ow) as usize) as i32;
        let oy = (position / ow as usize % oh as usize) as i32;
        let ox = (position % ow as usize) as i32;
        let mut iz = oz * sd as i32 + z - pd as i32;
        let mut iy = oy * sh as i32 + y - ph as i32;
        let mut ix = ox * sw as i32 + xx - pw as i32;
        let mut valid = c < channels as usize && position < (od * oh * ow) as usize;
        if transpose {
            iz = oz + pd as i32 - z;
            iy = oy + ph as i32 - y;
            ix = ox + pw as i32 - xx;
            valid = valid && iz % sd as i32 == 0 && iy % sh as i32 == 0 && ix % sw as i32 == 0;
            iz /= sd as i32; iy /= sh as i32; ix /= sw as i32;
        }
        let mut value = F::cast_from(0.0f32);
        if valid && iz >= 0 && iz < depth as i32 && iy >= 0 && iy < height as i32 && ix >= 0 && ix < width as i32 {
            value = x[((c * depth as usize + iz as usize) * height as usize + iy as usize) * width as usize + ix as usize];
        }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn matrix_pad<F: Float>(x: &Array<F>, out: &mut Array<F>, rows: u32, cols: u32, pitch: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let r = i / pitch as usize;
        let c = i % pitch as usize;
        let mut value = F::cast_from(0.0f32);
        if r < rows as usize && c < cols as usize { value = x[r * cols as usize + c]; }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn depthwise1d<F: Float>(x: &Array<F>, weight: &Array<F>, bias: &Array<F>, out: &mut Array<F>,
    seq: u32, out_seq: u32, channels: u32, kernel: u32, padding: u32, #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let c = i / out_seq as usize % channels as usize;
        let batch = i / (out_seq * channels) as usize;
        let t = (i % out_seq as usize) as i32;
        let mut value = 0.0f32;
        for j in 0u32..kernel {
            let source = t + j as i32 - padding as i32;
            if source >= 0 && source < seq as i32 {
                value += f32::cast_from(x[(batch * channels as usize + c) * seq as usize + source as usize]) * f32::cast_from(weight[c * kernel as usize + j as usize]);
            }
        }
        out[i] = F::cast_from(value + f32::cast_from(bias[c]));
    }
}
