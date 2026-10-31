use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

#[ruda(launch)]
pub fn timing(inv: &Array<f32>, out: &mut Array<f32>) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let width = inv.len() * 2;
        let position = 127.0f32 - (i / width) as f32;
        let d = i % width;
        let angle = position * inv[d % inv.len()];
        let mut value = angle.sin();
        if d >= inv.len() { value = angle.cos(); }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn local_logits(ac: &Array<f32>, bd: &Array<f32>, mask: &Array<u8>, out: &mut Array<f32>,
    length: u32, blocks: u32, cap: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let c = i % 160;
        let r = i / 160 % 32;
        let block = i / (160 * 32) % blocks as usize;
        let bh = i / (160 * 32 * blocks as usize);
        let oldrow = (r * 160 + c) / 161;
        let oldcol = (r * 160 + c) % 161;
        let mut position_bias = 0.0f32;
        if oldcol < 128 { position_bias = bd[((bh * blocks as usize + block) * 32 + oldrow) * 128 + oldcol]; }
        let key = (block * 32 + c) as i32 - 127;
        let mut value = -3.4028234663852886e38f32;
        if key >= 0 && key < length as i32 && c >= r && c <= r + 127 {
            if mask[(bh / 8) * length as usize + key as usize] == 0u8 {
                value = ((ac[i] + position_bias) / cap).tanh() * cap;
            }
        }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn conformer_position(dot: &Array<f32>, out: &mut Array<half::bf16>, blocks: u32, table: u32, remainder: u32, scale: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let k = i % 512;
        let q = i / 512 % 512;
        let bh = i / (512 * 512);
        let block = bh / 8 % blocks as usize;
        let position = q + 512 - k;
        let mut value = dot[(bh * 512 + q) * table as usize + position] * scale;
        if remainder != 0 && block + 1 == blocks as usize && (q >= remainder as usize || k >= remainder as usize) {
            value = -3.4028234663852886e38f32;
        }
        out[i] = half::bf16::cast_from(value);
    }
}
