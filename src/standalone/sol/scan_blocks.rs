use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

#[ruda(launch)]
pub fn segment_exp(x: &Array<f32>, prefix: &mut Array<f32>, out: &mut Array<f32>, width: u32) {
    let task = ABSOLUTE_POS as usize;
    let n = width as usize;
    if task < x.len() {
        let row = task / n;
        let column = task % n;
        let mut sum = 0.0f32;
        for i in 0usize..n {
            let mut value = 0.0f32;
            if i >= column {
                if i > column { sum += x[row * n + i]; }
                value = sum.exp();
            }
            out[(row * n + i) * n + column] = value;
        }
        let mut cumulative = 0.0f32;
        for i in 0usize..column + 1 { cumulative += x[row * n + i]; }
        prefix[task] = cumulative;
    }
}

#[ruda(launch)]
pub fn delta_decay(x: &Array<f32>, prefix: &Array<f32>, out: &mut Array<f32>, #[comptime] strict: bool) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let r = i / 64 % 64;
        let c = i % 64;
        let block = i / 4096;
        let mut value = 0.0f32;
        if r >= c {
            value = x[i] * (prefix[block * 64 + r] - prefix[block * 64 + c]).exp();
        }
        if strict { value = -value; if r == c { value = 0.0f32; } }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn delta_inverse(x: &Array<f32>, out: &mut Array<f32>) {
    let block = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let mut shared = SharedMemory::<f32>::new(4096usize);
    let mut row = SharedMemory::<f32>::new(64usize);
    for j in 0usize..16 { shared[thread + j * 256] = x[block * 4096 + thread + j * 256]; }
    sync_ruda();
    for i in 1usize..64 {
        if thread < 64 { row[thread] = shared[i * 64 + thread]; }
        sync_ruda();
        if thread < i {
            let mut sum = 0.0f32;
            for k in 0usize..i { sum += row[k] * shared[k * 64 + thread]; }
            shared[i * 64 + thread] = row[thread] + sum;
        }
        sync_ruda();
    }
    for j in 0usize..16 {
        let p = thread + j * 256;
        let mut value = shared[p];
        if p / 64 == p % 64 { value += 1.0f32; }
        out[block * 4096 + p] = value;
    }
}
