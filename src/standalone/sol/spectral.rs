use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

#[ruda(launch)]
pub fn hyena_basis(out: &mut Array<f32>, seq: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let t = i / 5;
        let d = i % 5;
        let mut v = t as f32 / (seq - 1) as f32;
        if d != 0 {
            let mut freq = 1.0f32;
            if d == 1 || d == 3 { freq = 0.0001f32; }
            let angle = -freq * (2.0f32 * std::f32::consts::PI * t as f32 / seq as f32);
            if d < 3 { v = angle.cos(); } else { v = angle.sin(); }
        }
        out[i] = v;
    }
}

#[ruda(launch)]
pub fn hyena_filter(x: &Array<f32>, deltas: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>, seq: u32, shift: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let channels = bias.len();
        let t = (i / channels) as f32 / (seq - 1) as f32;
        let d = i % channels;
        out[i] = x[i] * ((-t * deltas[d].abs()).exp() + shift) + bias[d];
    }
}

#[ruda(launch)]
pub fn convolution_product(a: &Array<Vector<f32, Const<2>>>, b: &Array<Vector<f32, Const<2>>>,
    out: &mut Array<Vector<f32, Const<2>>>, channels: u32, length: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let av = a[i]; let bv = b[i % (channels * length) as usize];
        let mut v = Vector::<f32, Const<2>>::empty();
        v[0] = av[0] * bv[0] - av[1] * bv[1];
        v[1] = av[0] * bv[1] + av[1] * bv[0];
        out[i] = v;
    }
}

#[ruda]
fn reverse(i: usize, #[comptime] size: usize) -> usize {
    let mut x = i;
    let mut out = 0usize;
    #[unroll]
    for _ in 0usize..comptime!(size.trailing_zeros() as usize) {
        out = (out << 1usize) | (x & 1usize);
        x >>= 1usize;
    }
    out
}

#[ruda(launch)]
pub fn stage(input: &Array<Vector<f32, Const<2>>>, out: &mut Array<Vector<f32, Const<2>>>, n1: u32, n2: u32,
    sign: f32, scale: f32, #[comptime] size: usize, #[comptime] first: bool,
) {
    let row = RUDA_POS_Y as usize;
    let tile = RUDA_POS_X as usize;
    let t = UNIT_POS as usize;
    let n = (n1 * n2) as usize;
    let mut re = SharedMemory::<f32>::new(size);
    let mut im = SharedMemory::<f32>::new(size);
    let mut i = t;
    while i < size {
        let mut p = tile * n2 as usize + i;
        if comptime!(first) { p = i * n2 as usize + tile; }
        let value = input[row * n + p];
        let r = reverse(i, size);
        re[r] = value[0]; im[r] = value[1];
        i += 256;
    }
    sync_ruda();
    #[unroll]
    for s in 0usize..comptime!(size.trailing_zeros() as usize) {
        let half = comptime!(1usize << s);
        let mut j = t;
        while j < size / 2 {
            let k = j % half;
            let p = j / half * (half * 2) + k;
            let phase = sign * core::f32::consts::TAU * k as f32 / (half * 2) as f32;
            let c = phase.cos(); let sn = phase.sin();
            let ar = re[p]; let ai = im[p];
            let br = re[p + half]; let bi = im[p + half];
            let tr = c * br - sn * bi;
            let ti = c * bi + sn * br;
            re[p] = ar + tr; im[p] = ai + ti;
            re[p + half] = ar - tr; im[p + half] = ai - ti;
            j += 256;
        }
        sync_ruda();
    }
    let mut k = t;
    while k < size {
        let mut rr = re[k]; let mut ii = im[k];
        let mut p = tile + n1 as usize * k;
        if comptime!(first) {
            let phase = sign * core::f32::consts::TAU * (k * tile) as f32 / n as f32;
            let c = phase.cos(); let sn = phase.sin();
            rr = re[k] * c - im[k] * sn;
            ii = im[k] * c + re[k] * sn;
            p = k * n2 as usize + tile;
        } else { rr *= scale; ii *= scale; }
        let mut value = Vector::<f32, Const<2>>::empty();
        value[0] = rr; value[1] = ii;
        out[row * n + p] = value;
        k += 256;
    }
}

#[ruda(launch)]
pub fn real_pad(x: &Array<f32>, out: &mut Array<Vector<f32, Const<2>>>, input_length: u32, length: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let row = i / length as usize;
        let col = i % length as usize;
        let mut value = Vector::<f32, Const<2>>::new(0.0f32);
        if col < input_length as usize { value[0] = x[row * input_length as usize + col]; }
        out[i] = value;
    }
}

#[ruda]
fn chirp(i: usize, n: u32, sign: f32) -> Vector<f32, Const<2>> {
    let square = (i as u64 * i as u64) % (2u64 * n as u64);
    let phase = sign * core::f32::consts::PI * square as f32 / n as f32;
    let mut result = Vector::<f32, Const<2>>::empty();
    result[0] = phase.cos(); result[1] = phase.sin();
    result
}

#[ruda]
fn product(a: Vector<f32, Const<2>>, b: Vector<f32, Const<2>>) -> Vector<f32, Const<2>> {
    let mut value = Vector::<f32, Const<2>>::empty();
    value[0] = a[0] * b[0] - a[1] * b[1];
    value[1] = a[0] * b[1] + a[1] * b[0];
    value
}

#[ruda(launch)]
pub fn chirp_input(x: &Array<Vector<f32, Const<2>>>, a: &mut Array<Vector<f32, Const<2>>>,
    b: &mut Array<Vector<f32, Const<2>>>, n: u32, padded: u32, sign: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < a.len() {
        let row = i / padded as usize;
        let col = i % padded as usize;
        let mut value = Vector::<f32, Const<2>>::new(0.0f32);
        if col < n as usize { value = product(x[row * n as usize + col], chirp(col, n, sign)); }
        a[i] = value;
        if row == 0 {
            let mut kernel = Vector::<f32, Const<2>>::new(0.0f32);
            if col < n as usize { kernel = chirp(col, n, -sign); }
            else if padded as usize - col < n as usize { kernel = chirp(padded as usize - col, n, -sign); }
            b[col] = kernel;
        }
    }
}

#[ruda(launch)]
pub fn multiply(a: &Array<Vector<f32, Const<2>>>, b: &Array<Vector<f32, Const<2>>>,
    out: &mut Array<Vector<f32, Const<2>>>, period: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() { out[i] = product(a[i], b[i % period as usize]); }
}

#[ruda(launch)]
pub fn chirp_finish(input: &Array<Vector<f32, Const<2>>>, out: &mut Array<Vector<f32, Const<2>>>,
    n: u32, padded: u32, sign: f32, scale: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() {
        let col = i % n as usize;
        out[i] = product(input[(i / n as usize) * padded as usize + col], chirp(col, n, sign)) * Vector::new(scale);
    }
}

#[ruda(launch)]
pub fn real_split(input: &Array<Vector<f32, Const<2>>>, re: &mut Array<f32>, im: &mut Array<f32>,
    n: u32, bins: u32, scale: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < re.len() {
        let value = input[i / bins as usize * n as usize + i % bins as usize];
        re[i] = value[0] * scale;
        im[i] = value[1] * scale;
    }
}
