use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn gather<F: Float>(input: &Array<F>, output: &mut Array<f32>,
    sequence: u32, start: u32, chunk: u32, width: u32,
    #[comptime] _source: String, #[define(F)] _dtype: StorageType)
{
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    let row = i / width as usize;
    let token = start as usize + row % chunk as usize;
    let mut value = 0f32;
    if token < sequence as usize {
        value = f32::cast_from(input[(row / chunk as usize * sequence as usize + token)
            * width as usize + i % width as usize]);
    }
    output[i] = value;
}

#[ruda(launch)]
pub(crate) fn scatter<F: Float>(input: &Array<f32>, output: &mut Array<F>,
    sequence: u32, start: u32, chunk: u32, width: u32,
    #[comptime] _source: String, #[define(F)] _dtype: StorageType)
{
    let i = ABSOLUTE_POS;
    if i >= input.len() { terminate!(); }
    let row = i / width as usize;
    let token = start as usize + row % chunk as usize;
    if token < sequence as usize {
        output[(row / chunk as usize * sequence as usize + token) * width as usize
            + i % width as usize] = F::cast_from(input[i]);
    }
}

#[ruda(launch)]
pub(crate) fn boundary(input: &Array<f32>, output: &mut Array<f32>,
    offset: u32, elements: u32, #[comptime] load: bool, #[comptime] _source: String)
{
    let i = ABSOLUTE_POS;
    if i < elements as usize {
        if comptime!(load) { output[i] = input[offset as usize + i]; }
        else { output[offset as usize + i] = input[i]; }
    }
}

#[ruda(launch)]
pub(crate) fn factors(q: &Array<f32>, k: &Array<f32>, g: &Array<f32>,
    prefix: &mut Array<f32>, tail: &mut Array<f32>, qg: &mut Array<f32>,
    kg: &mut Array<f32>, kt: &mut Array<f32>, chunk: u32, width: u32, scale: f32,
    #[comptime] _source: String)
{
    let i = ABSOLUTE_POS;
    let c = chunk as usize;
    let w = width as usize;
    if i >= q.len() / c { terminate!(); }
    let base = i / w * c * w + i % w;
    let mut sum = 0f32;
    for row in 0..c {
        let index = base + row * w;
        sum += g[index];
        let p = sum.exp();
        prefix[index] = p;
        qg[index] = q[index] * (scale * p);
        kg[index] = k[index] * p;
    }
    sum = 0f32;
    for offset in 0..c {
        let index = base + (c - 1 - offset) * w;
        let e = sum.exp();
        tail[index] = e;
        kt[index] = k[index] * e;
        sum += g[index];
    }
}

#[ruda(launch)]
pub(crate) fn pairs(q: &Array<f32>, k: &Array<f32>, g: &Array<f32>,
    kk: &mut Array<f32>, attention: &mut Array<f32>, chunk: u32, scale: f32,
    #[comptime] width: usize, #[comptime] lanes: usize, #[comptime] _source: String)
{
    let c = chunk as usize;
    let column = RUDA_POS_X as usize % c;
    let head = RUDA_POS_X as usize / c;
    let lane = UNIT_POS as usize;
    let mut log_segment = Array::<f32>::new((width + lanes - 1) / lanes);
    #[unroll]
    for j in 0usize..((width + lanes - 1) / lanes) { log_segment[j] = 0f32; }
    for row in 0..c {
        let mut ak = 0f32;
        let mut aq = 0f32;
        if row >= column {
            #[unroll]
            for j in 0usize..((width + lanes - 1) / lanes) {
                let d = lane + j * lanes;
                if d < width {
                    let r = (head * c + row) * width + d;
                    if row > column { log_segment[j] += g[r]; }
                    let carried = k[(head * c + column) * width + d] * log_segment[j].exp();
                    ak += k[r] * carried;
                    aq += q[r] * carried;
                }
            }
        }
        let key_dot = plane_sum(ak);
        let query_dot = plane_sum(aq);
        if lane == 0 {
            let index = (head * c + row) * c + column;
            kk[index] = key_dot;
            attention[index] = query_dot * scale;
        }
    }
}

#[ruda(launch)]
pub(crate) fn rhs(v: &Array<f32>, prediction: &Array<f32>, beta: &Array<f32>,
    output: &mut Array<f32>, width: u32, #[comptime] _source: String)
{
    let i = ABSOLUTE_POS;
    if i < output.len() { output[i] = beta[i / width as usize] * (v[i] - prediction[i]); }
}

#[ruda(launch)]
pub(crate) fn solve(kk: &Array<f32>, beta: &Array<f32>, rhs: &Array<f32>,
    out: &mut Array<f32>, columns: u32, #[comptime] chunk: usize,
    #[comptime] lanes: usize, #[comptime] transpose: bool, #[comptime] _source: String)
{
    let cols = columns as usize;
    let tiles = (cols + 3) / 4;
    let head = RUDA_POS_X as usize / tiles;
    let warp = UNIT_POS as usize / lanes;
    let lane = UNIT_POS as usize % lanes;
    let column = RUDA_POS_X as usize % tiles * 4 + warp;
    let mut solution = SharedMemory::<f32>::new(chunk * 4);
    for step in 0..chunk {
        let mut row = step;
        if comptime!(transpose) { row = chunk - 1 - step; }
        let mut sum = 0f32;
        let mut j = lane;
        while j < step {
            let mut col = j;
            if comptime!(transpose) { col = chunk - 1 - j; }
            let mut coefficient = kk[(head * chunk + row) * chunk + col] * beta[head * chunk + row];
            if comptime!(transpose) {
                coefficient = kk[(head * chunk + col) * chunk + row] * beta[head * chunk + col];
            }
            sum += coefficient * solution[col * 4 + warp];
            j += lanes;
        }
        let dot = plane_sum(sum);
        if lane == 0 {
            let mut value = 0f32;
            if column < cols { value = rhs[(head * chunk + row) * cols + column] - dot; }
            solution[row * 4 + warp] = value;
            if column < cols { out[(head * chunk + row) * cols + column] = value; }
        }
        sync_ruda();
    }
}

#[ruda(launch)]
pub(crate) fn combine(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>,
    alpha: f32, #[comptime] _source: String)
{
    let i = ABSOLUTE_POS;
    if i < out.len() { out[i] = a[i] + alpha * b[i]; }
}

#[ruda(launch)]
pub(crate) fn state_update(state: &Array<f32>, update: &Array<f32>, prefix: &Array<f32>,
    out: &mut Array<f32>, chunk: u32, kd: u32, vd: u32, #[comptime] _source: String)
{
    let i = ABSOLUTE_POS;
    if i >= out.len() { terminate!(); }
    let k = kd as usize;
    let v = vd as usize;
    let row = i / v;
    let last = (row / k * chunk as usize + chunk as usize - 1) * k + row % k;
    out[i] = state[i] * prefix[last] + update[i];
}

#[ruda(launch)]
pub(crate) fn beta_multiply(input: &Array<f32>, beta: &Array<f32>, out: &mut Array<f32>,
    width: u32, #[comptime] _source: String)
{
    let i = ABSOLUTE_POS;
    if i < out.len() { out[i] = input[i] * beta[i / width as usize]; }
}

#[ruda(launch)]
pub(crate) fn beta_gradient(db: &Array<f32>, v: &Array<f32>, prediction: &Array<f32>,
    dl: &Array<f32>, kk: &Array<f32>, out: &mut Array<f32>,
    chunk: u32, width: u32, #[comptime] _source: String)
{
    let row = ABSOLUTE_POS;
    if row >= out.len() { terminate!(); }
    let c = chunk as usize;
    let w = width as usize;
    let mut result = 0f32;
    for j in 0..w { result += db[row * w + j] * (v[row * w + j] - prediction[row * w + j]); }
    for j in 0..(row % c) { result -= dl[row * c + j] * kk[row * c + j]; }
    out[row] = result;
}

// dl contains dB @ U^T; the minus sign and strict-lower mask of dL are applied here.
#[ruda(launch)]
pub(crate) fn pair_rows(q: &Array<f32>, k: &Array<f32>, g: &Array<f32>, beta: &Array<f32>,
    da: &Array<f32>, dl: &Array<f32>, dqg: &Array<f32>, prefix: &Array<f32>,
    dq: &mut Array<f32>, dk: &mut Array<f32>, ends: &mut Array<f32>,
    chunk: u32, width: u32, scale: f32, #[comptime] _source: String)
{
    let index = ABSOLUTE_POS;
    if index >= dq.len() { terminate!(); }
    let c = chunk as usize;
    let w = width as usize;
    let row = index / w % c;
    let head = index / (w * c);
    let channel = index % w;
    let mut segment = 0f32;
    let mut qgrad = 0f32;
    let mut kgrad = 0f32;
    let mut gate_end = 0f32;
    for offset in 0..(row + 1) {
        let col = row - offset;
        let j = (head * c + col) * w + channel;
        let a = (head * c + row) * c + col;
        let carried = k[j] * segment.exp();
        let qa = da[a] * scale;
        let mut ka = 0f32;
        if col < row { ka = -dl[a] * beta[head * c + row]; }
        qgrad += qa * carried;
        kgrad += ka * carried;
        if col < row { gate_end += (qa * q[index] + ka * k[index]) * carried; }
        segment += g[j];
    }
    dq[index] = qgrad + dqg[index] * prefix[index] * scale;
    dk[index] = kgrad;
    ends[index] = gate_end;
}

#[ruda(launch)]
pub(crate) fn pair_columns(q: &Array<f32>, k: &Array<f32>, g: &Array<f32>, beta: &Array<f32>,
    da: &Array<f32>, dl: &Array<f32>, dk_rows: &Array<f32>, dkg_positive: &Array<f32>,
    dkt: &Array<f32>, prefix: &Array<f32>, tail: &Array<f32>,
    dk: &mut Array<f32>, starts: &mut Array<f32>,
    chunk: u32, width: u32, scale: f32, #[comptime] _source: String)
{
    let index = ABSOLUTE_POS;
    if index >= dk.len() { terminate!(); }
    let c = chunk as usize;
    let w = width as usize;
    let col = index / w % c;
    let head = index / (w * c);
    let channel = index % w;
    let mut segment = 0f32;
    let mut grad = 0f32;
    let mut gate_start = 0f32;
    for row in col..c {
        let i = (head * c + row) * w + channel;
        let a = (head * c + row) * c + col;
        let mut ka = 0f32;
        if row > col {
            segment += g[i];
            ka = -dl[a] * beta[head * c + row];
        }
        let contribution = (da[a] * scale * q[i] + ka * k[i]) * segment.exp();
        grad += contribution;
        if row > col { gate_start += contribution * k[index]; }
    }
    dk[index] = dk_rows[index] + grad - dkg_positive[index] * prefix[index] + dkt[index] * tail[index];
    starts[index] = gate_start;
}

#[ruda(launch)]
pub(crate) fn gate_gradient(qg: &Array<f32>, kg: &Array<f32>, kt: &Array<f32>,
    dqg: &Array<f32>, dkg_positive: &Array<f32>, dkt: &Array<f32>,
    starts: &Array<f32>, ends: &Array<f32>, state: &Array<f32>, dh: &Array<f32>,
    prefix: &Array<f32>, dg: &mut Array<f32>, chunk: u32, kd: u32, vd: u32,
    #[comptime] _source: String)
{
    let i = ABSOLUTE_POS;
    let c = chunk as usize;
    let k = kd as usize;
    let v = vd as usize;
    if i >= dg.len() / c { terminate!(); }
    let base = i / k * c * k + i % k;
    let mut state_term = 0f32;
    for col in 0..v { state_term += state[i * v + col] * dh[i * v + col]; }
    state_term *= prefix[base + (c - 1) * k];
    let mut sum = 0f32;
    for offset in 0..c {
        let j = base + (c - 1 - offset) * k;
        sum += dqg[j] * qg[j] - dkg_positive[j] * kg[j];
        dg[j] = sum + state_term;
    }
    let mut tail_sum = 0f32;
    let mut intervals = 0f32;
    for row in 0..c {
        let j = base + row * k;
        dg[j] += tail_sum + intervals;
        tail_sum += dkt[j] * kt[j];
        intervals += starts[j] - ends[j];
    }
}

#[ruda(launch)]
pub(crate) fn recurrent<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>, beta: &Array<f32>,
    g: &Array<f32>, initial: &Array<f32>, output: &mut Array<F>, final_state: &mut Array<f32>,
    sequence: u32, vd: u32, scale: f32, #[comptime] kd: usize, #[comptime] lanes: usize,
    #[comptime] _source: String, #[define(F)] _dtype: StorageType)
{
    let width = vd as usize;
    let tiles = (width + 3) / 4;
    let head = RUDA_POS_X as usize / tiles;
    let col = RUDA_POS_X as usize % tiles * 4 + UNIT_POS as usize / lanes;
    let lane = UNIT_POS as usize % lanes;
    let mut state = Array::<f32>::new((kd + lanes - 1) / lanes);
    #[unroll]
    for j in 0usize..((kd + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        state[j] = 0f32;
        if d < kd && col < width { state[j] = initial[(head * kd + d) * width + col]; }
    }
    for token in 0..sequence as usize {
        let row = head * sequence as usize + token;
        let mut prediction = 0f32;
        #[unroll]
        for j in 0usize..((kd + lanes - 1) / lanes) {
            let d = lane + j * lanes;
            if d < kd {
                state[j] *= g[row * kd + d].exp();
                prediction += f32::cast_from(k[row * kd + d]) * state[j];
            }
        }
        let predicted = plane_sum(prediction);
        let mut residual = 0f32;
        if col < width { residual = beta[row] * (f32::cast_from(v[row * width + col]) - predicted); }
        let mut result = 0f32;
        #[unroll]
        for j in 0usize..((kd + lanes - 1) / lanes) {
            let d = lane + j * lanes;
            if d < kd {
                state[j] += f32::cast_from(k[row * kd + d]) * residual;
                result += f32::cast_from(q[row * kd + d]) * state[j];
            }
        }
        let y = plane_sum(result) * scale;
        if lane == 0 && col < width { output[row * width + col] = F::cast_from(y); }
    }
    #[unroll]
    for j in 0usize..((kd + lanes - 1) / lanes) {
        let d = lane + j * lanes;
        if d < kd && col < width { final_state[(head * kd + d) * width + col] = state[j]; }
    }
}
