use super::{KdaError, KdaGradients, KdaInput, KdaOutput, check_size, check_tensor, kernel, validate};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::{RudaCount, RudaDim}},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous,
        permutation::swap_dims, reshape::reshape},
};
use rublas::tensor_matmul::{MatmulStrategy, matmul};

fn source() -> String { include_str!("kernel.rs").to_owned() }
fn transpose<R: Runtime>(x: &RudaTensor<R>) -> RudaTensor<R> { swap_dims(x.clone(), 1, 2) }

fn contiguous<R: Runtime>(input: KdaInput<R>) -> KdaInput<R> {
    KdaInput {
        query: into_contiguous(input.query), key: into_contiguous(input.key),
        value: into_contiguous(input.value), beta: into_contiguous(input.beta),
        log_decay: into_contiguous(input.log_decay), initial_state: into_contiguous(input.initial_state),
        query_scale: input.query_scale,
    }
}

struct Prepared<R: Runtime> {
    q: RudaTensor<R>, k: RudaTensor<R>, v: RudaTensor<R>, g: RudaTensor<R>, b: RudaTensor<R>,
    prefix: RudaTensor<R>, tail: RudaTensor<R>, qg: RudaTensor<R>, kg: RudaTensor<R>, kt: RudaTensor<R>,
    kk: RudaTensor<R>, attention: RudaTensor<R>,
}

struct Chunk<R: Runtime> {
    reference: RudaTensor<R>, h: usize, t: usize, k: usize, v: usize, c: usize, lanes: usize, scale: f32,
}

impl<R: Runtime> Chunk<R> {
    fn new(input: &KdaInput<R>, dims: [usize; 5], chunk: usize) -> Self {
        let [batch, heads, t, k, v] = dims;
        Self { reference: input.query.clone(), h: batch * heads, t, k, v,
            c: chunk.min(t.max(1)), lanes: input.query.client.properties().hardware.plane_size_max as usize,
            scale: input.query_scale }
    }

    fn allocate(&self, shape: Shape, dtype: DType) -> RudaTensor<R> {
        empty_device_contiguous_dtype(self.reference.client.clone(), self.reference.device.clone(), shape, dtype)
    }
    fn matrix(&self, rows: usize, columns: usize) -> RudaTensor<R> {
        self.allocate([self.h, rows, columns].into(), DType::F32)
    }
    fn grid(&self, elements: usize) -> (RudaCount, RudaDim) {
        let dim = RudaDim::new(self.reference.client.properties(), elements);
        (calculate_ruda_count_elemwise(&self.reference.client, elements, dim), dim)
    }
    fn product(&self, a: RudaTensor<R>, b: RudaTensor<R>) -> Result<RudaTensor<R>, KdaError> {
        let shape = ruda_core::tensor::calculate_matmul_output(a.meta.shape(), b.meta.shape())
            .map_err(|_| KdaError("incompatible KDA matrix shapes"))?;
        let output = self.allocate(shape, DType::F32);
        matmul(a, b, Some(output), MatmulStrategy::Naive, DType::F32)
            .map_err(|_| KdaError("KDA FP32 matrix product failed"))
    }
    fn gather(&self, input: &RudaTensor<R>, start: usize, width: usize) -> RudaTensor<R> {
        let result = self.matrix(self.c, width);
        let (grid, dim) = self.grid(self.h * self.c * width);
        kernel::gather::launch::<R>(&self.reference.client, grid, dim,
            input.clone().into_array_arg(), result.clone().into_array_arg(),
            self.t as u32, start as u32, self.c as u32, width as u32, source(), input.dtype.into());
        result
    }
    fn scatter(&self, input: RudaTensor<R>, output: &RudaTensor<R>, start: usize, width: usize) {
        let (grid, dim) = self.grid(self.h * self.c * width);
        kernel::scatter::launch::<R>(&self.reference.client, grid, dim,
            input.into_array_arg(), output.clone().into_array_arg(),
            self.t as u32, start as u32, self.c as u32, width as u32, source(), output.dtype.into());
    }
    fn boundary(&self, input: &RudaTensor<R>, output: &RudaTensor<R>, index: usize, load: bool) {
        let n = self.h * self.k * self.v;
        let (grid, dim) = self.grid(n);
        kernel::boundary::launch::<R>(&self.reference.client, grid, dim,
            input.clone().into_array_arg(), output.clone().into_array_arg(),
            (index * n) as u32, n as u32, load, source());
    }
    fn combine(&self, a: RudaTensor<R>, b: RudaTensor<R>, alpha: f32) -> RudaTensor<R> {
        let out = self.allocate(a.meta.shape().clone(), DType::F32);
        let (grid, dim) = self.grid(a.meta.shape().iter().product());
        kernel::combine::launch::<R>(&self.reference.client, grid, dim,
            a.into_array_arg(), b.into_array_arg(), out.clone().into_array_arg(), alpha, source());
        out
    }
    fn solve(&self, p: &Prepared<R>, rhs: RudaTensor<R>, transpose: bool) -> RudaTensor<R> {
        let out = self.matrix(self.c, self.v);
        kernel::solve::launch::<R>(&self.reference.client,
            RudaCount::Static((self.h * self.v.div_ceil(4)) as u32, 1, 1),
            RudaDim::new_1d((4 * self.lanes) as u32), p.kk.clone().into_array_arg(),
            p.b.clone().into_array_arg(), rhs.into_array_arg(), out.clone().into_array_arg(),
            self.v as u32, self.c, self.lanes, transpose, source());
        out
    }
    fn update(&self, state: RudaTensor<R>, update: RudaTensor<R>, prefix: &RudaTensor<R>) -> RudaTensor<R> {
        let result = self.matrix(self.k, self.v);
        let (grid, dim) = self.grid(self.h * self.k * self.v);
        kernel::state_update::launch::<R>(&self.reference.client, grid, dim,
            state.into_array_arg(), update.into_array_arg(), prefix.clone().into_array_arg(),
            result.clone().into_array_arg(), self.c as u32, self.k as u32, self.v as u32, source());
        result
    }
    fn prepare(&self, input: &KdaInput<R>, start: usize) -> Prepared<R> {
        let q = self.gather(&input.query, start, self.k);
        let k = self.gather(&input.key, start, self.k);
        let v = self.gather(&input.value, start, self.v);
        let g = self.gather(&input.log_decay, start, self.k);
        let b = self.gather(&input.beta, start, 1);
        let prefix = self.matrix(self.c, self.k);
        let tail = self.matrix(self.c, self.k);
        let qg = self.matrix(self.c, self.k);
        let kg = self.matrix(self.c, self.k);
        let kt = self.matrix(self.c, self.k);
        let (grid, dim) = self.grid(self.h * self.k);
        kernel::factors::launch::<R>(&self.reference.client, grid, dim,
            q.clone().into_array_arg(), k.clone().into_array_arg(), g.clone().into_array_arg(),
            prefix.clone().into_array_arg(), tail.clone().into_array_arg(), qg.clone().into_array_arg(),
            kg.clone().into_array_arg(), kt.clone().into_array_arg(),
            self.c as u32, self.k as u32, self.scale, source());
        let kk = self.matrix(self.c, self.c);
        let attention = self.matrix(self.c, self.c);
        kernel::pairs::launch::<R>(&self.reference.client, RudaCount::Static((self.h * self.c) as u32, 1, 1),
            RudaDim::new_1d(self.lanes as u32), q.clone().into_array_arg(), k.clone().into_array_arg(),
            g.clone().into_array_arg(), kk.clone().into_array_arg(), attention.clone().into_array_arg(),
            self.c as u32, self.scale, self.k, self.lanes, source());
        Prepared { q, k, v, g, b, prefix, tail, qg, kg, kt, kk, attention }
    }
    fn residual(&self, p: &Prepared<R>, state: &RudaTensor<R>)
        -> Result<(RudaTensor<R>, RudaTensor<R>), KdaError>
    {
        let prediction = self.product(p.kg.clone(), state.clone())?;
        let rhs = self.matrix(self.c, self.v);
        let (grid, dim) = self.grid(self.h * self.c * self.v);
        kernel::rhs::launch::<R>(&self.reference.client, grid, dim,
            p.v.clone().into_array_arg(), prediction.clone().into_array_arg(), p.b.clone().into_array_arg(),
            rhs.clone().into_array_arg(), self.v as u32, source());
        Ok((prediction, self.solve(p, rhs, false)))
    }
}

/// Chunk/WY-equivalent unit-lower solve, with channel-wise decay inside pair products.
/// Accumulation and state stay FP32; no [T,K,V] state history is materialized.
/// One-token decode uses a fused register-state recurrence. Inputs are never mutated.
pub fn chunk_kda<R: Runtime>(input: KdaInput<R>, chunk_size: usize, retain_boundaries: bool)
    -> Result<KdaOutput<R>, KdaError>
{
    let dims = validate(&input, chunk_size)?;
    let [batch, heads, t, k, v] = dims;
    let count = t.div_ceil(chunk_size);
    if retain_boundaries { check_size(&[count,batch,heads,k,v])?; }
    let input = contiguous(input);
    let op = Chunk::new(&input, dims, chunk_size);
    let output = op.allocate([batch,heads,t,v].into(), input.query.dtype);
    let boundaries = retain_boundaries.then(|| op.allocate([count,batch,heads,k,v].into(), DType::F32));
    let mut state = reshape(input.initial_state.clone(), [op.h,k,v].into());
    if t == 0 || batch == 0 {
        return Ok(KdaOutput { output, final_state: input.initial_state, boundaries });
    }
    if t == 1 {
        if let Some(saved) = &boundaries { op.boundary(&state, saved, 0, false); }
        let final_state = op.allocate([batch,heads,k,v].into(), DType::F32);
        kernel::recurrent::launch::<R>(&input.query.client,
            RudaCount::Static((op.h * v.div_ceil(4)) as u32, 1, 1), RudaDim::new_1d((4 * op.lanes) as u32),
            input.query.clone().into_array_arg(), input.key.into_array_arg(), input.value.into_array_arg(),
            input.beta.into_array_arg(), input.log_decay.into_array_arg(), input.initial_state.into_array_arg(),
            output.clone().into_array_arg(), final_state.clone().into_array_arg(),
            t as u32, v as u32, op.scale, k, op.lanes, source(), input.query.dtype.into());
        return Ok(KdaOutput { output, final_state, boundaries });
    }
    for start in (0..t).step_by(op.c) {
        if let Some(saved) = &boundaries { op.boundary(&state, saved, start / op.c, false); }
        let p = op.prepare(&input, start);
        let (_, residual) = op.residual(&p, &state)?;
        let inter = op.product(p.qg, state.clone())?;
        let intra = op.product(p.attention, residual.clone())?;
        op.scatter(op.combine(inter, intra, 1.0), &output, start, v);
        let update = op.product(transpose(&p.kt), residual)?;
        state = op.update(state, update, &p.prefix);
    }
    Ok(KdaOutput { output, final_state: reshape(state, [batch,heads,k,v].into()), boundaries })
}

/// Exact first-order adjoint of the chunk equations, including final-state gradients.
/// Only boundary states survive forward; each chunk is rematerialized on device.
pub fn chunk_kda_backward<R: Runtime>(input: KdaInput<R>, boundaries: RudaTensor<R>,
    grad_output: RudaTensor<R>, grad_final_state: RudaTensor<R>, chunk_size: usize)
    -> Result<KdaGradients<R>, KdaError>
{
    let dims = validate(&input, chunk_size)?;
    let [batch, heads, t, k, v] = dims;
    check_tensor(&boundaries, &input.query, &[t.div_ceil(chunk_size),batch,heads,k,v], DType::F32)?;
    check_tensor(&grad_output, &input.query, &[batch,heads,t,v], input.query.dtype)?;
    check_tensor(&grad_final_state, &input.query, &[batch,heads,k,v], DType::F32)?;
    let input = contiguous(input);
    let boundaries = into_contiguous(boundaries);
    let grad_output = into_contiguous(grad_output);
    let op = Chunk::new(&input, dims, chunk_size);
    let dq = op.allocate([batch,heads,t,k].into(), input.query.dtype);
    let dk = op.allocate([batch,heads,t,k].into(), input.query.dtype);
    let dv = op.allocate([batch,heads,t,v].into(), input.query.dtype);
    let dbeta = op.allocate([batch,heads,t].into(), DType::F32);
    let dg = op.allocate([batch,heads,t,k].into(), DType::F32);
    let mut dh = reshape(into_contiguous(grad_final_state), [op.h,k,v].into());
    if t != 0 && batch != 0 {
        for index in (0..t.div_ceil(op.c)).rev() {
            let start = index * op.c;
            let state = op.matrix(k, v);
            op.boundary(&boundaries, &state, index, true);
            let p = op.prepare(&input, start);
            let (prediction, residual) = op.residual(&p, &state)?;
            let dy = op.gather(&grad_output, start, v);
            let du = op.combine(op.product(transpose(&p.attention), dy.clone())?,
                op.product(p.kt.clone(), dh.clone())?, 1.0);
            let da = op.product(dy.clone(), transpose(&residual))?;
            let dqg = op.product(dy.clone(), transpose(&state))?;
            let dkt = op.product(residual.clone(), transpose(&dh))?;
            let drhs = op.solve(&p, du, true);
            let dl_positive = op.product(drhs.clone(), transpose(&residual))?;
            let dv_chunk = op.matrix(op.c, v);
            let (grid, dim) = op.grid(op.h * op.c * v);
            kernel::beta_multiply::launch::<R>(&op.reference.client, grid, dim,
                drhs.clone().into_array_arg(), p.b.clone().into_array_arg(), dv_chunk.clone().into_array_arg(),
                v as u32, source());
            let dkg_positive = op.product(dv_chunk.clone(), transpose(&state))?;
            let db_chunk = op.matrix(op.c, 1);
            let (grid, dim) = op.grid(op.h * op.c);
            kernel::beta_gradient::launch::<R>(&op.reference.client, grid, dim,
                drhs.into_array_arg(), p.v.clone().into_array_arg(), prediction.into_array_arg(),
                dl_positive.clone().into_array_arg(), p.kk.clone().into_array_arg(), db_chunk.clone().into_array_arg(),
                op.c as u32, v as u32, source());
            let dq_chunk = op.matrix(op.c, k);
            let dk_rows = op.matrix(op.c, k);
            let ends = op.matrix(op.c, k);
            let dk_chunk = op.matrix(op.c, k);
            let starts = op.matrix(op.c, k);
            let dg_chunk = op.matrix(op.c, k);
            let (grid, dim) = op.grid(op.h * op.c * k);
            kernel::pair_rows::launch::<R>(&op.reference.client, grid.clone(), dim,
                p.q.clone().into_array_arg(), p.k.clone().into_array_arg(), p.g.clone().into_array_arg(),
                p.b.clone().into_array_arg(), da.clone().into_array_arg(), dl_positive.clone().into_array_arg(),
                dqg.clone().into_array_arg(), p.prefix.clone().into_array_arg(), dq_chunk.clone().into_array_arg(),
                dk_rows.clone().into_array_arg(), ends.clone().into_array_arg(),
                op.c as u32, k as u32, op.scale, source());
            kernel::pair_columns::launch::<R>(&op.reference.client, grid, dim,
                p.q.into_array_arg(), p.k.into_array_arg(), p.g.into_array_arg(), p.b.into_array_arg(),
                da.into_array_arg(), dl_positive.into_array_arg(), dk_rows.into_array_arg(),
                dkg_positive.clone().into_array_arg(), dkt.clone().into_array_arg(),
                p.prefix.clone().into_array_arg(), p.tail.into_array_arg(), dk_chunk.clone().into_array_arg(),
                starts.clone().into_array_arg(), op.c as u32, k as u32, op.scale, source());
            let (grid, dim) = op.grid(op.h * k);
            kernel::gate_gradient::launch::<R>(&op.reference.client, grid, dim,
                p.qg.clone().into_array_arg(), p.kg.clone().into_array_arg(), p.kt.into_array_arg(),
                dqg.into_array_arg(), dkg_positive.into_array_arg(), dkt.into_array_arg(),
                starts.into_array_arg(), ends.into_array_arg(), state.into_array_arg(), dh.clone().into_array_arg(),
                p.prefix.clone().into_array_arg(), dg_chunk.clone().into_array_arg(),
                op.c as u32, k as u32, v as u32, source());
            let inter = op.product(transpose(&p.qg), dy)?;
            let prediction_grad = op.product(transpose(&p.kg), dv_chunk.clone())?;
            dh = op.update(dh, op.combine(inter, prediction_grad, -1.0), &p.prefix);
            op.scatter(dq_chunk, &dq, start, k);
            op.scatter(dk_chunk, &dk, start, k);
            op.scatter(dv_chunk, &dv, start, v);
            op.scatter(db_chunk, &dbeta, start, 1);
            op.scatter(dg_chunk, &dg, start, k);
        }
    }
    Ok(KdaGradients { query: dq, key: dk, value: dv, beta: dbeta, log_decay: dg,
        initial_state: reshape(dh, [batch,heads,k,v].into()) })
}
