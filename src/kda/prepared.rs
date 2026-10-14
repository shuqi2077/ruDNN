use super::{KdaError, KdaInput, KdaOutput, KdaGradients, check_tensor, kernel, validate};
use crate::sequence_graph::{Builder, SequenceKernelPlan};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::prelude::*, tensor::{RudaTensor, permutation::swap_dims, reshape::reshape}};

fn source() -> String { include_str!("kernel.rs").to_owned() }
fn trans<R: Runtime>(x: &RudaTensor<R>) -> RudaTensor<R> { swap_dims(x.clone(), 1, 2) }

struct Factors<R: Runtime> {
    q: RudaTensor<R>, k: RudaTensor<R>, v: RudaTensor<R>, g: RudaTensor<R>, b: RudaTensor<R>,
    prefix: RudaTensor<R>, tail: RudaTensor<R>, qg: RudaTensor<R>, kg: RudaTensor<R>, kt: RudaTensor<R>,
    kk: RudaTensor<R>, attention: RudaTensor<R>,
}
struct Chunk<R: Runtime> {
    plan: Builder<R>, h: usize, t: usize, k: usize, v: usize, c: usize, lanes: usize, scale: f32,
}
impl<R: Runtime> Chunk<R> {
    fn new(input: &KdaInput<R>, dims: [usize;5], chunk: usize) -> Self {
        let [b,h,t,k,v] = dims;
        Self { plan: Builder::new(&input.query), h: b*h,t,k,v,c: chunk.min(t.max(1)),
            lanes: input.query.client.properties().hardware.plane_size_max as usize, scale: input.query_scale }
    }
    fn matrix(&mut self, rows: usize, cols: usize) -> RudaTensor<R> { self.plan.matrix(self.h,rows,cols) }
    fn gather(&mut self, input: &RudaTensor<R>, start: usize, width: usize) -> RudaTensor<R> {
        let out = self.matrix(self.c,width); let (grid,dim) = self.plan.grid(self.h*self.c*width);
        self.plan.kernels.push(kernel::gather::prepare::<R>(&self.plan.reference.client,grid,dim,
            input.clone().into_array_arg(),out.clone().into_array_arg(),self.t as u32,start as u32,
            self.c as u32,width as u32,source(),input.dtype.into())); out
    }
    fn scatter(&mut self, input: RudaTensor<R>, out: &RudaTensor<R>, start: usize, width: usize) {
        let (grid,dim) = self.plan.grid(self.h*self.c*width);
        self.plan.kernels.push(kernel::scatter::prepare::<R>(&self.plan.reference.client,grid,dim,
            input.into_array_arg(),out.clone().into_array_arg(),self.t as u32,start as u32,
            self.c as u32,width as u32,source(),out.dtype.into()));
    }
    fn boundary(&mut self, input: &RudaTensor<R>, out: &RudaTensor<R>, index: usize, load: bool) {
        let n=self.h*self.k*self.v; let (grid,dim)=self.plan.grid(n);
        self.plan.kernels.push(kernel::boundary::prepare::<R>(&self.plan.reference.client,grid,dim,
            input.clone().into_array_arg(),out.clone().into_array_arg(),(index*n) as u32,n as u32,load,source()));
    }
    fn combine(&mut self, a: RudaTensor<R>, b: RudaTensor<R>, alpha: f32) -> RudaTensor<R> {
        let out=self.plan.allocate(a.meta.shape().clone()); let (grid,dim)=self.plan.grid(out.meta.shape().iter().product());
        self.plan.kernels.push(kernel::combine::prepare::<R>(&self.plan.reference.client,grid,dim,
            a.into_array_arg(),b.into_array_arg(),out.clone().into_array_arg(),alpha,source())); out
    }
    fn solve(&mut self, p: &Factors<R>, rhs: RudaTensor<R>, transpose: bool) -> RudaTensor<R> {
        let out=self.matrix(self.c,self.v);
        self.plan.kernels.push(kernel::solve::prepare::<R>(&self.plan.reference.client,
            RudaCount::Static((self.h*self.v.div_ceil(4)) as u32,1,1),RudaDim::new_1d((4*self.lanes) as u32),
            p.kk.clone().into_array_arg(),p.b.clone().into_array_arg(),rhs.into_array_arg(),out.clone().into_array_arg(),
            self.v as u32,self.c,self.lanes,transpose,source())); out
    }
    fn update(&mut self, state: &RudaTensor<R>, update: RudaTensor<R>, prefix: &RudaTensor<R>) -> RudaTensor<R> {
        let out=self.matrix(self.k,self.v); let (grid,dim)=self.plan.grid(self.h*self.k*self.v);
        self.plan.kernels.push(kernel::state_update::prepare::<R>(&self.plan.reference.client,grid,dim,
            state.clone().into_array_arg(),update.into_array_arg(),prefix.clone().into_array_arg(),out.clone().into_array_arg(),
            self.c as u32,self.k as u32,self.v as u32,source())); out
    }
    fn factors(&mut self, input: &KdaInput<R>, start: usize) -> Factors<R> {
        let q=self.gather(&input.query,start,self.k); let k=self.gather(&input.key,start,self.k);
        let v=self.gather(&input.value,start,self.v); let g=self.gather(&input.log_decay,start,self.k);
        let b=self.gather(&input.beta,start,1); let prefix=self.matrix(self.c,self.k);
        let tail=self.matrix(self.c,self.k); let qg=self.matrix(self.c,self.k);
        let kg=self.matrix(self.c,self.k); let kt=self.matrix(self.c,self.k);
        let (grid,dim)=self.plan.grid(self.h*self.k);
        self.plan.kernels.push(kernel::factors::prepare::<R>(&self.plan.reference.client,grid,dim,
            q.clone().into_array_arg(),k.clone().into_array_arg(),g.clone().into_array_arg(),
            prefix.clone().into_array_arg(),tail.clone().into_array_arg(),qg.clone().into_array_arg(),
            kg.clone().into_array_arg(),kt.clone().into_array_arg(),self.c as u32,self.k as u32,self.scale,source()));
        let kk=self.matrix(self.c,self.c); let attention=self.matrix(self.c,self.c);
        self.plan.kernels.push(kernel::pairs::prepare::<R>(&self.plan.reference.client,
            RudaCount::Static((self.h*self.c) as u32,1,1),RudaDim::new_1d(self.lanes as u32),
            q.clone().into_array_arg(),k.clone().into_array_arg(),g.clone().into_array_arg(),kk.clone().into_array_arg(),
            attention.clone().into_array_arg(),self.c as u32,self.scale,self.k,self.lanes,source()));
        Factors {q,k,v,g,b,prefix,tail,qg,kg,kt,kk,attention}
    }
    fn residual(&mut self, p: &Factors<R>, state: &RudaTensor<R>) -> (RudaTensor<R>,RudaTensor<R>) {
        let prediction=self.plan.product(p.kg.clone(),state.clone()); let rhs=self.matrix(self.c,self.v);
        let (grid,dim)=self.plan.grid(self.h*self.c*self.v);
        self.plan.kernels.push(kernel::rhs::prepare::<R>(&self.plan.reference.client,grid,dim,
            p.v.clone().into_array_arg(),prediction.clone().into_array_arg(),p.b.clone().into_array_arg(),
            rhs.clone().into_array_arg(),self.v as u32,source()));
        (prediction,self.solve(p,rhs,false))
    }
}

fn packed<R: Runtime>(values: &[&RudaTensor<R>]) -> Result<(),KdaError> {
    if values.iter().any(|v| !v.is_contiguous()) { return Err(KdaError("prepared KDA requires contiguous bindings")); }
    Ok(())
}

pub fn prepare_chunk_kda<R: Runtime>(input: &KdaInput<R>, out: &KdaOutput<R>, chunk: usize)
    -> Result<SequenceKernelPlan<R>,KdaError>
{
    let dims=validate(input,chunk)?; let [b,h,t,k,v]=dims;
    if b==0 { return Err(KdaError("prepared KDA requires a nonempty batch")); }
    check_tensor(&out.output,&input.query,&[b,h,t,v],input.query.dtype)?;
    check_tensor(&out.final_state,&input.query,&[b,h,k,v],DType::F32)?;
    let mut bindings=vec![&input.query,&input.key,&input.value,&input.beta,&input.log_decay,&input.initial_state,&out.output,&out.final_state];
    if let Some(saved)=&out.boundaries {
        check_tensor(saved,&input.query,&[t.div_ceil(chunk),b,h,k,v],DType::F32)?; bindings.push(saved);
    }
    packed(&bindings)?;
    let mut op=Chunk::new(input,dims,chunk);
    let initial=reshape(input.initial_state.clone(),[b*h,k,v].into());
    let final_state=reshape(out.final_state.clone(),[b*h,k,v].into());
    if t==0 { op.plan.copy(&initial,&final_state); return Ok(op.plan.finish()); }
    if t==1 {
        if let Some(saved)=&out.boundaries { op.boundary(&initial,saved,0,false); }
        op.plan.kernels.push(kernel::recurrent::prepare::<R>(&op.plan.reference.client,
            RudaCount::Static((op.h*v.div_ceil(4)) as u32,1,1),RudaDim::new_1d((4*op.lanes) as u32),
            input.query.clone().into_array_arg(),input.key.clone().into_array_arg(),input.value.clone().into_array_arg(),
            input.beta.clone().into_array_arg(),input.log_decay.clone().into_array_arg(),input.initial_state.clone().into_array_arg(),
            out.output.clone().into_array_arg(),out.final_state.clone().into_array_arg(),t as u32,v as u32,op.scale,k,op.lanes,source(),input.query.dtype.into()));
        return Ok(op.plan.finish());
    }
    let state=op.matrix(k,v); op.plan.copy(&initial,&state); let mark=op.plan.mark();
    for start in (0..t).step_by(op.c) {
        op.plan.rewind(mark);
        if let Some(saved)=&out.boundaries { op.boundary(&state,saved,start/op.c,false); }
        let p=op.factors(input,start); let (_,residual)=op.residual(&p,&state);
        let inter=op.plan.product(p.qg,state.clone()); let intra=op.plan.product(p.attention,residual.clone());
        let result=op.combine(inter,intra,1.0); op.scatter(result,&out.output,start,v);
        let update=op.plan.product(trans(&p.kt),residual); let next=op.update(&state,update,&p.prefix);
        op.plan.copy(&next,&state);
    }
    op.plan.copy(&state,&final_state); Ok(op.plan.finish())
}

pub fn prepare_chunk_kda_backward<R: Runtime>(input: &KdaInput<R>, boundaries: &RudaTensor<R>,
    grad_output: &RudaTensor<R>, grad_final: &RudaTensor<R>, out: &KdaGradients<R>, chunk: usize)
    -> Result<SequenceKernelPlan<R>,KdaError>
{
    let dims=validate(input,chunk)?; let [b,h,t,k,v]=dims;
    if b==0 { return Err(KdaError("prepared KDA requires a nonempty batch")); }
    check_tensor(boundaries,&input.query,&[t.div_ceil(chunk),b,h,k,v],DType::F32)?;
    check_tensor(grad_output,&input.query,&[b,h,t,v],input.query.dtype)?;
    check_tensor(grad_final,&input.query,&[b,h,k,v],DType::F32)?;
    let inputs=[&input.query,&input.key,&input.value,&input.beta,&input.log_decay,&input.initial_state];
    let outputs=[&out.query,&out.key,&out.value,&out.beta,&out.log_decay,&out.initial_state];
    for (src,dst) in inputs.iter().zip(outputs.iter()) { check_tensor(dst,&input.query,src.meta.shape(),src.dtype)?; }
    packed(&inputs)?; packed(&outputs)?; packed(&[boundaries,grad_output,grad_final])?;
    let mut op=Chunk::new(input,dims,chunk); let dh=op.matrix(k,v);
    op.plan.copy(&reshape(grad_final.clone(),[b*h,k,v].into()),&dh); let mark=op.plan.mark();
    for index in (0..t.div_ceil(op.c)).rev() {
        op.plan.rewind(mark); let start=index*op.c; let state=op.matrix(k,v);
        op.boundary(boundaries,&state,index,true); let p=op.factors(input,start);
        let (prediction,residual)=op.residual(&p,&state); let dy=op.gather(grad_output,start,v);
        let du1=op.plan.product(trans(&p.attention),dy.clone()); let du2=op.plan.product(p.kt.clone(),dh.clone());
        let du=op.combine(du1,du2,1.0);
        let da=op.plan.product(dy.clone(),trans(&residual)); let dqg=op.plan.product(dy.clone(),trans(&state));
        let dkt=op.plan.product(residual.clone(),trans(&dh)); let drhs=op.solve(&p,du,true);
        let dl=op.plan.product(drhs.clone(),trans(&residual)); let dv=op.matrix(op.c,v);
        let (grid,dim)=op.plan.grid(op.h*op.c*v);
        op.plan.kernels.push(kernel::beta_multiply::prepare::<R>(&op.plan.reference.client,grid,dim,
            drhs.clone().into_array_arg(),p.b.clone().into_array_arg(),dv.clone().into_array_arg(),v as u32,source()));
        let dkg=op.plan.product(dv.clone(),trans(&state)); let db=op.matrix(op.c,1);
        let (grid,dim)=op.plan.grid(op.h*op.c);
        op.plan.kernels.push(kernel::beta_gradient::prepare::<R>(&op.plan.reference.client,grid,dim,
            drhs.into_array_arg(),p.v.clone().into_array_arg(),prediction.into_array_arg(),dl.clone().into_array_arg(),
            p.kk.clone().into_array_arg(),db.clone().into_array_arg(),op.c as u32,v as u32,source()));
        let dq=op.matrix(op.c,k); let dk_rows=op.matrix(op.c,k); let ends=op.matrix(op.c,k);
        let dk=op.matrix(op.c,k); let starts=op.matrix(op.c,k); let dg=op.matrix(op.c,k);
        let (grid,dim)=op.plan.grid(op.h*op.c*k);
        op.plan.kernels.push(kernel::pair_rows::prepare::<R>(&op.plan.reference.client,grid.clone(),dim,
            p.q.clone().into_array_arg(),p.k.clone().into_array_arg(),p.g.clone().into_array_arg(),p.b.clone().into_array_arg(),
            da.clone().into_array_arg(),dl.clone().into_array_arg(),dqg.clone().into_array_arg(),p.prefix.clone().into_array_arg(),
            dq.clone().into_array_arg(),dk_rows.clone().into_array_arg(),ends.clone().into_array_arg(),op.c as u32,k as u32,op.scale,source()));
        op.plan.kernels.push(kernel::pair_columns::prepare::<R>(&op.plan.reference.client,grid,dim,
            p.q.into_array_arg(),p.k.into_array_arg(),p.g.into_array_arg(),p.b.into_array_arg(),da.into_array_arg(),dl.into_array_arg(),
            dk_rows.into_array_arg(),dkg.clone().into_array_arg(),dkt.clone().into_array_arg(),p.prefix.clone().into_array_arg(),
            p.tail.into_array_arg(),dk.clone().into_array_arg(),starts.clone().into_array_arg(),op.c as u32,k as u32,op.scale,source()));
        let (grid,dim)=op.plan.grid(op.h*k);
        op.plan.kernels.push(kernel::gate_gradient::prepare::<R>(&op.plan.reference.client,grid,dim,
            p.qg.clone().into_array_arg(),p.kg.clone().into_array_arg(),p.kt.into_array_arg(),dqg.into_array_arg(),dkg.into_array_arg(),
            dkt.into_array_arg(),starts.into_array_arg(),ends.into_array_arg(),state.into_array_arg(),dh.clone().into_array_arg(),
            p.prefix.clone().into_array_arg(),dg.clone().into_array_arg(),op.c as u32,k as u32,v as u32,source()));
        let inter=op.plan.product(trans(&p.qg),dy); let pred=op.plan.product(trans(&p.kg),dv.clone());
        let update=op.combine(inter,pred,-1.0); let next=op.update(&dh,update,&p.prefix); op.plan.copy(&next,&dh);
        op.scatter(dq,&out.query,start,k); op.scatter(dk,&out.key,start,k); op.scatter(dv,&out.value,start,v);
        op.scatter(db,&out.beta,start,1); op.scatter(dg,&out.log_decay,start,k);
    }
    op.plan.copy(&dh,&reshape(out.initial_state.clone(),[b*h,k,v].into())); Ok(op.plan.finish())
}
