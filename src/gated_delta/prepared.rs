use super::{GatedDeltaError, GatedDeltaInput, GatedDeltaOutput, chunk::prefix_tile, chunk_kernel as kernel, validate};
use crate::sequence_graph::{Builder, SequenceKernelPlan};
use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::{dsl::prelude::*, tensor::{RudaTensor, permutation::swap_dims, reshape::reshape}};

fn source() -> String { include_str!("chunk_kernel.rs").to_owned() }

struct Chunk<R: Runtime> { plan: Builder<R>, h: usize, t: usize, c: usize }
impl<R: Runtime> Chunk<R> {
    fn matrix(&mut self, rows: usize, cols: usize) -> RudaTensor<R> { self.plan.matrix(self.h,rows,cols) }
    fn prepare(&mut self, tensor: &RudaTensor<R>, beta: &RudaTensor<R>, cumulative: &RudaTensor<R>,
        start: usize, width: usize, scale: f32, mode: u32) -> RudaTensor<R>
    {
        let out=self.matrix(self.c,width); let (grid,dim)=self.plan.grid(self.h*self.c*width);
        self.plan.kernels.push(kernel::prepare::prepare::<R>(&self.plan.reference.client,grid,dim,
            tensor.clone().into_array_arg(),beta.clone().into_array_arg(),cumulative.clone().into_array_arg(),
            out.clone().into_array_arg(),self.t as u32,start as u32,self.c as u32,width as u32,
            scale,mode,source(),tensor.dtype.into())); out
    }
    fn mask(&mut self, products: RudaTensor<R>, cumulative: &RudaTensor<R>, strict: bool) -> RudaTensor<R> {
        let out=self.matrix(self.c,self.c); let (grid,dim)=self.plan.grid(self.h*self.c*self.c);
        self.plan.kernels.push(kernel::decay_mask::prepare::<R>(&self.plan.reference.client,grid,dim,
            products.into_array_arg(),cumulative.clone().into_array_arg(),out.clone().into_array_arg(),self.c as u32,strict,source())); out
    }
}

pub fn prepare_chunk_gated_delta_rule<R: Runtime>(input: &GatedDeltaInput<R>, out: &GatedDeltaOutput<R>, chunk: usize)
    -> Result<SequenceKernelPlan<R>,GatedDeltaError>
{
    let [b,h,t,k,v]=validate(input)?; let heads=b*h; let c=chunk;
    if c==0 || c>u32::MAX as usize { return Err(GatedDeltaError("chunk size must be positive and fit U32 indexing")); }
    let shared=c.checked_mul(c).and_then(|n|n.checked_add(c)).and_then(|n|n.checked_mul(4));
    if shared.is_none_or(|n|n>input.query.client.properties().hardware.max_shared_memory_size as usize) {
        return Err(GatedDeltaError("chunk triangular workspace exceeds device shared memory"));
    }
    for width in [c,k,v] {
        if heads.checked_mul(c).and_then(|n|n.checked_mul(width)).is_none_or(|n|n>u32::MAX as usize) {
            return Err(GatedDeltaError("chunk workspace exceeds U32 indexing"));
        }
    }
    for (tensor,shape,dtype) in [(&out.output,vec![b,h,t,v],input.query.dtype),(&out.final_state,vec![b,h,k,v],DType::F32)] {
        if tensor.meta.shape()[..]!=shape[..] || tensor.dtype!=dtype || tensor.qparams.is_some()
            || tensor.device.to_id()!=input.query.device.to_id()
        { return Err(GatedDeltaError("prepared gated delta output binding mismatch")); }
    }
    for tensor in [&input.query,&input.key,&input.value,&input.beta,&input.log_decay,&input.initial_state,&out.output,&out.final_state] {
        if !tensor.is_contiguous() { return Err(GatedDeltaError("prepared gated delta requires contiguous bindings")); }
    }
    let mut op=Chunk {plan:Builder::new(&input.query),h:heads,t,c};
    let initial=reshape(input.initial_state.clone(),[heads,k,v].into());
    let final_state=reshape(out.final_state.clone(),[heads,k,v].into());
    if t==0 { op.plan.copy(&initial,&final_state); return Ok(op.plan.finish()); }
    let state=op.matrix(k,v); op.plan.copy(&initial,&state); let mark=op.plan.mark();
    let tile=prefix_tile(heads,t,c); let plane=input.query.client.properties().hardware.plane_size_max;
    for start in (0..t).step_by(c) {
        op.plan.rewind(mark); let cumulative=op.matrix(c,1);
        op.plan.kernels.push(kernel::prefix::prepare::<R>(&op.plan.reference.client,RudaCount::Static(heads as u32,1,1),
            RudaDim::new_1d((tile/2).max(1) as u32),input.log_decay.clone().into_array_arg(),
            cumulative.clone().into_array_arg(),t as u32,start as u32,c,tile,source()));
        let q=op.prepare(&input.query,&input.beta,&cumulative,start,k,input.query_scale,0);
        let key=op.prepare(&input.key,&input.beta,&cumulative,start,k,1.0,0);
        let kb=op.prepare(&input.key,&input.beta,&cumulative,start,k,1.0,1);
        let vb=op.prepare(&input.value,&input.beta,&cumulative,start,v,1.0,1);
        let kk=op.plan.product(kb,swap_dims(key.clone(),1,2)); let triangular=op.mask(kk,&cumulative,true);
        let inverse=op.matrix(c,c);
        op.plan.kernels.push(kernel::triangular_inverse::prepare::<R>(&op.plan.reference.client,RudaCount::Static(heads as u32,1,1),
            RudaDim::new_1d(plane),triangular.into_array_arg(),inverse.clone().into_array_arg(),c,source()));
        let value=op.plan.product(inverse.clone(),vb);
        let kp=op.prepare(&input.key,&input.beta,&cumulative,start,k,1.0,2);
        let kc=op.plan.product(inverse,kp); let prediction=op.plan.product(kc,state.clone());
        let residual=op.matrix(c,v); let (grid,dim)=op.plan.grid(heads*c*v);
        op.plan.kernels.push(kernel::subtract::prepare::<R>(&op.plan.reference.client,grid.clone(),dim,
            value.into_array_arg(),prediction.into_array_arg(),residual.clone().into_array_arg(),source()));
        let qk=op.plan.product(q,swap_dims(key,1,2)); let attention=op.mask(qk,&cumulative,false);
        let qp=op.prepare(&input.query,&input.beta,&cumulative,start,k,input.query_scale,3);
        let inter=op.plan.product(qp,state.clone()); let intra=op.plan.product(attention,residual.clone());
        op.plan.kernels.push(kernel::write_output::prepare::<R>(&op.plan.reference.client,grid,dim,
            inter.into_array_arg(),intra.into_array_arg(),out.output.clone().into_array_arg(),
            t as u32,start as u32,c as u32,v as u32,source(),input.query.dtype.into()));
        let tail=op.prepare(&input.key,&input.beta,&cumulative,start,k,1.0,4);
        let update=op.plan.product(swap_dims(tail,1,2),residual); let next=op.matrix(k,v);
        let (grid,dim)=op.plan.grid(heads*k*v);
        op.plan.kernels.push(kernel::update_state::prepare::<R>(&op.plan.reference.client,grid,dim,
            state.clone().into_array_arg(),update.into_array_arg(),cumulative.into_array_arg(),next.clone().into_array_arg(),
            c as u32,(k*v) as u32,source())); op.plan.copy(&next,&state);
    }
    op.plan.copy(&state,&final_state); Ok(op.plan.finish())
}
