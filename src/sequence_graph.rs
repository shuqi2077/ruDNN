use ruda_core::tensor::{DType, Shape, Metadata};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::{RudaTensor,
    allocation::empty_device_contiguous_dtype, permutation::swap_dims}};
use ruda_kernel::tiling::InputBinding;
use rublas::kernel_ir::{definition::{MatmulElems, MatmulGlobalElems, F32MathMode},
    launch::launch_naive::prepare_canonical_ref};

pub struct SequenceKernelPlan<R: Runtime> {
    pub kernels: Vec<PreparedKernel<R>>,
    pub workspace: Vec<RudaTensor<R>>,
}

pub(crate) struct Builder<R: Runtime> {
    pub reference: RudaTensor<R>,
    pub kernels: Vec<PreparedKernel<R>>,
    workspace: Vec<RudaTensor<R>>,
    cursor: usize,
}

impl<R: Runtime> Builder<R> {
    pub fn new(reference: &RudaTensor<R>) -> Self {
        Self { reference: reference.clone(), kernels: Vec::new(), workspace: Vec::new(), cursor: 0 }
    }
    pub fn mark(&self) -> usize { self.cursor }
    pub fn rewind(&mut self, mark: usize) { self.cursor = mark; }
    pub fn allocate(&mut self, shape: Shape) -> RudaTensor<R> {
        assert!(shape.iter().try_fold(1usize, |n,&s| n.checked_mul(s)).is_some_and(|n| n<=u32::MAX as usize));
        if self.cursor == self.workspace.len() {
            self.workspace.push(empty_device_contiguous_dtype(self.reference.client.clone(),
                self.reference.device.clone(), shape.clone(), DType::F32));
        }
        let result = self.workspace[self.cursor].clone();
        assert_eq!(result.meta.shape(), &shape, "sequence graph workspace schedule changed");
        self.cursor += 1;
        result
    }
    pub fn matrix(&mut self, heads: usize, rows: usize, columns: usize) -> RudaTensor<R> {
        self.allocate([heads, rows, columns].into())
    }
    pub fn grid(&self, n: usize) -> (RudaCount, RudaDim) {
        let dim = RudaDim::new(self.reference.client.properties(), n);
        (kernel_dsl::calculate_ruda_count_elemwise(&self.reference.client, n, dim), dim)
    }
    pub fn copy(&mut self, src: &RudaTensor<R>, dst: &RudaTensor<R>) {
        let n: usize = dst.meta.shape().iter().product();
        assert_eq!(src.dtype, DType::F32); assert_eq!(dst.dtype, DType::F32);
        assert_eq!(src.meta.shape(), dst.meta.shape());
        assert_eq!(src.meta.num_dims(), 3); assert!(dst.is_contiguous());
        if n == 0 { return; }
        let (grid, dim) = self.grid(n);
        self.kernels.push(copy_matrix::prepare::<R>(&self.reference.client, grid, dim,
            src.clone().into_tensor_arg(), dst.clone().into_array_arg()));
    }
    fn row_major(&mut self, mut value: RudaTensor<R>) -> RudaTensor<R> {
        if value.is_contiguous() {
            let shape=value.meta.shape().clone(); let mut strides=vec![1;shape.len()];
            let mut stride=1;
            for axis in (0..shape.len()).rev() {strides[axis]=stride;stride*=shape[axis];}
            *value.meta=Metadata::new(shape,strides);
            return value;
        }
        let out = self.allocate(value.meta.shape().clone());
        self.copy(&value, &out); out
    }
    pub fn product(&mut self, a: RudaTensor<R>, b: RudaTensor<R>) -> RudaTensor<R> {
        assert_eq!(a.dtype,DType::F32); assert_eq!(b.dtype,DType::F32);
        assert!(a.qparams.is_none() && b.qparams.is_none());
        assert_eq!(a.meta.num_dims(),3); assert_eq!(b.meta.num_dims(),3);
        let shape = ruda_core::tensor::calculate_matmul_output(a.meta.shape(), b.meta.shape())
            .expect("sequence graph matrix shape mismatch");
        let a = self.row_major(a);
        let bt = self.row_major(swap_dims(b, 1, 2));
        let b = swap_dims(bt, 1, 2);
        let out = self.allocate(shape);
        let f32_type = DType::F32.into();
        let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            f32_math: F32MathMode::Strict, lhs: f32_type, rhs: f32_type, out: f32_type,
        });
        self.kernels.push(prepare_canonical_ref(&self.reference.client,
            InputBinding::new(a.binding(), f32_type), InputBinding::new(b.binding(), f32_type),
            out.clone().binding(), &dtypes).expect("sequence graph FP32 matrix preparation failed"));
        out
    }
    pub fn finish(self) -> SequenceKernelPlan<R> {
        SequenceKernelPlan { kernels: self.kernels, workspace: self.workspace }
    }
}

#[ruda(launch)]
fn copy_matrix(src: &Tensor<f32>, dst: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= dst.len() { terminate!(); }
    let columns = src.shape(2) as usize; let rows = src.shape(1) as usize;
    let offset = i / (rows * columns) * src.stride(0) as usize
        + i / columns % rows * src.stride(1) as usize + i % columns * src.stride(2) as usize;
    dst[i] = src[offset];
}
