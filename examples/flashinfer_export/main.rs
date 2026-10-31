use std::{fs, path::PathBuf};
use ruda_compiler::ptx::{PtxCompilationOptions, PtxCompiler, PtxTarget};
use ruda_core::{compiler::Compiler,
    ir::{FloatKind, IntKind, UIntKind, ManagedVariable, Scope, Type, Variable, VariableKind},
    kernel::{KernelArg, KernelDefinition, KernelOptions, ScalarKernelArg, Visibility},
    launch::{ExecutionMode, RudaDim}};
use ruda_kernel::dsl::prelude::Const;
use rudnn::standalone::{self, attention, moe};

struct Export {
    kernel: KernelDefinition,
    widths: Vec<usize>,
    scalar_types: Vec<&'static str>,
    maps: Vec<(u32, usize, usize)>,
}

impl Export {
    fn new(name: String, x: u32, y: u32) -> Self {
        let mut kernel = KernelDefinition {
            buffers: vec![], tensor_maps: vec![], scalars: vec![],
            ruda_dim: RudaDim::new_2d(x, y), body: Scope::root(false),
            options: KernelOptions { kernel_name: name, ..Default::default() },
        };
        kernel.body.register_type::<usize>(UIntKind::U32.into());
        Self { kernel, widths: vec![], scalar_types: vec![], maps: vec![] }
    }
    fn buffer(&mut self, ty: Type, width: usize, write: bool) -> ManagedVariable {
        let id = self.widths.len() as u32;
        let ty = ty.with_vector_size(width);
        self.widths.push(width);
        self.kernel.buffers.push(KernelArg { id, visibility: if write { Visibility::ReadWrite } else { Visibility::Read },
            ty, size: None, has_extended_meta: false });
        ManagedVariable::Plain(Variable::new(if write { VariableKind::GlobalOutputArray(id) } else { VariableKind::GlobalInputArray(id) }, ty))
    }
    fn tensor_map(&mut self, ty: Type, rows: usize, columns: usize) -> ManagedVariable {
        let id = self.widths.len() as u32;
        self.widths.push(1);
        self.maps.push((id, rows, columns));
        self.kernel.tensor_maps.push(KernelArg { id, visibility: Visibility::Read,
            ty, size: None, has_extended_meta: false });
        ManagedVariable::Plain(Variable::new(VariableKind::TensorMapInput(id), ty))
    }
    fn scalar(&mut self, kind: &'static str) -> ManagedVariable {
        let ty = match kind { "I" => Type::from(UIntKind::U32), "i" => Type::from(IntKind::I32), "f" => Type::from(FloatKind::F32), _ => unreachable!() };
        let position = self.scalar_types.iter().position(|&t|t==kind).unwrap_or_else(|| {
            self.scalar_types.push(kind);
            self.kernel.scalars.push(ScalarKernelArg { ty: ty.storage_type(), count: 0 });
            self.scalar_types.len()-1
        });
        let index = self.kernel.scalars[position].count;
        self.kernel.scalars[position].count += 1;
        ManagedVariable::Plain(Variable::new(VariableKind::GlobalScalar(index as u32), ty))
    }
    fn save(self, directory: &PathBuf, index: &mut String) {
        let scalars = self.scalar_types.iter().zip(&self.kernel.scalars).map(|(t,a)|format!("{}:{}", t,a.count)).collect::<Vec<_>>().join(",");
        let widths = self.widths.iter().map(|n|n.to_string()).collect::<Vec<_>>().join(",");
        let name = self.kernel.options.kernel_name.clone();
        let maps = self.maps.iter().map(|(id, r, c)|format!("{id}:{r}:{c}")).collect::<Vec<_>>().join(",");
        let options = PtxCompilationOptions { target: Some(PtxTarget { version: (8,7), sm: 100 }) };
        let code = PtxCompiler.compile(self.kernel, &options, ExecutionMode::Unchecked, UIntKind::U32.into()).unwrap_or_else(|e|panic!("{name}: {e:?}"));
        assert!(code.dynamic_metadata_index.is_none());
        fs::write(directory.join(format!("{name}.ptx")), code.source).unwrap();
        index.push_str(&format!("{name}\t{}, {}, {}\t{}\t{widths}\t{scalars}\t{maps}\n",code.ruda_dim.x,code.ruda_dim.y,code.ruda_dim.z,code.shared_memory_bytes));
        println!("exported {name}");
    }
}

fn main() {
    let directory = PathBuf::from(std::env::args().nth(1).expect("output directory"));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("gemm_policy.json"), rublas::tensor_matmul::export_gemm_policy()).unwrap();
    let mut index = String::new();
    let bf = Type::from(FloatKind::BF16);
    let fp = Type::from(FloatKind::F32);
    let hp = Type::from(FloatKind::F16);
    let int = Type::from(IntKind::I32);
    let uint = Type::from(UIntKind::U32);
    let byte = Type::from(UIntKind::U8);
    for width in [128,512,1536,2048,4096,7168] {
        for add in [false,true] {
            if add && width < 2048 { continue; }
            let threads = if width <= 512 {32} else if width <= 2048 {64} else {128};
            let rows = if width <= 2048 {4} else {2};
            let mut e = Export::new(format!("rms_{width}_{}",u32::from(add)),threads * rows,1);
            let x=e.buffer(bf,8,false);let r=e.buffer(bf,8,false);let w=e.buffer(bf,8,false);let o=e.buffer(bf,8,true);
            standalone::rms::expand::<half::bf16>(&mut e.kernel.body,x.into(),r.into(),w.into(),o.into(),width,threads as usize,rows as usize,add,bf.storage_type());
            e.save(&directory,&mut index);
        }
    }
    for rows in 1usize..=4 {
        let mut e=Export::new(format!("gemv_{rows}"),32,4);
        let a=e.buffer(hp,8,false);let b=e.buffer(hp,8,false);let o=e.buffer(hp,1,true);
        let k=e.scalar("I");let n=e.scalar("I");let cs=e.scalar("I");let rs=e.scalar("I");let os=e.scalar("I");
        rublas::tensor_matmul::export_gemv_vector::expand::<half::f16,half::f16,Const<8>>(&mut e.kernel.body,
            8usize,a.into(),b.into(),o.into(),k.into(),n.into(),cs.into(),rs.into(),os.into(),rows,hp.storage_type(),hp.storage_type());
        e.save(&directory,&mut index);
    }
    for (bm,bn,wr,wc,stages) in [(16,64,1,2,3),(32,64,1,2,3),(64,64,2,2,2),(128,64,2,2,3),(128,128,2,4,3)] {
        for partitions in [1,2,4,8,16] {
            let mut e=Export::new(format!("mma_{bm}_{bn}_{partitions}"),32,wr*wc);
            let out_ty=if partitions==1 {hp}else{fp};
            let a=e.buffer(hp,1,false);let b=e.buffer(hp,1,false);let o=e.buffer(out_ty,1,true);let bias=e.buffer(out_ty,1,false);
            let m=e.scalar("I");let n=e.scalar("I");let k=e.scalar("I");let pk=e.scalar("I");
            let alpha=e.scalar("f");let beta=e.scalar("f");let br=e.scalar("I");let bc=e.scalar("I");
            macro_rules! expand { ($output:ty) => {
                rublas::tensor_matmul::export_mma_pipeline::expand::<half::f16,$output>(&mut e.kernel.body,
                    a.into(),b.into(),o.into(),bias.into(),m.into(),n.into(),k.into(),pk.into(),alpha.into(),beta.into(),br.into(),bc.into(),
                    hp.storage_type(),out_ty.storage_type(),wr as usize,wc as usize,bm,bn,true,stages,4,64,true,true,false,false,false,partitions,
                    Vec::new(),Vec::new(),Vec::new(),0,0);
            }; }
            if partitions==1 { expand!(half::f16); } else { expand!(f32); }
            e.save(&directory,&mut index);
        }
    }
    for bn in [128,256] {
        for partitions in [1,2,4,8,16] {
            let mut e=Export::new(format!("blackwell_128_{bn}_{partitions}"),128,1);
            let out_ty=if partitions==1 {hp}else{fp};
            let a=e.tensor_map(hp,128,64);let b=e.tensor_map(hp,bn,64);
            let o=e.buffer(out_ty,1,true);let bias=e.buffer(out_ty,1,false);
            let m=e.scalar("I");let n=e.scalar("I");let k=e.scalar("I");let pk=e.scalar("I");let workers=e.scalar("I");
            let alpha=e.scalar("f");let beta=e.scalar("f");let br=e.scalar("I");let bc=e.scalar("I");
            macro_rules! expand { ($output:ty) => {
                rublas::tensor_matmul::export_blackwell_gemm::expand::<half::f16,$output>(&mut e.kernel.body,
                    a.into(),b.into(),o.into(),bias.into(),m.into(),n.into(),k.into(),pk.into(),workers.into(),
                    alpha.into(),beta.into(),br.into(),bc.into(),hp.storage_type(),out_ty.storage_type(),
                    bn,3,false,partitions,false,false,vec![],vec![],vec![]);
            }; }
            if partitions==1 {expand!(half::f16);} else {expand!(f32);}
            e.save(&directory,&mut index);
        }
    }
    for bn in [128,256] {
        let mut e = Export::new(format!("blackwell_ws_{bn}"),192,1);
        let a = e.tensor_map(hp,128,64); let b = e.tensor_map(hp,bn,64);
        let out = e.buffer(hp,8,true); let bias = e.buffer(hp,1,false); let residual = e.buffer(hp,8,false);
        let sa = e.buffer(fp,1,false); let sb = e.buffer(fp,1,false); let groups = e.buffer(uint,1,false);
        let scalars: Vec<_> = (0..27).map(|_|e.scalar("I")).collect();
        let alpha = e.scalar("f"); let beta = e.scalar("f");
        let s = |i: usize| scalars[i].clone().into();
        standalone::sol_gemm::gemm_ws::expand::<half::f16,half::f16>(&mut e.kernel.body,
            a.into(),b.into(),out.into(),bias.into(),residual.into(),sa.into(),sb.into(),groups.into(),
            s(0),s(1),s(2),s(3),s(4),s(5),s(6),s(7),s(8),s(9),s(10),s(11),s(12),
            s(13),s(14),s(15),s(16),s(17),s(18),s(19),s(20),s(21),s(22),s(23),s(24),s(25),s(26),
            alpha.into(),beta.into(),hp.storage_type(),hp.storage_type(),bn,if bn==256 {4}else{6},
            0u32,false,false,0u32,0u32,0u32,0u32,0u32,false);
        e.save(&directory,&mut index);
    }
    for partitions in [2,4,8,16] {
        let mut e=Export::new(format!("split_reduce_{partitions}"),256,1);
        let a=e.buffer(fp,1,false);let o=e.buffer(hp,1,true);let n=e.scalar("I");
        standalone::split_reduce::expand::<half::f16>(&mut e.kernel.body,a.into(),o.into(),n.into(),partitions,hp.storage_type());
        e.save(&directory,&mut index);
    }
    for (label,heads,kv_heads,dim,pos,prefill,paged,nan) in [
        ("gqa_decode4",32,4,128,0,false,true,false), ("gqa_decode8",32,8,128,0,false,true,false),
        ("gqa_paged4",32,4,128,0,true,true,false), ("gqa_paged8",32,8,128,0,true,true,false),
        ("gqa_ragged4",32,4,128,0,true,false,true), ("gqa_ragged8",32,8,128,0,true,false,true),
        ("mla_decode",16,1,512,64,false,true,false), ("mla_prefill",16,1,512,64,true,true,true)] {
        for warps in [1usize,4] {
            for finalize in [false,true] {
                let mut e=Export::new(format!("{label}_w{warps}_{}",u32::from(finalize)),32,warps as u32);
                let q=e.buffer(bf,1,false);let k=e.buffer(bf,1,false);let v=e.buffer(bf,1,false);let qp=e.buffer(bf,1,false);let kp=e.buffer(bf,1,false);
                let qo=e.buffer(int,1,false);let kv=e.buffer(int,1,false);let pages=e.buffer(int,1,false);let partials=e.buffer(fp,1,true);
                let out=e.buffer(bf,1,true);let lse=e.buffer(fp,1,true);
                let scale=e.scalar("f");let splits=e.scalar("I");
                attention::csr_attention::expand::<half::bf16>(&mut e.kernel.body,q.into(),k.into(),v.into(),qp.into(),kp.into(),qo.into(),kv.into(),pages.into(),partials.into(),out.into(),lse.into(),
                    scale.into(),splits.into(),heads,kv_heads,dim,pos,prefill,paged,nan,warps,finalize,bf.storage_type());
                e.save(&directory,&mut index);
            }
        }
    }
    for dim in [128,512] {
        let mut e=Export::new(format!("merge_lse_{dim}"),32,1);
        let p=e.buffer(fp,1,false);let o=e.buffer(bf,1,true);let l=e.buffer(fp,1,true);let s=e.scalar("I");
        attention::merge_lse::expand::<half::bf16>(&mut e.kernel.body,p.into(),o.into(),l.into(),s.into(),dim,bf.storage_type());
        e.save(&directory,&mut index);
    }
    let mut e=Export::new("moe_route".into(),128,1);
    let a=e.buffer(fp,1,false);let b=e.buffer(bf,1,false);let w=e.buffer(fp,1,true);let s=e.buffer(int,1,true);let off=e.scalar("i");let factor=e.scalar("f");
    moe::route::expand::<half::bf16>(&mut e.kernel.body,a.into(),b.into(),w.into(),s.into(),off.into(),factor.into(),bf.storage_type());
    e.save(&directory,&mut index);
    let mut e=Export::new("moe_compact".into(),32,1);
    let s=e.buffer(int,1,false);let r=e.buffer(uint,1,true);let c=e.buffer(uint,1,true);let p=e.buffer(int,1,true);let t=e.scalar("I");
    moe::compact::expand(&mut e.kernel.body,s.into(),r.into(),c.into(),p.into(),t.into());e.save(&directory,&mut index);
    let mut e=Export::new("moe_offsets".into(),32,1);
    let c=e.buffer(uint,1,false);let o=e.buffer(uint,1,true);
    moe::offsets::expand(&mut e.kernel.body,c.into(),o.into());e.save(&directory,&mut index);
    let mut e=Export::new("moe_swiglu".into(),128,1);
    let a=e.buffer(byte,1,false);let scale=e.buffer(fp,1,false);let w=e.buffer(byte,1,false);let ws=e.buffer(fp,1,false);
    let r=e.buffer(uint,1,false);let c=e.buffer(uint,1,false);let off=e.buffer(uint,1,false);let o=e.buffer(fp,1,true);let t=e.scalar("I");
    rudnn::grouped_fp8::swiglu::expand(&mut e.kernel.body,a.into(),scale.into(),w.into(),ws.into(),r.into(),c.into(),off.into(),o.into(),t.into(),7168usize,2048usize);
    e.save(&directory,&mut index);
    let mut e=Export::new("moe_swiglu_sm100".into(),128,1);
    let a=e.buffer(byte,1,false);let scale=e.buffer(fp,1,false);let w=e.buffer(byte,1,false);let ws=e.buffer(fp,1,false);
    let r=e.buffer(uint,1,false);let c=e.buffer(uint,1,false);let off=e.buffer(uint,1,false);let o=e.buffer(fp,1,true);let t=e.scalar("I");
    rudnn::grouped_fp8::blackwell::swiglu::expand(&mut e.kernel.body,a.into(),scale.into(),w.into(),ws.into(),r.into(),c.into(),off.into(),o.into(),t.into(),7168usize,2048usize);
    e.save(&directory,&mut index);
    for first in [true,false] {
        let mut e=Export::new(format!("moe_project{}",if first {1}else{2}),128,1);
        let a=e.buffer(byte,1,false);let act=e.buffer(fp,1,false);let scale=e.buffer(fp,1,false);
        let w=e.buffer(byte,1,false);let ws=e.buffer(fp,1,false);let r=e.buffer(uint,1,false);let c=e.buffer(uint,1,false);let off=e.buffer(uint,1,false);let o=e.buffer(fp,1,true);let t=e.scalar("I");
        moe::project::expand(&mut e.kernel.body,a.into(),act.into(),scale.into(),w.into(),ws.into(),r.into(),c.into(),off.into(),o.into(),t.into(),first);
        e.save(&directory,&mut index);
    }
    let mut e=Export::new("moe_activate".into(),256,1);
    let a=e.buffer(fp,1,false);let c=e.buffer(uint,1,false);let off=e.buffer(uint,1,false);let o=e.buffer(fp,1,true);
    moe::activate::expand(&mut e.kernel.body,a.into(),c.into(),off.into(),o.into());e.save(&directory,&mut index);
    let mut e=Export::new("moe_combine".into(),256,1);
    let a=e.buffer(fp,1,false);let p=e.buffer(int,1,false);let off=e.buffer(uint,1,false);let w=e.buffer(fp,1,false);let o=e.buffer(bf,1,true);
    moe::combine::expand::<half::bf16>(&mut e.kernel.body,a.into(),p.into(),off.into(),w.into(),o.into(),bf.storage_type());e.save(&directory,&mut index);
    fs::write(directory.join("manifest.tsv"), index).unwrap();
}
