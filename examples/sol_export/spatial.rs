use std::path::PathBuf;
use ruda_core::ir::{FloatKind, Type};
use rudnn::standalone::sol::spatial;
use super::Export;

pub fn export(d: &PathBuf, ix: &mut String) {
    let fp = Type::from(FloatKind::F32);
    let mut e = Export::new("dwconv2d_input_grad", 256, 1);
    let g = e.buffer(fp, 1, false); let w = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let c = e.scalar("I"); let h = e.scalar("I"); let wi = e.scalar("I");
    spatial::depthwise2d_input_grad::expand(&mut e.kernel.body, g.into(), w.into(), o.into(), c.into(), h.into(), wi.into()); e.save(d, ix);
    let mut e = Export::new("dwconv2d_weight_grad", 256, 1);
    let x = e.buffer(fp, 1, false); let g = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let c = e.scalar("I"); let h = e.scalar("I"); let wi = e.scalar("I");
    spatial::depthwise2d_weight_grad::expand(&mut e.kernel.body, x.into(), g.into(), o.into(), c.into(), h.into(), wi.into()); e.save(d, ix);
    let mut e = Export::new("maxpool2_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true); let h = e.scalar("I"); let w = e.scalar("I");
    spatial::pool2::expand(&mut e.kernel.body, x.into(), o.into(), h.into(), w.into()); e.save(d, ix);
    let mut e = Export::new("bilinear2_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let h = e.scalar("I"); let w = e.scalar("I"); let oh = e.scalar("I"); let ow = e.scalar("I");
    spatial::resize2::expand(&mut e.kernel.body, x.into(), o.into(), h.into(), w.into(), oh.into(), ow.into()); e.save(d, ix);
    macro_rules! arithmetic {
        ($tag:expr, $f:ty, $o:ty, $ft:expr, $ot:expr) => {{
            let ft = Type::from($ft); let ot = Type::from($ot);
            let mut e = Export::new(&format!("affine_{}", $tag), 256, 1);
            let x = e.buffer(ft, 1, false); let o = e.buffer(ot, 1, true);
            let s = e.scalar("f"); let b = e.scalar("f");
            rudnn::standalone::sol::common::affine_scalar::expand::<$f, $o>(&mut e.kernel.body, x.into(), o.into(), s.into(), b.into(), ft.storage_type(), ot.storage_type());
            e.save(d, ix);
            let mut e = Export::new(&format!("reduce_last_{}", $tag), 256, 1);
            let x = e.buffer(ft, 1, false); let o = e.buffer(ot, 1, true);
            let n = e.scalar("I"); let s = e.scalar("f");
            rudnn::standalone::sol::common::reduce_last::expand::<$f, $o>(&mut e.kernel.body, x.into(), o.into(), n.into(), s.into(), ft.storage_type(), ot.storage_type());
            e.save(d, ix);
        }};
    }
    arithmetic!("ff", f32, f32, FloatKind::F32, FloatKind::F32);
    arithmetic!("bb", half::bf16, half::bf16, FloatKind::BF16, FloatKind::BF16);
    arithmetic!("bf", half::bf16, f32, FloatKind::BF16, FloatKind::F32);
    arithmetic!("fb", f32, half::bf16, FloatKind::F32, FloatKind::BF16);
    for (tag, fp) in [("f", Type::from(FloatKind::F32)), ("b", Type::from(FloatKind::BF16))] {
        for trans in [false, true] {
            let mut e = Export::new(&format!("spatial_columns_{tag}_{}", trans as u8), 256, 1);
            let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
            let v: Vec<_> = (0..17).map(|_| e.scalar("I")).collect();
            if tag == "f" {
                spatial::columns::expand::<f32>(&mut e.kernel.body, x.into(), o.into(), v[0].clone().into(), v[1].clone().into(), v[2].clone().into(), v[3].clone().into(), v[4].clone().into(), v[5].clone().into(), v[6].clone().into(), v[7].clone().into(), v[8].clone().into(), v[9].clone().into(), v[10].clone().into(), v[11].clone().into(), v[12].clone().into(), v[13].clone().into(), v[14].clone().into(), v[15].clone().into(), v[16].clone().into(), trans, fp.storage_type());
            } else {
                spatial::columns::expand::<half::bf16>(&mut e.kernel.body, x.into(), o.into(), v[0].clone().into(), v[1].clone().into(), v[2].clone().into(), v[3].clone().into(), v[4].clone().into(), v[5].clone().into(), v[6].clone().into(), v[7].clone().into(), v[8].clone().into(), v[9].clone().into(), v[10].clone().into(), v[11].clone().into(), v[12].clone().into(), v[13].clone().into(), v[14].clone().into(), v[15].clone().into(), v[16].clone().into(), trans, fp.storage_type());
            }
            e.save(d, ix);
        }
        let mut e = Export::new(&format!("matrix_pad_{tag}"), 256, 1);
        let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
        let r = e.scalar("I"); let c = e.scalar("I"); let p = e.scalar("I");
        if tag == "f" { spatial::matrix_pad::expand::<f32>(&mut e.kernel.body, x.into(), o.into(), r.into(), c.into(), p.into(), fp.storage_type()); }
        else { spatial::matrix_pad::expand::<half::bf16>(&mut e.kernel.body, x.into(), o.into(), r.into(), c.into(), p.into(), fp.storage_type()); }
        e.save(d, ix);
    }
    for (tag, ft) in [("f32", Type::from(FloatKind::F32)), ("bf16", Type::from(FloatKind::BF16))] {
        let mut e = Export::new(&format!("depthwise1d_{tag}"), 256, 1);
        let x = e.buffer(ft, 1, false); let w = e.buffer(ft, 1, false); let b = e.buffer(ft, 1, false); let o = e.buffer(ft, 1, true);
        let v: Vec<_> = (0..5).map(|_| e.scalar("I")).collect();
        if tag == "f32" { spatial::depthwise1d::expand::<f32>(&mut e.kernel.body, x.into(), w.into(), b.into(), o.into(), v[0].clone().into(), v[1].clone().into(), v[2].clone().into(), v[3].clone().into(), v[4].clone().into(), ft.storage_type()); }
        else { spatial::depthwise1d::expand::<half::bf16>(&mut e.kernel.body, x.into(), w.into(), b.into(), o.into(), v[0].clone().into(), v[1].clone().into(), v[2].clone().into(), v[3].clone().into(), v[4].clone().into(), ft.storage_type()); }
        e.save(d, ix);
    }
}
