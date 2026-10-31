use std::path::PathBuf;
use ruda_core::ir::{FloatKind, Type};
use rudnn::standalone::sol::scan_blocks;
use super::Export;
pub fn export(d: &PathBuf, ix: &mut String) {
    let fp = Type::from(FloatKind::F32);
    let mut e = Export::new("segment_exp_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let p = e.buffer(fp, 1, true); let o = e.buffer(fp, 1, true); let n = e.scalar("I");
    scan_blocks::segment_exp::expand(&mut e.kernel.body, x.into(), p.into(), o.into(), n.into()); e.save(d, ix);
    for strict in [false, true] {
        let mut e = Export::new(&format!("delta_decay_{}", strict as u8), 256, 1);
        let x = e.buffer(fp, 1, false); let p = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
        scan_blocks::delta_decay::expand(&mut e.kernel.body, x.into(), p.into(), o.into(), strict); e.save(d, ix);
    }
    let mut e = Export::new("delta_inverse_f32", 256, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    scan_blocks::delta_inverse::expand(&mut e.kernel.body, x.into(), o.into()); e.save(d, ix);
}
