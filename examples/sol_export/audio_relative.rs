use std::path::PathBuf;
use ruda_core::ir::{FloatKind, UIntKind, Type};
use rudnn::standalone::sol::audio_relative;
use super::Export;
pub fn export(d: &PathBuf, ix: &mut String) {
    let fp = Type::from(FloatKind::F32); let bf = Type::from(FloatKind::BF16);
    let mut e = Export::new("audio_relative_timing", 256, 1);
    let inv = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    audio_relative::timing::expand(&mut e.kernel.body, inv.into(), o.into()); e.save(d, ix);
    let mut e = Export::new("audio_local_logits", 256, 1);
    let a = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false); let mask = e.buffer(Type::from(UIntKind::U8), 1, false); let o = e.buffer(fp, 1, true);
    let n = e.scalar("I"); let blocks = e.scalar("I"); let cap = e.scalar("f");
    audio_relative::local_logits::expand(&mut e.kernel.body, a.into(), b.into(), mask.into(), o.into(), n.into(), blocks.into(), cap.into()); e.save(d, ix);
    let mut e = Export::new("conformer_relative_position", 256, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(bf, 1, true);
    let n = e.scalar("I"); let table = e.scalar("I"); let remainder = e.scalar("I"); let scale = e.scalar("f");
    audio_relative::conformer_position::expand(&mut e.kernel.body, x.into(), o.into(), n.into(), table.into(), remainder.into(), scale.into()); e.save(d, ix);
}
