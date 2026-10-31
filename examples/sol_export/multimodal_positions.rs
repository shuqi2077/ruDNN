use std::path::PathBuf;
use ruda_core::ir::{FloatKind, IntKind, Type};
use rudnn::standalone::sol::multimodal_positions;
use super::Export;
pub fn export(d: &PathBuf, ix: &mut String) {
    let long = Type::from(IntKind::I64);
    let mut e = Export::new("mrope_compact", 32, 1);
    let ids = e.buffer(long, 1, false); let mask = e.buffer(long, 1, false); let tokens = e.buffer(long, 1, true);
    let ranks = e.buffer(long, 1, true); let counts = e.buffer(long, 1, true); let n = e.scalar("I");
    multimodal_positions::compact::expand(&mut e.kernel.body, ids.into(), mask.into(), tokens.into(), ranks.into(), counts.into(), n.into()); e.save(d, ix);
    let mut e = Export::new("mrope_positions", 256, 1);
    let tokens = e.buffer(long, 1, false); let ranks = e.buffer(long, 1, false); let counts = e.buffer(long, 1, false);
    let images = e.buffer(long, 1, false); let videos = e.buffer(long, 1, false); let seconds = e.buffer(Type::from(FloatKind::F32), 1, false);
    let o = e.buffer(long, 1, true); let delta = e.buffer(long, 1, true); let n = e.scalar("I");
    multimodal_positions::positions::expand(&mut e.kernel.body, tokens.into(), ranks.into(), counts.into(), images.into(), videos.into(), seconds.into(), o.into(), delta.into(), n.into()); e.save(d, ix);
}
