use std::path::PathBuf;
use ruda_core::ir::{FloatKind, Type};
use rudnn::standalone::sol::spectral;
use super::Export;

pub fn export(d: &PathBuf, ix: &mut String) {
    let fp = Type::from(FloatKind::F32);
    let mut e = Export::new("hyena_basis", 256, 1);
    let o = e.buffer(fp, 1, true); let n = e.scalar("I");
    spectral::hyena_basis::expand(&mut e.kernel.body, o.into(), n.into()); e.save(d, ix);
    let mut e = Export::new("hyena_filter", 256, 1);
    let x = e.buffer(fp, 1, false); let delta = e.buffer(fp, 1, false); let b = e.buffer(fp, 1, false); let o = e.buffer(fp, 1, true);
    let n = e.scalar("I"); let shift = e.scalar("f");
    spectral::hyena_filter::expand(&mut e.kernel.body, x.into(), delta.into(), b.into(), o.into(), n.into(), shift.into()); e.save(d, ix);
    let mut e = Export::new("fft_convolution_product", 256, 1);
    let a = e.buffer(fp, 2, false); let b = e.buffer(fp, 2, false); let o = e.buffer(fp, 2, true);
    let c = e.scalar("I"); let n = e.scalar("I");
    spectral::convolution_product::expand(&mut e.kernel.body, a.into(), b.into(), o.into(), c.into(), n.into()); e.save(d, ix);
    for size in [1usize,2,4,8,16,32,64,128,256,512,1024] {
        for first in [false, true] {
            let name = format!("fft_{}_{size}", if first {"first"} else {"last"});
            let mut e = Export::new(&name, 256, 1);
            let x = e.buffer(fp, 2, false); let o = e.buffer(fp, 2, true);
            let n1 = e.scalar("I"); let n2 = e.scalar("I"); let sign = e.scalar("f"); let scale = e.scalar("f");
            spectral::stage::expand(&mut e.kernel.body, x.into(), o.into(), n1.into(), n2.into(), sign.into(), scale.into(), size, first);
            e.save(d, ix);
        }
    }
    let mut e = Export::new("fft_real_pad", 256, 1);
    let x = e.buffer(fp, 1, false); let o = e.buffer(fp, 2, true); let n = e.scalar("I"); let len = e.scalar("I");
    spectral::real_pad::expand(&mut e.kernel.body, x.into(), o.into(), n.into(), len.into());
    e.save(d, ix);
    let mut e = Export::new("fft_chirp_input", 256, 1);
    let x = e.buffer(fp, 2, false); let a = e.buffer(fp, 2, true); let b = e.buffer(fp, 2, true);
    let n = e.scalar("I"); let p = e.scalar("I"); let sign = e.scalar("f");
    spectral::chirp_input::expand(&mut e.kernel.body, x.into(), a.into(), b.into(), n.into(), p.into(), sign.into());
    e.save(d, ix);
    let mut e = Export::new("fft_multiply", 256, 1);
    let a = e.buffer(fp, 2, false); let b = e.buffer(fp, 2, false); let o = e.buffer(fp, 2, true); let p = e.scalar("I");
    spectral::multiply::expand(&mut e.kernel.body, a.into(), b.into(), o.into(), p.into());
    e.save(d, ix);
    let mut e = Export::new("fft_chirp_finish", 256, 1);
    let x = e.buffer(fp, 2, false); let o = e.buffer(fp, 2, true);
    let n = e.scalar("I"); let p = e.scalar("I"); let sign = e.scalar("f"); let scale = e.scalar("f");
    spectral::chirp_finish::expand(&mut e.kernel.body, x.into(), o.into(), n.into(), p.into(), sign.into(), scale.into());
    e.save(d, ix);
    let mut e = Export::new("fft_real_split", 256, 1);
    let x = e.buffer(fp, 2, false); let re = e.buffer(fp, 1, true); let im = e.buffer(fp, 1, true);
    let n = e.scalar("I"); let bins = e.scalar("I"); let scale = e.scalar("f");
    spectral::real_split::expand(&mut e.kernel.body, x.into(), re.into(), im.into(), n.into(), bins.into(), scale.into());
    e.save(d, ix);
}
