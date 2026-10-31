use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
fn rescale(value: f32) -> f32 {
    f32::powf(2.0f32, value * core::f32::consts::LOG2_E)
}

#[ruda]
pub(crate) fn merge_warps(acc: &mut Array<f32>, maximum: f32, denominator: f32,
    warp: usize, lane: usize, #[comptime] width: usize, #[comptime] lanes: usize,
    #[comptime] warps: usize,
) -> (f32, f32) {
    let mut merged_maximum = maximum;
    let mut merged_denominator = denominator;
    if comptime!(warps > 1) {
        let mut partials = SharedMemory::<f32>::new(warps * (width + 2));
        let base = warp * (width + 2);
        #[unroll] for j in 0usize..((width + lanes - 1) / lanes) {
            let d = lane + j * lanes;
            if d < width { partials[base + d] = acc[j]; }
        }
        if lane == 0 {
            partials[base + width] = maximum;
            partials[base + width + 1] = denominator;
        }
        sync_ruda();
        if warp == 0 {
            merged_maximum = 0.0;
            let mut nonempty = false;
            #[unroll] for part in 0usize..warps {
                if partials[part * (width + 2) + width + 1] != 0.0 {
                    let part_maximum = partials[part * (width + 2) + width];
                    if nonempty { merged_maximum = f32::max(merged_maximum, part_maximum); }
                    else { merged_maximum = part_maximum; }
                    nonempty = true;
                }
            }
            merged_denominator = 0.0;
            #[unroll] for j in 0usize..((width + lanes - 1) / lanes) { acc[j] = 0.0; }
            #[unroll] for part in 0usize..warps {
                let part_denominator = partials[part * (width + 2) + width + 1];
                let mut factor = 1.0f32;
                if part_denominator != 0.0 {
                    factor = rescale(partials[part * (width + 2) + width] - merged_maximum);
                    merged_denominator = fma(part_denominator, factor, merged_denominator);
                }
                #[unroll] for j in 0usize..((width + lanes - 1) / lanes) {
                    let d = lane + j * lanes;
                    if d < width { acc[j] = fma(partials[part * (width + 2) + d], factor, acc[j]); }
                }
            }
        }
    }
    (merged_maximum, merged_denominator)
}
