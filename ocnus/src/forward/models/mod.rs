mod corem;
mod cylm;
mod wsahux;

pub use corem::*;
pub use cylm::*;
pub use wsahux::*;

macro_rules! concat_arrays {
    ($a: expr, $b: expr) => {{
        let mut c = [$b[0]; $a.len() + $b.len()];

        let mut i1 = 0;
        let mut i2 = 0;

        while i1 < $a.len() {
            c[i1] = $a[i1];

            i1 += 1;
        }

        while i2 < $b.len() {
            c[$a.len() + i2] = $b[i2];

            i2 += 1;
        }

        c
    }};
}

pub(crate) use concat_arrays;
