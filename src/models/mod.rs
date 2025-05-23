//! Implemented models and forward model state types.

mod corem;
mod cylm;
mod wsahux;

pub use corem::*;
pub use cylm::*;
pub use wsahux::*;

/// Concatenates the &str array `b` with an [`OVector<&str>`] `a` and returns an [`ArrayStorage`].
macro_rules! concat_strs {
    ($a: expr, $b: expr) => {{
        const LEN_A: usize = $a.data.0[0].len();

        let mut c = [$b[0]; LEN_A + $b.len()];

        let mut i1 = 0;
        let mut i2 = 0;

        while i1 < LEN_A {
            c[i1] = $a.data.0[0][i1];

            i1 += 1;
        }

        while i2 < $b.len() {
            c[LEN_A + i2] = $b[i2];

            i2 += 1;
        }

        nalgebra::ArrayStorage([c; 1])
    }};
}

pub(crate) use concat_strs;
