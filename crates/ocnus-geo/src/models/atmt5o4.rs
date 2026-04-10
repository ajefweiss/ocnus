use std::iter::Sum;

use nalgebra::{Const, Dim, RealField, SVector, U1, Vector3, VectorView, VectorView3};
use num_traits::AsPrimitive;
use ocnus::{
    base::{Model, ModelError},
    coords::{CartesianGeometry, Coordinates},
    instr::Plasma,
    math::sph_yml,
    model_impl_concat_strs,
    obs::conf::ObsPosition,
};
use prodef::{Density, domain::Domain};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::models::{impl_atmos_model, spherical_harmonics};

macro_rules! coeff_name {
    ($name: literal, $l: expr, $m: expr) => {
        if $m == 0 {
            concat!($name, "_y", stringify!($l), stringify!($m))
        } else if $m > 0 {
            concat!($name, "_y", stringify!($l), "+", stringify!($m))
        } else {
            concat!($name, "_y", stringify!($l), stringify!($m))
        }
    };
}

impl<T, OC, G> Plasma<T, OC, 3, 53> for AtmosphereT5O4Model<T, G>
where
    T: AsPrimitive<usize> + Default + Copy + RealField + SampleUniform + Sum,
    OC: ObsPosition<T, 3>,
    G: 'static + Density<T, Const<53>>,
    for<'a> &'a G: Density<T, Const<53>>,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    fn observe_pbs_ics(
        &self,
        _ics: &nalgebra::SVectorView<T, 3>,
        _params: &nalgebra::SVectorView<T, 53>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        T::zero()
    }

    fn observe_rho_ics(
        &self,
        ics: &nalgebra::SVectorView<T, 3>,
        params: &nalgebra::SVectorView<T, 53>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        let temp = spherical_harmonics!(0, 5, params, ics[2], ics[1]);
        let oxygen = spherical_harmonics!(30, 4, params, ics[2], ics[1]);

        let oxygen_amu = T::from_f64(15.9994).unwrap();

        let g = T::from_f64(9.80665).unwrap();

        let scale_height = T::from_f64(8.31446).unwrap() / g / oxygen_amu * temp;

        oxygen * (-(ics[0] - T::from_f64(0.5).unwrap()) / (scale_height)).exp()
    }

    fn observe_temp_ics(
        &self,
        ics: &nalgebra::SVectorView<T, 3>,
        params: &nalgebra::SVectorView<T, 53>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        let temp = spherical_harmonics!(0, 2, params, ics[2], ics[1]);

        temp * T::from_f64(1000.0).unwrap()
    }
}

impl_atmos_model!(
    AtmosphereT5O4Model,
    [
        coeff_name!("t", 0, 0),  // 0
        coeff_name!("t", 1, -1), // 1
        coeff_name!("t", 1, 0),  // 2
        coeff_name!("t", 1, 1),  // 3
        coeff_name!("t", 2, -2), // 4
        coeff_name!("t", 2, -1), // 5
        coeff_name!("t", 2, 0),  // 6
        coeff_name!("t", 2, 1),  // 7
        coeff_name!("t", 2, 2),  // 8
        coeff_name!("t", 3, -3), // 9
        coeff_name!("t", 3, -2), // 10
        coeff_name!("t", 3, -1), // 11
        coeff_name!("t", 3, 0),  // 12
        coeff_name!("t", 3, 1),  // 13
        coeff_name!("t", 3, 2),  // 14
        coeff_name!("t", 3, 3),  // 15
        // coeff_name!("t", 4, -4), // 16
        coeff_name!("t", 4, -3), // 17
        coeff_name!("t", 4, -2), // 53
        coeff_name!("t", 4, -1), // 19
        coeff_name!("t", 4, 0),  // 20
        coeff_name!("t", 4, 1),  // 21
        coeff_name!("t", 4, 2),  // 22
        coeff_name!("t", 4, 3),  // 23
        // coeff_name!("t", 4, 4),  // 24
        // coeff_name!("t", 5, -5), // 25
        // coeff_name!("t", 5, -4), // 26
        coeff_name!("t", 5, -3), // 27
        coeff_name!("t", 5, -2), // 28
        coeff_name!("t", 5, -1), // 29
        coeff_name!("t", 5, 0),  // 30
        coeff_name!("t", 5, 1),  // 31
        coeff_name!("t", 5, 2),  // 32
        coeff_name!("t", 5, 3),  // 33
        // coeff_name!("t", 5, 4),  // 34
        // coeff_name!("t", 5, 5),  // 35
        coeff_name!("o", 0, 0),  // 36
        coeff_name!("o", 1, -1), // 37
        coeff_name!("o", 1, 0),  // 38
        coeff_name!("o", 1, 1),  // 39
        coeff_name!("o", 2, -2), // 40
        coeff_name!("o", 2, -1), // 41
        coeff_name!("o", 2, 0),  // 42
        coeff_name!("o", 2, 1),  // 43
        coeff_name!("o", 2, 2),  // 44
        coeff_name!("o", 3, -3), // 45
        coeff_name!("o", 3, -2), // 46
        coeff_name!("o", 3, -1), // 47
        coeff_name!("o", 3, 0),  // 48
        coeff_name!("o", 3, 1),  // 49
        coeff_name!("o", 3, 2),  // 50
        coeff_name!("o", 3, 3),  // 51
        // coeff_name!("o", 4, -4), // 53
        coeff_name!("o", 4, -3), // 53
        coeff_name!("o", 4, -2), // 54
        coeff_name!("o", 4, -1), // 53
        coeff_name!("o", 4, 0),  // 56
        coeff_name!("o", 4, 1),  // 57
        coeff_name!("o", 4, 2),  // 58
        coeff_name!("o", 4, 3),  // 59
                                 // coeff_name!("o", 4, 4),  // 60
    ],
    53
);
