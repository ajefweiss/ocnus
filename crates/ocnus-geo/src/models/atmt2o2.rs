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

/// Calculate surface gravity as a function of latitude.
fn surface_gravity<T>(lat_rad: T) -> T
where
    T: RealField,
{
    let sin_lat = lat_rad.sin();
    let sin_lat_sq = sin_lat.clone() * sin_lat;

    let g_equator = T::from_f64(9.7803253359).unwrap();
    let k1 = T::from_f64(0.00193185265241).unwrap();
    let k2 = T::from_f64(0.00669437999013).unwrap();

    g_equator * (T::one() + k1 * sin_lat_sq.clone()) / (T::one() - k2 * sin_lat_sq).sqrt()
}

/// Calculate Earth's radius as a function of latitude.
fn earth_radius<T>(lat_rad: T) -> T
where
    T: RealField,
{
    let a = T::from_f64(6.3781370).unwrap();
    let b = T::from_f64(6.356752314245).unwrap();

    let cos_lat = lat_rad.clone().cos();
    let sin_lat = lat_rad.sin();

    let numerator = (a.clone().powi(2) * cos_lat.clone()).powi(2)
        + (b.clone().powi(2) * sin_lat.clone()).powi(2);
    let denominator = (a * cos_lat).powi(2) + (b * sin_lat).powi(2);

    (numerator / denominator).sqrt()
}

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

impl<T, OC, G> Plasma<T, OC, 3, 18> for AtmosphereT2O2Model<T, G>
where
    T: AsPrimitive<usize> + Default + Copy + RealField + SampleUniform + Sum,
    OC: ObsPosition<T, 3>,
    G:  Density<T, Const<18>>,
    for<'a> &'a G: Density<T, Const<18>>,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    fn observe_pbs_ics(
        &self,
        _ics: &nalgebra::SVectorView<T, 3>,
        _params: &nalgebra::SVectorView<T, 18>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        T::zero()
    }

    fn observe_rho_ics(
        &self,
        ics: &nalgebra::SVectorView<T, 3>,
        params: &nalgebra::SVectorView<T, 18>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        let temp = spherical_harmonics!(0, 2, params, ics[2], ics[1]);
        let oxygen = spherical_harmonics!(9, 2, params, ics[2], ics[1]);

        let oxygen_amu = T::from_f64(15.9994).unwrap();

        let g = surface_gravity(ics[2]) / (T::one() + ics[0] / earth_radius(ics[2])).powi(2);

        let scale_height = T::from_f64(8.31446).unwrap() / g / oxygen_amu * temp;

        // Chemical correction term
        let chemical_correction =
            -(ics[0] - T::from_f64(0.475).unwrap()) / T::from_f64(100.5).unwrap();

        oxygen
            * (-(ics[0] - T::from_f64(0.475).unwrap()) / (scale_height + chemical_correction)).exp()
    }

    fn observe_temp_ics(
        &self,
        ics: &nalgebra::SVectorView<T, 3>,
        params: &nalgebra::SVectorView<T, 18>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        let temp = spherical_harmonics!(0, 2, params, ics[2], ics[1]);

        temp * T::from_f64(1000.0).unwrap()
    }
}

impl_atmos_model!(
    AtmosphereT2O2Model,
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
        coeff_name!("o", 0, 0),  // 9
        coeff_name!("o", 1, -1), // 10
        coeff_name!("o", 1, 0),  // 11
        coeff_name!("o", 1, 1),  // 12
        coeff_name!("o", 2, -2), // 13
        coeff_name!("o", 2, -1), // 14
        coeff_name!("o", 2, 0),  // 15
        coeff_name!("o", 2, 1),  // 16
        coeff_name!("o", 2, 2),  // 17
    ],
    18
);
