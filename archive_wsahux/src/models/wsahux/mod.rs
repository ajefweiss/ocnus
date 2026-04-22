mod types;

pub use types::*;

use nalgebra::{Dim, RealField, SVector, SVectorView, U1, U8, Vector3, VectorView, VectorView3};
use num_traits::AsPrimitive;
use ocnus::{
    base::{Model, ModelError},
    coords::{Coordinates, SPHUGeometry, param_value},
    instr::Plasma,
    model_impl_concat_strs,
    obs::conf::VecConf,
};
use prodef::{Density, domain::Domain};
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

/// Standard WSA model
pub fn wsa_map<T, const D: usize>(
    (efs, dist): (T, T),
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
) -> T
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let a1 = param_value("a1", names, params);
    let a2 = param_value("a2", names, params);
    let a3 = param_value("a3", names, params);
    let a4 = param_value("a4", names, params);
    let a5 = param_value("a5", names, params);
    let a6 = param_value("a6", names, params);
    let a7 = param_value("a7", names, params);
    let a8 = param_value("a8", names, params);

    a1 + a2 / (T::one() + efs).powf(a3)
        * (a4 - a5 * (-(T::from_f64(180.0).unwrap() * dist / T::pi() / a6).powf(a7)).exp()).powf(a8)
}

/// Standard WSA solar wind model.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WSAHUXModel<T, const R: usize, G>(G, pub WSAInputData<T>, T)
where
    T: Copy + RealField;

impl<T, const R: usize, G> WSAHUXModel<T, R, G>
where
    T: Copy + RealField,
{
    #[doc = "Calculates the latitude index from a given latitude angle (in radians)"]
    pub fn latitude_index(&self, value: T) -> usize
    where
        T: AsPrimitive<usize>,
    {
        assert!(
            value <= T::frac_pi_2(),
            "latitude angle must be smaller than +pi"
        );
        assert!(
            value >= -T::frac_pi_2(),
            "latitude angle must be larger than -pi"
        );

        if !self.1.lat_indices.is_empty() {
            self.1
                .lat_indices
                .iter()
                .enumerate()
                .fold((0, T::max_value().unwrap()), |acc, (idx, next)| {
                    if (self.1.lat_1d[*next] - value).abs() < acc.1 {
                        (idx, (self.1.lat_1d[*next] - value).abs())
                    } else {
                        acc
                    }
                })
                .0
        } else {
            self.1
                .lat_1d
                .iter()
                .enumerate()
                .fold((0, T::max_value().unwrap()), |acc, (idx, next)| {
                    if (*next - value).abs() < acc.1 {
                        (idx, (*next - value).abs())
                    } else {
                        acc
                    }
                })
                .0
        }
    }

    #[doc = "Limits the latitude to the +/- of the given value."]
    pub fn limit_latitude(&mut self, max_lat: T) {
        self.1.lat_indices = self
            .1
            .lat_1d
            .iter()
            .enumerate()
            .filter_map(|(idx, lat)| {
                if lat.abs() <= (max_lat * T::pi() / T::from_usize(180).unwrap()).abs() {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
    }

    #[doc = concat!("Create a new [`", stringify!(WSAHUXModel), "`].")]
    pub fn new(pdf: G, input: WSAInputData<T>, radial_resolution: T) -> Self
    where
        G: Density<T, U8>,
    {
        Self(pdf, input, radial_resolution)
    }

    #[doc = concat!("Create a new [`", stringify!(WSAHUXModel), "`] from a JSON5 file.")]
    pub fn from_file<L>(pdf: G, path: L, radial_resolution: T) -> io::Result<Self>
    where
        T: Copy + RealField + for<'de> Deserialize<'de>,
        L: AsRef<Path>,
        G: Density<T, U8>,
    {
        let input = serde_json5::from_str::<WSAInputData<T>>(&fs::read_to_string(path)?)
            .expect("deserialization failed");

        Ok(Self(pdf, input, radial_resolution))
    }
}

impl<T, const R: usize, G> Plasma<T, VecConf<T, 3>, 3, 8> for WSAHUXModel<T, R, G>
where
    T: AsPrimitive<usize> + Default + Copy + RealField,
    G: Density<T, U8>,
    for<'a> &'a G: Density<T, U8>,
{
    fn observe_pbs_ics(
        &self,
        ics: &SVectorView<T, 3>,
        _params: &VectorView<T, U8>,
        fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        let r_index: usize = ((ics[0] * T::from_usize(R).unwrap() - T::from_f64(2.5).unwrap())
            / self.2)
            .round()
            .as_();

        let lon = T::two_pi() - ics[1] - fm_state.angle;

        let lon_index = (lon / (self.1.lon_1d[1] - self.1.lon_1d[0])).round().as_();
        let lat_index = self.latitude_index(T::frac_pi_2() - ics[2]);

        if lat_index >= fm_state.wsahux.len() {
            panic!("Latitude index out of bounds");
        }

        fm_state.wsahux[lat_index].0[(r_index, lon_index)]
    }

    fn observe_rho_ics(
        &self,
        _ics: &VectorView3<T>,
        _params: &VectorView<T, U8>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        unimplemented!("WSA does not support plasma density measurements")
    }

    fn observe_temp_ics(
        &self,
        _ics: &SVectorView<T, 3>,
        _params: &SVectorView<T, 8>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        unimplemented!("WSA does not support plasma temperature measurements")
    }
}

// Re-implement the Coordinates trait because we have no inheritance.
// Here we make use of the fact that the parameters for the coords are at the front
// and we pass on smaller fixed views of each parameter vector.
impl<T, const R: usize, G> BFMGeometry<T, 3, 8> for WSAHUXModel<T, R, G>
where
    T: Copy + RealField,
{
    const PARAM_NAMES: SVector<&'static str, 8> =
        SVector::from_array_storage(model_impl_concat_strs!(
            SPHUGeometry::<f32>::PARAMS,
            ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]
        ));

    type CSST = ();

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<[Vector3<T>; 3]> {
        let v = &params.fixed_rows::<{ SPHUGeometry::<f32>::NPARAMS }>(0);
        SPHUGeometry::contravariant_basis(ics, v, cs_state)
    }

    fn sqrtdetg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<T> {
        SPHUGeometry::sqrtdetg(
            ics,
            &params.fixed_rows::<{ SPHUGeometry::<f32>::NPARAMS }>(0),
            cs_state,
        )
    }

    fn initialize_cs<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &mut Self::CSST,
    ) {
        SPHUGeometry::initialize_cs(
            &params.fixed_rows::<{ SPHUGeometry::<f32>::NPARAMS }>(0),
            cs_state,
        )
    }

    fn transform_internal_to_external<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        SPHUGeometry::transform_internal_to_external(
            ics,
            &params.fixed_rows::<{ SPHUGeometry::<f32>::NPARAMS }>(0),
            cs_state,
        )
    }

    fn transform_external_to_internal<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        SPHUGeometry::transform_external_to_internal(
            ecs,
            &params.fixed_rows::<{ SPHUGeometry::<f32>::NPARAMS }>(0),
            cs_state,
        )
    }
}

impl<T, const R: usize, G> Model<T, 3, 8> for WSAHUXModel<T, R, G>
where
    T: AsPrimitive<usize> + Copy + Default + RealField,
    G: Density<T, U8>,
    for<'a> &'a G: Density<T, U8>,
{
    const RCS: usize = 32;

    type FMST = WSAState<T, R>;

    fn domain(&self) -> impl Domain<T, U8> {
        self.0.domain().clone()
    }

    fn evolve_state(
        &self,
        time_step: T,
        _params: &VectorView<T, U8, U1, U8>,
        fm_state: &mut Self::FMST,
        _cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>> {
        fm_state.angle += time_step * T::two_pi() / T::from_f64(27.2753 * 86400.0).unwrap();
        Ok(())
    }

    fn initialize_states(
        &self,
        params: &VectorView<T, U8>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>> {
        Self::initialize_cs(params, cs_state);

        fm_state.initialize(self.2, &self.1);

        let dr = self.2;
        let dphi = self.1.lon_1d[1] - self.1.lon_1d[0];

        // WSA implementation.
        fm_state
            .wsahux
            .iter_mut()
            .zip(self.1.lat_indices.iter())
            .for_each(|(slice, lat_index)| {
                let efs_slc = self.1.efs.column(*lat_index);
                let dmap_slc = self.1.dmap.column(*lat_index);
                slice
                    .0
                    .row_mut(0)
                    .iter_mut()
                    .zip(efs_slc.iter())
                    .zip(dmap_slc)
                    .for_each(|((value, efs), dmap)| {
                        *value = wsa_map::<T, 8>((*efs, *dmap), &Self::PARAMS, params);
                    });

                // NaN interpolation, we use a for loop as it is guaranteed
                // that NaN values are solitary.
                for i in 0..slice.ncols() {
                    if !slice[(0, i)].is_finite() {
                        if i == 0 {
                            slice[(0, i)] = (slice[(0, slice.ncols() - 1)] + slice[(0, 1)])
                                / T::from_usize(2).unwrap();
                        } else if i == slice.ncols() - 1 {
                            slice[(0, i)] = (slice[(0, slice.ncols() - 2)] + slice[(0, 0)])
                                / T::from_usize(2).unwrap();
                        } else {
                            slice[(0, i)] =
                                (slice[(0, i + 1)] + slice[(0, i - 1)]) / T::from_usize(2).unwrap();
                        }
                    }
                }

                // Apply the 1d heliospheric upwind model to the inviscid burgers equation.
                for r_i in 0..(slice.nrows() - 1) {
                    for phi_i in 0..(slice.ncols()) {
                        // Force periodicity
                        if phi_i == slice.ncols() - 1 {
                            slice[(r_i + 1, phi_i)] = slice[(r_i + 1, 0)]
                        } else {
                            let f1 = (slice[(r_i, phi_i + 1)] - slice[(r_i, phi_i)])
                                / slice[(r_i, phi_i)];
                            let f2 = T::two_pi() / T::from_f64(25.38 * 86400.0).unwrap()
                                * dr
                                * T::from_f64(695700.0).unwrap()
                                / dphi;
                            slice[(r_i + 1, phi_i)] = slice[(r_i, phi_i)] + f1 * f2
                        }
                    }
                }
            });

        Ok(())
    }

    fn prior(&self) -> impl Density<T, U8> {
        self.0.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{Dyn, OMatrix};
    use ocnus::{
        base::ModelEnsbl,
        obs::{Obs, ObsEnsbl, noise::NullNoise},
    };
    use prodef::multivariate::{ConstantDensity, MultivariateDensity};

    #[test]
    fn test_wsahux() {
        let prior = MultivariateDensity::new(SVector::from([
            ConstantDensity::new(285.0).into(),     // a1 = 285 # speeed
            ConstantDensity::new(625.0).into(),     // a2 = 625 # speed
            ConstantDensity::new(2.0 / 9.0).into(), // a3 = 2/9 # expansion factor coefficient
            ConstantDensity::new(1.0).into(),       // a4 = 1 # exp offset
            ConstantDensity::new(0.8).into(),       // a5 = 0.8 # exp factor
            ConstantDensity::new(2.0).into(),       // a6 = 2 # distance fraction (DO NOT CHANGE)
            ConstantDensity::new(2.0).into(),       // a7 = 2 # distance factor coefficient
            ConstantDensity::new(3.0).into(),       // a8 = 3 # coefficient everything
        ]));

        let obs = Obs::from_iter(
            (0..100).map(|i| VecConf::from((14400.0 * i as f32, Vector3::new(1.0, 0.0, 0.0)))),
        );

        let path = Path::new("examples/data").join("nso_gong_CR2047.json");

        let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior, path, 2.0).unwrap();

        model.limit_latitude(1.0);

        let mut input = OMatrix::<f32, U8, Dyn>::zeros(1);
        input.set_column(
            0,
            &SVector::<f32, 8>::from([285.0, 625.0, 2.0 / 9.0, 1.0, 0.8, 2.0, 2.0, 3.0]),
        );

        let mut model_ensbl = ModelEnsbl::new(input, None, None);
        let mut obs_ensbl = ObsEnsbl::new(obs.clone(), 1, None).unwrap();

        model.initialize_states_ensbl(&mut model_ensbl).unwrap();

        model
            .simulate_ensbl(
                &mut model_ensbl,
                &mut obs_ensbl,
                &WSAHUXModel::observe_pbs,
                &mut None::<&mut NullNoise<f32>>,
            )
            .unwrap();

        let speed = Vec::<f32>::from_iter(obs_ensbl.output(0).iter().map(|value| value[0]));

        assert!(ulps_eq!(speed[0], 298.2084));
        assert!(ulps_eq!(speed[25], 440.25708));
        assert!(ulps_eq!(speed[50], 442.22626));
        assert!(ulps_eq!(speed[75], 435.19177));
        assert!(ulps_eq!(speed[99], 320.37265));
    }
}
