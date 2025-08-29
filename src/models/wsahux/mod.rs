mod types;

pub use types::*;

use crate::{
    base::{Model, ModelError, ScConf},
    coords::{Coordinates, SPHUGeometry, param_value_static},
    models::concat_strs,
    obsty::{InSituPlasmaBulkVelocity, ObserVec},
    stats::{Density, DensityRange},
};
use nalgebra::{Const, Dim, RealField, SVector, U1, U8, Vector3, VectorView, VectorView3};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

/// Standard WSA model
pub fn wsa_map<T, const D: usize>(
    (efs, dist): (T, T),
    names: &SVector<&'static str, D>,
    params: &SVector<T, D>,
) -> T
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let a1 = param_value_static("a1", names, params).unwrap();
    let a2 = param_value_static("a2", names, params).unwrap();
    let a3 = param_value_static("a3", names, params).unwrap();
    let a4 = param_value_static("a4", names, params).unwrap();
    let a5 = param_value_static("a5", names, params).unwrap();
    let a6 = param_value_static("a6", names, params).unwrap();
    let a7 = param_value_static("a7", names, params).unwrap();
    let a8 = param_value_static("a8", names, params).unwrap();

    a1 + a2 / (T::one() + efs).powf(a3)
        * (a4 - a5 * (-(T::from_f32(180.0).unwrap() * dist / T::pi() / a6).powf(a7)).exp()).powf(a8)
}

/// Standard WSA solar wind model.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WSAHUXModel<T, const R: usize, P>(P, pub WSAInputData<T>, T)
where
    T: Copy + RealField;

impl<T, const R: usize, P> WSAHUXModel<T, R, P>
where
    T: Copy + RealField,
    for<'x> &'x P: Density<T, 8>,
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
    pub fn new(pdf: P, input: WSAInputData<T>, radial_resolution: T) -> Self {
        Self(pdf, input, radial_resolution)
    }

    #[doc = concat!("Create a new [`", stringify!(WSAHUXModel), "`] from a JSON5 file.")]
    pub fn from_file<L>(pdf: P, path: L, radial_resolution: T) -> io::Result<Self>
    where
        T: Copy + for<'x> Deserialize<'x> + RealField,
        L: AsRef<Path>,
    {
        let input = serde_json5::from_str::<WSAInputData<T>>(&fs::read_to_string(path)?)
            .expect("deserialization failed");

        Ok(Self(pdf, input, radial_resolution))
    }
}

impl<T, const R: usize, P> InSituPlasmaBulkVelocity<T, 8> for WSAHUXModel<T, R, P>
where
    T: AsPrimitive<usize> + Default + Copy + RealField,
    for<'x> &'x P: Density<T, 8>,
{
    fn observe_pbv(
        &self,
        scconf: &ScConf<T>,
        params: &SVector<T, 8>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserVec<T, 1>, ModelError<T>> {
        let sc_pos = scconf.position();

        let q = match Self::transform_ecs_to_ics(
            &sc_pos.as_view(),
            &params.generic_view((0, 0), (U8, Const::<1>)),
            cs_state,
        ) {
            Some(value) => value,
            None => {
                return Err(ModelError::CoordinateTransform(sc_pos.into_owned()));
            }
        };

        let r_index: usize = ((q[0] * T::from_usize(R).unwrap() - T::from_f32(2.5).unwrap())
            / self.2)
            .round()
            .as_();

        let lon = T::two_pi() - q[1] - fm_state.angle;

        let lon_index = (lon / (self.1.lon_1d[1] - self.1.lon_1d[0])).round().as_();
        let lat_index = self.latitude_index(T::frac_pi_2() - q[2]);

        Ok(ObserVec::from([
            fm_state.wsahux[lat_index].0[(r_index, lon_index)]
        ]))
    }
}

// Re-implement the Coordinates trait because we have no inheritance.
// Here we make use of the fact that the parameters for the coords are at the front
// and we pass on smaller fixed views of each parameter vector.
impl<T, const R: usize, P> Coordinates<T, 8> for WSAHUXModel<T, R, P>
where
    T: Copy + RealField,
{
    const PARAMS: SVector<&'static str, 8> = SVector::from_array_storage(concat_strs!(
        SPHUGeometry::<f32>::PARAMS,
        ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]
    ));

    type CSST = ();

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<[Vector3<T>; 3]> {
        SPHUGeometry::contravariant_basis(
            ics,
            &params.fixed_rows::<{ SPHUGeometry::<f32>::PARAMS_COUNT }>(0),
            cs_state,
        )
    }

    fn detg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<T> {
        SPHUGeometry::detg(
            ics,
            &params.fixed_rows::<{ SPHUGeometry::<f32>::PARAMS_COUNT }>(0),
            cs_state,
        )
    }

    fn initialize_cs<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &mut Self::CSST,
    ) {
        SPHUGeometry::initialize_cs(
            &params.fixed_rows::<{ SPHUGeometry::<f32>::PARAMS_COUNT }>(0),
            cs_state,
        )
    }

    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        SPHUGeometry::transform_ics_to_ecs(
            ics,
            &params.fixed_rows::<{ SPHUGeometry::<f32>::PARAMS_COUNT }>(0),
            cs_state,
        )
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        SPHUGeometry::transform_ecs_to_ics(
            ecs,
            &params.fixed_rows::<{ SPHUGeometry::<f32>::PARAMS_COUNT }>(0),
            cs_state,
        )
    }
}

impl<T, const R: usize, P> Model<T, 8> for WSAHUXModel<T, R, P>
where
    T: AsPrimitive<usize> + Copy + Default + RealField,
    for<'x> &'x P: Density<T, 8>,
{
    const RCS: usize = 32;

    type FMST = WSAState<T, R>;

    fn forward(
        &self,
        time_step: T,
        _params: &VectorView<T, U8, U1, U8>,
        fm_state: &mut Self::FMST,
        _cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>> {
        fm_state.angle += time_step * T::two_pi() / T::from_f32(27.2753 * 86400.0).unwrap();
        Ok(())
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 8> {
        (&self.0).get_range()
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
                        *value =
                            wsa_map::<T, 8>((*efs, *dmap), &Self::PARAMS, &params.clone_owned());
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
                            let f2 = T::two_pi() / T::from_f32(25.38 * 86400.0).unwrap()
                                * dr
                                * T::from_f32(695700.0).unwrap()
                                / dphi;
                            slice[(r_i + 1, phi_i)] = slice[(r_i, phi_i)] + f1 * f2
                        }
                    }
                }
            });

        Ok(())
    }

    fn model_prior(&self) -> impl Density<T, 8> {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        base::{ModelEnsbl, Obser, ScObs},
        obsty::NullNoise,
        stats::{ConstantDensity, MultivariateDensity},
    };
    use approx::ulps_eq;

    #[test]
    fn test_wsahux() {
        let sc = ScObs::from_iterator((0..100).map(|i| {
            (
                (14400 * i) as f32,
                ScConf::Position(Vector3::new(1.0, 0.0, 0.0)),
            )
        }));

        let prior = MultivariateDensity::<_, 8>::new(&[
            ConstantDensity::new(285.0),     // a1 = 285 # speeed
            ConstantDensity::new(625.0),     // a2 = 625 # speed
            ConstantDensity::new(2.0 / 9.0), // a3 = 2/9 # expansion factor coefficient
            ConstantDensity::new(1.0),       // a4 = 1 # exp offset
            ConstantDensity::new(0.8),       // a5 = 0.8 # exp factor
            ConstantDensity::new(2.0),       // a6 = 2 # distance fraction (DO NOT CHANGE)
            ConstantDensity::new(2.0),       // a7 = 2 # distance factor coefficient
            ConstantDensity::new(3.0),       // a8 = 3 # coefficient everything
        ]);

        let path = Path::new("examples")
            .join("data")
            .join("wsapy_NSO-GONG_CR2047_0_NSteps90.json");

        let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior, path, 2.0).unwrap();

        model.limit_latitude(1.0);

        let mut ensbl = ModelEnsbl::new(1, None);
        let mut obser = Obser::new(sc.clone(), 1);

        model
            .initialize_ensbl(&mut ensbl, None::<&MultivariateDensity<f32, 8>>, 100, 41)
            .unwrap();

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obser,
                &WSAHUXModel::observe_pbv,
                &mut None::<&mut NullNoise<f32>>,
            )
            .unwrap();

        let speed = Vec::<f32>::from_iter(obser.get_output(0).iter().map(|value| value[0]));

        assert!(ulps_eq!(speed[0], 298.2084));
        assert!(ulps_eq!(speed[25], 440.25708));
        assert!(ulps_eq!(speed[50], 442.22626));
        assert!(ulps_eq!(speed[75], 435.19177));
        assert!(ulps_eq!(speed[99], 320.37265));
    }
}
