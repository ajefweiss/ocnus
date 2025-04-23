mod types;

pub use types::*;

use crate::{
    base::{OcnusModel, OcnusModelError, ScObs, ScObsConf},
    coords::{OcnusCoords, SPHUGeometry, param_value},
    models::concat_strs,
    obser::{MeasureInSituPlasmaBulkVelocity, ObserVec},
    stats::{Density, DensityRange},
};
use nalgebra::{Const, Dim, RealField, SVector, U1, Vector3, VectorView, VectorView3};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
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
    let a1 = param_value("a1", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let a2 = param_value("a2", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let a3 = param_value("a3", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let a4 = param_value("a4", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let a5 = param_value("a5", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let a6 = param_value("a6", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let a7 = param_value("a7", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let a8 = param_value("a8", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();

    a1 + a2 / (T::one() + efs).powf(a3)
        * (a4 - a5 * (-(T::from_f32(180.0).unwrap() * dist / T::pi() / a6).powf(a7)).exp()).powf(a8)
}

macro_rules! impl_wsa_model {
    ($model: ident,  $coords: ident, $params: expr, $fn_map: tt, $docs: literal) => {
        #[doc=$docs]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $model<T, const R: usize, P>(P, pub WSAInputData<T>, (T, T, T))
        where
            T: Copy + RealField;

        impl<T, const R: usize, P> $model<T, R, P>
        where
            T: Copy + RealField,
        {
            #[doc = "Limits the latitude to the given value."]
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
                    .collect::<Vec<usize>>()
            }

            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: P, input: WSAInputData<T>, radial_resolution: T) -> Self {
                let lon_res = input.lon_1d[1] - input.lon_1d[0];
                let lat_res = input.lat_1d[1] - input.lat_1d[0];

                Self(pdf, input, (radial_resolution, lon_res, lat_res))
            }

            #[doc = concat!("Create a new [`", stringify!($model), "`] from a JSON5 file.")]
            pub fn from_file<L>(pdf: P, path: L, radial_resolution: T) -> io::Result<Self>
            where
                T: Copy + RealField + for<'x> Deserialize<'x>,
                L: AsRef<Path>,
            {
                let input = serde_json5::from_str::<WSAInputData<T>>(&fs::read_to_string(path)?)
                    .expect("deserialization failed");

                let lon_res = input.lon_1d[1] - input.lon_1d[0];
                let lat_res = input.lat_1d[1] - input.lat_1d[0];

                Ok(Self(pdf, input, (radial_resolution, lon_res, lat_res)))
            }
        }

        impl<'a, T, const R: usize, P>
            MeasureInSituPlasmaBulkVelocity<
                T,
                { $coords::<f32>::PARAMS_COUNT + $params.len() },
                WSAState<T, R>,
                (),
            > for $model<T, R, P>
        where
            T: AsPrimitive<usize> + Copy + RealField,
            Self: Send + Sync,
        {
            fn observe_pbv(
                &self,
                scobs: &ScObs<T>,
                params: &SVector<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                fm_state: &WSAState<T, R>,
                cs_state: &(),
            ) -> Result<ObserVec<T, 1>, OcnusModelError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Position(r) => *r,
                });

                let q = match Self::transform_ecs_to_ics(
                    &sc_pos.as_view(),
                    &params.generic_view(
                        (0, 0),
                        (
                            Const::<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                            Const::<1>,
                        ),
                    ),
                    cs_state,
                ) {
                    Some(value) => value,
                    None => {
                        return Err(OcnusModelError::CoordinateTransform(sc_pos.into_owned()));
                    }
                };

                let r_index: usize =
                    ((q[0] * T::from_usize(R).unwrap() - T::from_f32(2.5).unwrap()) / self.2.0)
                        .round()
                        .as_();

                let (lon, lat) = (T::two_pi() - q[1] - fm_state.angle, T::frac_pi_2() - q[2]);

                let lon_index = self
                    .1
                    .lon_1d
                    .iter()
                    .enumerate()
                    .fold((0, T::max_value().unwrap()), |acc, (idx, next)| {
                        if (*next - lon).abs() < acc.1 {
                            (idx, (*next - lon).abs())
                        } else {
                            acc
                        }
                    })
                    .0;

                let lat_index = if self.1.lat_indices.len() > 0 {
                    self.1
                        .lat_indices
                        .iter()
                        .enumerate()
                        .fold((0, T::max_value().unwrap()), |acc, (idx, next)| {
                            if ((self.1.lat_1d[*next] - lat).abs() < acc.1)
                                && self.1.lat_indices.contains(&idx.clone())
                            {
                                (idx, (self.1.lat_1d[*next] - lat).abs())
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
                            if (*next - lat).abs() < acc.1 {
                                (idx, (*next - lat).abs())
                            } else {
                                acc
                            }
                        })
                        .0
                };

                Ok(ObserVec::from([
                    fm_state.wsahux[lat_index].0[(r_index, lon_index)]
                ]))
            }
        }

        // Re-implement the OcnusCoords trait because we have no inheritance.
        // Here we make use of the fact that the parameters for the coordinates are at the front
        // and we pass on smaller fixed views of each parameter vector.
        impl<'a, T, const R: usize, P>
            OcnusCoords<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, ()> for $model<T, R, P>
        where
            T: Copy + RealField,
            Self: Send + Sync,
        {
            const PARAMS: SVector<&'static str, { $coords::<f32>::PARAMS_COUNT + $params.len() }> =
                SVector::from_array_storage(concat_strs!($coords::<f32>::PARAMS, $params));

            fn contravariant_basis<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &(),
            ) -> Option<[Vector3<T>; 3]> {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn detg<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &(),
            ) -> Option<T> {
                $coords::detg(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<RStride: Dim, CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &mut (),
            ) {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &(),
            ) -> Option<Vector3<T>> {
                $coords::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &(),
            ) -> Option<Vector3<T>> {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }
        }

        impl<T, const R: usize, P>
            OcnusModel<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, WSAState<T, R>, ()>
            for $model<T, R, P>
        where
            T: AsPrimitive<usize>
                + Copy
                + Default
                + for<'x> Deserialize<'x>
                + RealField
                + SampleUniform
                + Serialize,
            P: for<'x> Deserialize<'x> + Serialize,
            for<'x> &'x P: Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
            Self: OcnusCoords<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, ()>,
        {
            const RCS: usize = 32;

            fn forward(
                &self,
                time_step: T,
                _params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    U1,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                >,
                fm_state: &mut WSAState<T, R>,
                _cs_state: &mut (),
            ) -> Result<(), OcnusModelError<T>> {
                fm_state.angle += time_step * T::two_pi() / T::from_f32(27.2753 * 86400.0).unwrap();

                Ok(())
            }

            fn get_range(
                &self,
            ) -> SVector<DensityRange<T>, { $coords::<f32>::PARAMS_COUNT + $params.len() }> {
                (&self.0).get_range()
            }

            fn initialize_states(
                &self,
                params: &VectorView<T, Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>>,
                fm_state: &mut WSAState<T, R>,
                cs_state: &mut (),
            ) -> Result<(), OcnusModelError<T>> {
                Self::initialize_cs(params, cs_state);

                fm_state.initialize(self.2.0, &self.1);

                let dr = self.2.0;
                let dphi = self.2.1;

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
                                *value = $fn_map::<
                                    T,
                                    { $coords::<f32>::PARAMS_COUNT + $params.len() },
                                >(
                                    (*efs, *dmap), &Self::PARAMS, &params.clone_owned()
                                );
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
                                    slice[(0, i)] = (slice[(0, i + 1)] + slice[(0, i - 1)])
                                        / T::from_usize(2).unwrap();
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

            fn observe_ics_basis(
                &self,
                scobs: &ScObs<T>,
                params: &VectorView<T, Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>>,
                _fm_state: &WSAState<T, R>,
                cs_state: &(),
            ) -> Result<ObserVec<T, 12>, OcnusModelError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Position(r) => *r,
                });

                let q = match Self::transform_ecs_to_ics(&sc_pos.as_view(), params, cs_state) {
                    Some(value) => value,
                    None => {
                        return Err(OcnusModelError::CoordinateTransform(sc_pos.into_owned()));
                    }
                };

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let [e1, e2, e3] = Self::contravariant_basis(
                    &Vector3::from([mu, nu, s]).as_view(),
                    params,
                    cs_state,
                )
                .expect("failed to construct contravariant basis");

                Ok(ObserVec::<T, 12>::from([
                    mu, nu, s, e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
                ]))
            }

            fn model_prior(
                &self,
            ) -> impl Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }> {
                &self.0
            }
        }
    };
}

impl_wsa_model!(
    WSAHUXModel,
    SPHUGeometry,
    ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
    wsa_map,
    "Standard WSA solar wind model."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        base::OcnusEnsbl,
        base::ScObsSeries,
        obser::NullNoise,
        stats::{ConstantDensity, MultivariateDensity},
    };
    use approx::ulps_eq;
    use nalgebra::DMatrix;

    #[test]
    fn test_wsahux() {
        let sc: ScObsSeries<f32> = ScObsSeries::<f32>::from_iterator(
            (0..100).map(|i| ScObs::new((14400 * i) as f32, ScObsConf::Position([1.0, 0.0, 0.0]))),
        );

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

        let range = (&prior).get_range();
        let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior, path, 2.0).unwrap();

        model.limit_latitude(1.0);

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<_, 1>>::zeros(sc.len(), 1);

        model
            .initialize_ensbl::<100, _>(&mut ensbl, None::<&MultivariateDensity<f32, 8>>, 41)
            .unwrap();

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &WSAHUXModel::<f32, 215, MultivariateDensity<f32, 8>>::observe_pbv,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .unwrap();

        let speed = Vec::<f32>::from_iter(output.column(0).iter().map(|value| value[0]));

        assert!(ulps_eq!(speed[0], 298.2084));
        assert!(ulps_eq!(speed[25], 440.25708));
        assert!(ulps_eq!(speed[50], 442.22626));
        assert!(ulps_eq!(speed[75], 435.19177));
        assert!(ulps_eq!(speed[99], 320.37265));
    }
}
