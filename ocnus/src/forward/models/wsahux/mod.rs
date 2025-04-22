mod types;

pub use types::*;

use crate::{
    OcnusError,
    coords::{CNSPHGeometry, CoordsError, OcnusCoords},
    fXX,
    forward::{
        FSMError, FisherInformation, OcnusFSM,
        filters::{ABCParticleFilter, ParticleFilter, SIRParticleFilter},
        models::concat_arrays,
    },
    math::{T, abs, exp, powf},
    obser::{ObserVec, ScObs, ScObsConf, ScObsSeries},
    prodef::OcnusProDeF,
};
use nalgebra::{Const, Dim, SVectorView, U1, Vector3, VectorView, VectorView3};
use num_traits::Float;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::Deserialize;
use std::{fs, io, path::Path};

/// Standard WSA model
pub fn wsa_map<T, const P: usize, M>((efs, dist): (T, T), params: &SVectorView<T, P>) -> T
where
    T: fXX,
    M: OcnusCoords<T, P, ()>,
{
    // Extract parameters using their identifiers.
    let a1 = M::param_value("a1", params).unwrap();
    let a2 = M::param_value("a2", params).unwrap();
    let a3 = M::param_value("a3", params).unwrap();
    let a4 = M::param_value("a4", params).unwrap();
    let a5 = M::param_value("a5", params).unwrap();
    let a6 = M::param_value("a6", params).unwrap();
    let a7 = M::param_value("a7", params).unwrap();
    let a8 = M::param_value("a8", params).unwrap();

    a1 + a2 / powf!(T::one() + efs, a3)
        * powf!(
            a4 - a5 * exp!(-powf!(T!(180.0) * dist / T::pi() / a6, a7)),
            a8
        )
}

macro_rules! impl_wsa_model {
    ($model: ident,  $coords: ident, $params: expr, $fn_map: tt, $docs: literal) => {
        #[doc=$docs]
        #[derive(Debug)]
        pub struct $model<T, const R: usize, D>(D, pub WSAInputData<T>, pub (T, T, T))
        where
            T: fXX,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>;

        impl<T, const R: usize, D> $model<T, R, D>
        where
            T: fXX,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
            #[doc = "Limits the latitude to the given value."]
            pub fn limit_latitude(&mut self, max_lat: T) {
                self.1.lat_indices = self
                    .1
                    .lat_1d
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, lat)| {
                        if abs!(*lat) <= abs!(max_lat.to_radians()) {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<usize>>()
            }

            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: D, data: WSAInputData<T>, radial_resolution: T) -> Self {
                let lon_res = data.lon_1d[1] - data.lon_1d[0];
                let lat_res = data.lat_1d[1] - data.lat_1d[0];

                Self(pdf, data, (radial_resolution, lon_res, lat_res))
            }

            #[doc = concat!("Create a new [`", stringify!($model), "`] from a JSON5 file.")]
            pub fn from_file<P>(pdf: D, path: P, radial_resolution: T) -> io::Result<Self>
            where
                T: fXX + for<'x> Deserialize<'x>,
                P: AsRef<Path>,
            {
                let data = serde_json5::from_str::<WSAInputData<T>>(&fs::read_to_string(path)?)
                    .expect("deserialization failed");

                let lon_res = data.lon_1d[1] - data.lon_1d[0];
                let lat_res = data.lat_1d[1] - data.lat_1d[0];

                Ok(Self(pdf, data, (radial_resolution, lon_res, lat_res)))
            }
        }

        // Re-implement the OcnusCoords trait because we have no inheritance.
        // Here we make use of the fact that the parameters for the coordinates are at the front
        // and we pass on smaller fixed views of each parameter vector.
        impl<T, const R: usize, D>
            OcnusCoords<T, { $coords::<f64>::PARAMS.len() + $params.len() }, ()> for $model<T, R, D>
        where
            T: fXX,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
            const PARAMS: [&'static str; { $coords::<f64>::PARAMS.len() + $params.len() }] =
                concat_arrays!($coords::<f64>::PARAMS, $params);

            fn contravariant_basis<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                cs_state: &(),
            ) -> Result<[Vector3<T>; 3], CoordsError> {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }

            fn detg<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                cs_state: &(),
            ) -> Result<T, CoordsError> {
                $coords::detg(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                cs_state: &mut (),
            ) -> Result<(), CoordsError> {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }

            fn transform_ics_to_ecs<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                cs_state: &(),
            ) -> Result<Vector3<T>, CoordsError> {
                $coords::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }

            fn transform_ecs_to_ics<CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                cs_state: &(),
            ) -> Result<Vector3<T>, CoordsError> {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }
        }

        impl<T, const R: usize, D>
            OcnusFSM<
                T,
                ObserVec<T, 1>,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                WSAState<T, R>,
                (),
            > for $model<T, R, D>
        where
            T: fXX,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
            Self: OcnusCoords<T, { $coords::<f64>::PARAMS.len() + $params.len() }, ()>,
        {
            const RCS: usize = 128;

            fn fsm_forward(
                &self,
                time_step: T,
                _params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                fm_state: &mut WSAState<T, R>,
                _cs_state: &mut (),
            ) -> Result<(), FSMError<T>> {
                fm_state.angle += time_step * T::two_pi() / T!(27.2753 * 86400.0);

                Ok(())
            }

            fn fsm_initialize_states(
                &self,
                _series: &ScObsSeries<T, ObserVec<T, 1>>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                fm_state: &mut WSAState<T, R>,
                cs_state: &mut (),
            ) -> Result<(), OcnusError<T>> {
                // Extract parameters using their identifiers.
                let dr = self.2.0;
                let dphi = self.2.1;

                Self::initialize_cs(params, cs_state)?;

                fm_state.initialize(dr, &self.1);

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
                                    { $coords::<f64>::PARAMS.len() + $params.len() },
                                    Self,
                                >((*efs, *dmap), params);
                            });

                        // NaN interpolation, we use a for loop as it is guaranteed
                        // that NaN values are solitary.
                        for i in 0..slice.ncols() {
                            if slice[(0, i)].is_nan() {
                                if i == 0 {
                                    slice[(0, i)] =
                                        (slice[(0, slice.ncols() - 1)] + slice[(0, 1)]) / T!(2.0);
                                } else if i == slice.ncols() - 1 {
                                    slice[(0, i)] =
                                        (slice[(0, slice.ncols() - 2)] + slice[(0, 0)]) / T!(2.0);
                                } else {
                                    slice[(0, i)] =
                                        (slice[(0, i + 1)] + slice[(0, i - 1)]) / T!(2.0);
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

                                    let f2 = T::two_pi() / T!(25.38 * 86400.0) * dr * T!(695700.0)
                                        / dphi;

                                    slice[(r_i + 1, phi_i)] = slice[(r_i, phi_i)] + f1 * f2
                                }
                            }
                        }
                    });

                Ok(())
            }

            fn fsm_observe(
                &self,
                scobs: &ScObs<T, ObserVec<T, 1>>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                fm_state: &WSAState<T, R>,
                cs_state: &(),
            ) -> Result<ObserVec<T, 1>, OcnusError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, T::zero(), T::zero()],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::transform_ecs_to_ics(&sc_pos.as_view(), params, cs_state)?;

                let r_index: usize = Float::round((q[0] * T!(R as f64) - T!(2.5)) / self.2.0).as_();

                let (lon, lat) = (T::two_pi() - q[1] - fm_state.angle, T::half_pi() - q[2]);

                let lon_index = self
                    .1
                    .lon_1d
                    .iter()
                    .enumerate()
                    .fold((0, Float::max_value()), |acc, (idx, next)| {
                        if abs!(*next - lon) < acc.1 {
                            (idx, abs!(*next - lon))
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
                        .fold((0, Float::max_value()), |acc, (idx, next)| {
                            if (abs!(self.1.lat_1d[*next] - lat) < acc.1)
                                && self.1.lat_indices.contains(&idx.clone())
                            {
                                (idx, abs!(self.1.lat_1d[*next] - lat))
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
                        .fold((0, Float::max_value()), |acc, (idx, next)| {
                            if abs!(*next - lat) < acc.1 {
                                (idx, abs!(*next - lat))
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

            fn fsm_observe_ics_plus_basis(
                &self,
                scobs: &ScObs<T, ObserVec<T, 1>>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fm_state: &WSAState<T, R>,
                cs_state: &(),
            ) -> Result<ObserVec<T, 12>, OcnusError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, T::zero(), T::zero()],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::transform_ecs_to_ics(&sc_pos.as_view(), params, cs_state)?;

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let [e1, e2, e3] = Self::contravariant_basis(
                    &Vector3::from([mu, nu, s]).as_view(),
                    params,
                    cs_state,
                )?;

                Ok(ObserVec::<T, 12>::from([
                    mu, nu, s, e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
                ]))
            }

            fn param_step_sizes(&self) -> [T; { $coords::<f64>::PARAMS.len() + $params.len() }] {
                (&self.0)
                    .get_valid_range()
                    .iter()
                    .map(|(min, max)| (*max - *min) * T!(128.0) * T::epsilon())
                    .collect::<Vec<T>>()
                    .try_into()
                    .unwrap()
            }

            fn model_prior(
                &self,
            ) -> impl OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }> {
                &self.0
            }
        }

        impl<T, const R: usize, D>
            ParticleFilter<
                T,
                ObserVec<T, 1>,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                WSAState<T, R>,
                (),
            > for $model<T, R, D>
        where
            T: fXX,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, const R: usize, D>
            ABCParticleFilter<
                T,
                ObserVec<T, 1>,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                WSAState<T, R>,
                (),
            > for $model<T, R, D>
        where
            T: fXX + SampleUniform,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, const R: usize, D>
            SIRParticleFilter<
                T,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                1,
                WSAState<T, R>,
                (),
            > for $model<T, R, D>
        where
            T: fXX + SampleUniform,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, const R: usize, D>
            FisherInformation<
                T,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                1,
                WSAState<T, R>,
                (),
            > for $model<T, R, D>
        where
            T: fXX,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }
    };
}

impl_wsa_model!(
    WSAHUXModel,
    CNSPHGeometry,
    ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
    wsa_map,
    "Standard WSA solar wind model."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        forward::FSMEnsbl,
        obser::NoNoise,
        prodef::{Constant1D, UnivariateND},
    };
    use nalgebra::DMatrix;

    #[test]
    fn test_wsahux() {
        let sc: ScObsSeries<f32, ObserVec<f32, 1>> =
            ScObsSeries::<f32, ObserVec<f32, 1>>::from_iterator((0..100).map(|i| {
                ScObs::new(
                    (14400 * i) as f32,
                    ScObsConf::Position([1.0, 0.0, 0.0]),
                    None,
                )
            }));

        let prior = UnivariateND::new([
            Constant1D::new(285.0),     // a1 = 285 # speeed
            Constant1D::new(625.0),     // a2 = 625 # speed
            Constant1D::new(2.0 / 9.0), // a3 = 2/9 # expansion factor coefficient
            Constant1D::new(1.0),       // a4 = 1 # exp offset
            Constant1D::new(0.8),       // a5 = 0.8 # exp factor
            Constant1D::new(2.0),       // a6 = 2 # distance fraction (DO NOT CHANGE)
            Constant1D::new(2.0),       // a7 = 2 # distance factor coefficient
            Constant1D::new(3.0),       // a8 = 3 # coefficient everything
        ]);

        let path = Path::new("examples")
            .join("data")
            .join("wsapy_NSO-GONG_CR2047_0_NSteps90.json");

        let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior, path, 2.0).unwrap();

        model.limit_latitude(1.0);

        let mut ensbl = FSMEnsbl::new(1);
        let mut output = DMatrix::<ObserVec<_, 1>>::zeros(sc.len(), 1);

        model
            .fsm_initialize_ensbl(&sc, &mut ensbl, None::<&UnivariateND<f32, 8>>, 41)
            .unwrap();

        model
            .fsm_simulate_ensbl(
                &sc,
                &mut ensbl,
                &mut output.as_view_mut(),
                None::<&mut NoNoise<f32>>,
            )
            .unwrap();

        let speed = Vec::<f32>::from_iter(output.column(0).iter().map(|v| v[0] as f32));

        assert!(abs!(speed[0] - 298.2084) < 1e-4);
        assert!(abs!(speed[25] - 440.25708) < 1e-4);
        assert!(abs!(speed[50] - 442.22626) < 1e-4);
        assert!(abs!(speed[75] - 435.19177) < 1e-4);
        assert!(abs!(speed[99] - 320.37265) < 1e-4);
    }
}
