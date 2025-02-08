use crate::{
    fevms::{
        ABCParticleFilter, FEVModelError, ForwardEnsembleVectorModel, ModelObserVec,
        ParticleFiltering,
    },
    math::{bessel_jn, quaternion_xyz},
    model::{OcnusModel, OcnusState},
    scobs::ScConf,
    stats::PDF,
    Fp,
};
use nalgebra::{SVectorView, Vector3};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Model state for models with linear propagation.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct LinearModelState {
    /// Time-tracker.
    pub t_i: Fp,

    /// FR position along the x-axis.
    pub x: Fp,

    /// FR offset along the z-axis.
    pub z_init: Fp,
}

impl OcnusState for LinearModelState {}

/// The circular-cylindric coordinate transformation.
pub fn cc_coords<M, const P: usize>(
    (x, y, z): (Fp, Fp, Fp),
    params: &SVectorView<Fp, P>,
    _state: &LinearModelState,
) -> [Vector3<Fp>; 4]
where
    M: OcnusModel<P>,
{
    let radius = M::get_param_value("radius", params).unwrap();
    let y_offset = M::get_param_value("y", params).unwrap();

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute polar coordinates (r, phi).
    let r = (x.powi(2) + z.powi(2)).sqrt() / radius_linearized;
    let phi = z.atan2(x);

    let dphi = Vector3::from([-phi.sin(), 0.0, phi.cos()]);
    let dpsi = Vector3::from([0.0, 1.0, 0.0]);

    [Vector3::new(r, phi, y), Vector3::zeros(), dphi, dpsi]
}

/// The ellliptic-cylindric coordinate transformation (with non-unit basis vectors).
pub fn ec_coords<M, const P: usize>(
    (x, y, z): (Fp, Fp, Fp),
    params: &SVectorView<Fp, P>,
    _state: &LinearModelState,
) -> [Vector3<Fp>; 4]
where
    M: OcnusModel<P>,
{
    let delta = M::get_param_value("delta", params).unwrap();
    let radius = M::get_param_value("radius", params).unwrap();

    // Compute polar coordinates (r, phi).
    let r = (x.powi(2) + z.powi(2)).sqrt() / radius;
    let phi = z.atan2(x / delta);

    let dphi = (x.powi(2) + z.powi(2)).sqrt() * Vector3::from([-delta * phi.sin(), 0.0, phi.cos()]);
    let dpsi = Vector3::from([0.0, 1.0, 0.0]);

    [Vector3::new(r, phi, y), Vector3::zeros(), dphi, dpsi]
}

/// Linear force-free magnetic field observable.
pub fn cc_lff_obs<M, const P: usize>(
    (r, _phi, _psi): (Fp, Fp, Fp),
    params: &SVectorView<Fp, P>,
    _state: &LinearModelState,
) -> Option<[Fp; 3]>
where
    M: OcnusModel<P>,
{
    // Extract parameters using their identifiers.
    let b = M::get_param_value("B", params).unwrap();
    let y_offset = M::get_param_value("y", params).unwrap();
    let alpha_signed = M::get_param_value("alpha", params).unwrap();

    let (alpha, sign) = match alpha_signed.partial_cmp(&0.0) {
        Some(ord) => match ord {
            Ordering::Less => (-alpha_signed, -1.0),
            _ => (alpha_signed, 1.0),
        },
        None => {
            return None;
        }
    };

    match r.partial_cmp(&1.0) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (1.0 - y_offset.powi(2));

                // Bessel function evaluation uses 11 terms.
                let b_s = b_linearized * bessel_jn(alpha * r, 0);
                let b_phi = b_linearized * sign * bessel_jn(alpha * r, 1);

                Some([0.0, b_phi, b_s])
            }
        },
        None => None,
    }
}

/// Uniform twist magnetic field observable.
pub fn cc_ut_obs<M, const P: usize>(
    (r, _phi, _psi): (Fp, Fp, Fp),
    params: &SVectorView<Fp, P>,
    _state: &LinearModelState,
) -> Option<[Fp; 3]>
where
    M: OcnusModel<P>,
{
    // Extract parameters using their identifiers.
    let b = M::get_param_value("B", params).unwrap();
    let y_offset = M::get_param_value("y", params).unwrap();
    let tau = M::get_param_value("tau", params).unwrap();

    match r.partial_cmp(&1.0) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (1.0 - y_offset.powi(2));

                let b_s = b_linearized / (1.0 + tau.powi(2) * (r).powi(2));
                let b_phi = r * b_linearized * tau / (1.0 + tau.powi(2) * (r).powi(2));

                Some([0.0, b_phi, b_s])
            }
        },
        None => None,
    }
}

/// Magnetic field configuration as is used in Nieves-Chinchilla et al. (2018).
pub fn ec_c10_obs<M, const P: usize>(
    (r, _phi, _psi): (Fp, Fp, Fp),
    params: &SVectorView<Fp, P>,
    _state: &LinearModelState,
) -> Option<[Fp; 3]>
where
    M: OcnusModel<P>,
{
    // Extract parameters using their identifiers.
    let b = M::get_param_value("B", params).unwrap();
    let c_10 = M::get_param_value("c10", params).unwrap();
    let delta = M::get_param_value("delta", params).unwrap();
    let tau = M::get_param_value("tau", params).unwrap();

    match r.partial_cmp(&1.0) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_s = b * delta * (tau - r.powi(2));
                let b_phi = -2.0 * b * delta / (delta.powi(2) + 1.0) / c_10 * r;

                Some([0.0, b_phi, b_s])
            }
        },
        None => None,
    }
}

macro_rules! impl_acylm {
    ($model: ident, $params: expr, $fn_coords: tt, $fn_obs: tt, $docs: literal) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        pub struct $model<T>(pub T);

        impl<T> OcnusModel<{ $params.len() }> for $model<T>
        where
            for<'a> &'a T: PDF<{ $params.len() }>,
        {
            const PARAMS: [&'static str; { $params.len() }] = $params;

            fn valid_range(&self) -> [(Fp, Fp); { $params.len() }] {
                (&self.0).valid_range()
            }
        }

        impl<T> ForwardEnsembleVectorModel<LinearModelState, { $params.len() }, 3> for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $params.len() }>,
            Self: OcnusModel<{ $params.len() }>,
        {
            const RCS: usize = 128;

            fn fevm_forward(
                &self,
                time_step: Fp,
                params: &SVectorView<Fp, { $params.len() }>,
                state: &mut LinearModelState,
            ) -> Result<(), FEVModelError> {
                // Extract parameters using their identifiers.
                let vel = Self::get_param_value("v", params).unwrap() / 1.496e8;
                state.t_i += time_step;
                state.x += vel * time_step as Fp;

                Ok(())
            }

            fn fevm_observe(
                &self,
                scconf: &ScConf,
                params: &SVectorView<Fp, { $params.len() }>,
                state: &LinearModelState,
            ) -> Result<Option<ModelObserVec<3>>, FEVModelError> {
                // Extract parameters using their identifiers.
                let phi = Self::get_param_value("phi", params).unwrap();
                let theta = Self::get_param_value("theta", params).unwrap();

                let mut sc_pos = match scconf {
                    ScConf::TimeDistance(_, x) => [*x, 0.0, 0.0],
                    ScConf::TimePosition(_, r) => *r,
                };

                let quaternion = quaternion_xyz(phi, 0.0, theta);

                // Correct for the position for the FR center.
                let x_0 = state.x;
                let z_0 = state.z_init;

                sc_pos[0] -= x_0;
                sc_pos[2] -= z_0;

                let sc_fr_pos = quaternion.conjugate().transform_vector(&sc_pos.into());

                // Compute q (internal) coordinates.
                let [q, _dr, dphi, dpsi] = $fn_coords::<Self, { $params.len() }>(
                    (sc_fr_pos[0], sc_fr_pos[1], sc_fr_pos[2]),
                    params,
                    state,
                );

                let r = q.x;
                let phi = q.y;
                let psi = q.z;

                let obs = $fn_obs::<Self, { $params.len() }>((r, phi, psi), params, state);

                match obs {
                    Some(b_q) => {
                        let b_xyz = dphi * b_q[1] + dpsi * b_q[2];
                        let b_s = quaternion.transform_vector(&b_xyz);
                        Ok(Some(ModelObserVec::<3>::from([b_s[0], b_s[1], b_s[2]])))
                    }
                    None => Ok(None),
                }
            }

            fn fevm_state(
                &self,
                _scconf: &[ScConf],
                params: &SVectorView<Fp, { $params.len() }>,
                state: &mut LinearModelState,
            ) -> Result<(), FEVModelError> {
                // Extract parameters using their identifiers.
                let phi = Self::get_param_value("phi", params).unwrap();
                let theta = Self::get_param_value("theta", params).unwrap();

                let y = Self::get_param_value("y", params).unwrap();
                let radius = Self::get_param_value("radius", params).unwrap();
                let x_init = Self::get_param_value("x_0", params).unwrap();

                state.t_i = 0.0;
                state.x = x_init;

                if !(-1.0..=1.0).contains(&y) {
                    return Err(FEVModelError::InvalidModelParam(("y", y)));
                }

                state.z_init =
                    radius * y * ((1.0 - (phi.sin() * theta.cos()).powi(2)) as Fp).sqrt()
                        / phi.cos()
                        / theta.cos();
                Ok(())
            }

            fn model_prior(&self) -> impl PDF<{ $params.len() }> {
                &self.0
            }
        }

        impl<T> ParticleFiltering<LinearModelState, { $params.len() }, 3> for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $params.len() }>,
        {
        }

        impl<T> ABCParticleFilter<LinearModelState, { $params.len() }, 3> for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $params.len() }>,
        {
        }

        // impl<'a> MVLLParticleFilter<LinearModelState, { $params.len() }, 3> for $model<'a> {}

        // impl<'a> FisherInformation<LinearModelState, { $params.len() }, 3> for $model<'a> {}
    };
}

// Implementation of the classical linear force-free cylindrical flux rope model.
impl_acylm!(
    OCModel_CC_LFF,
    ["phi", "theta", "y", "radius", "v", "B", "alpha", "x_0"],
    cc_coords,
    cc_lff_obs,
    "Classical circular-cylindric linear force-free model."
);

// Implementation of the classical uniform twist cylindrical flux rope model.
impl_acylm!(
    OCModel_CC_UT,
    ["phi", "theta", "y", "radius", "v", "B", "tau", "x_0"],
    cc_coords,
    cc_ut_obs,
    "Classical circular-cylindric uniform twist model."
);

// Implementation of the ellliptic-cylindrical flux rope model as described in Nieves-Chinchilla et al. (2018).
impl_acylm!(
    OCModel_CC_NC18,
    ["phi", "theta", "y", "radius", "delta", "v", "B", "tau", "c10", "x_0"],
    ec_coords,
    ec_c10_obs,
    "Elliptical-cylindrical model as described in Nieves-Chinchilla et al. (2018)."
);

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        fevms::{FEVMEnsbl, ModelObserArray},
        model::OcnusModel,
        scobs::{ScConf, ScObs},
        stats::{ConstantPDF, PUnivariatePDF, UniformPDF, UnivariatePDF},
        Fp,
    };
    use nalgebra::SVector;

    #[test]
    fn test_acylm_lff() {
        let prior_density = PUnivariatePDF::new([
            UnivariatePDF::Uniform(
                UniformPDF::new((-90.0_f32.to_radians(), 90.0_f32.to_radians())).unwrap(),
            ),
            UnivariatePDF::Uniform(
                UniformPDF::new((0.0_f32.to_radians(), 360.0_f32.to_radians())).unwrap(),
            ),
            UnivariatePDF::Uniform(UniformPDF::new((-0.75, 0.75)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((0.1, 0.25)).unwrap()),
            UnivariatePDF::Constant(ConstantPDF::new(1125.0)),
            UnivariatePDF::Uniform(UniformPDF::new((5.0, 25.0)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((-2.4, 2.4)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((0.0, 1.0)).unwrap()),
        ]);

        let model = OCModel_CC_LFF(prior_density);

        let sc = ScObs::<ModelObserVec<3>>::from_iterator((0..8).map(|i| {
            (
                None,
                ScConf::TimeDistance(224640.0 + i as Fp * 3600.0 * 2.0, 1.0),
            )
        }));

        let mut data = FEVMEnsbl::new(1);

        let mut output = ModelObserArray::new(&sc, 1);

        data.ensbl.set_column(
            0,
            &SVector::from([0.0, 0.0, 0.0, 0.2, 600.0, 20.0, 1.0, 0.0]),
        );

        assert!(model
            .fevm_initialize_states_only(sc.as_scconf_slice(), &mut data)
            .is_ok());

        assert!(model
            .fevm_simulate(sc.as_scconf_slice(), &mut data, &mut output)
            .is_ok(),);

        assert!((output[(0, 0)].as_ref().unwrap()[1] - 18.7926).abs() < 1e-4);
        assert!((output[(2, 0)].as_ref().unwrap()[1] - 19.7875).abs() < 1e-4);
        assert!((output[(4, 0)].as_ref().unwrap()[2] + 0.8228).abs() < 1e-4);

        assert!(OCModel_CC_LFF::<PUnivariatePDF<8>>::get_param_index("B").unwrap() == 5);
    }

    #[test]
    fn test_acylm_ut() {
        let prior_density = PUnivariatePDF::new([
            UnivariatePDF::Uniform(
                UniformPDF::new((-90.0_f32.to_radians(), 90.0_f32.to_radians())).unwrap(),
            ),
            UnivariatePDF::Uniform(
                UniformPDF::new((0.0_f32.to_radians(), 360.0_f32.to_radians())).unwrap(),
            ),
            UnivariatePDF::Uniform(UniformPDF::new((-0.75, 0.75)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((0.1, 0.25)).unwrap()),
            UnivariatePDF::Constant(ConstantPDF::new(1125.0)),
            UnivariatePDF::Uniform(UniformPDF::new((5.0, 25.0)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((-5.0, 5.0)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((0.0, 1.0)).unwrap()),
        ]);

        let model = OCModel_CC_UT(prior_density);

        let sc = ScObs::<ModelObserVec<3>>::from_iterator((0..8).map(|i| {
            (
                None,
                ScConf::TimeDistance(224640.0 + i as Fp * 3600.0 * 2.0, 1.0),
            )
        }));

        let mut data = FEVMEnsbl::new(1);

        let mut output = ModelObserArray::new(&sc, 1);

        data.ensbl.set_column(
            0,
            &SVector::from([0.0, 0.0, 0.0, 0.2, 600.0, 20.0, 1.0, 0.0]),
        );

        assert!(model
            .fevm_initialize_states_only(sc.as_scconf_slice(), &mut data)
            .is_ok());

        assert!(model
            .fevm_simulate(sc.as_scconf_slice(), &mut data, &mut output)
            .is_ok(),);

        assert!((output[(0, 0)].as_ref().unwrap()[1] - 16.0615).abs() < 1e-4);
        assert!((output[(2, 0)].as_ref().unwrap()[1] - 19.1827).abs() < 1e-4);
        assert!((output[(4, 0)].as_ref().unwrap()[2] + 1.636).abs() < 1e-4);
    }

    #[test]
    fn test_acylm_nc18() {
        let prior_density = PUnivariatePDF::new([
            UnivariatePDF::Uniform(
                UniformPDF::new((-90.0_f32.to_radians(), 90.0_f32.to_radians())).unwrap(),
            ),
            UnivariatePDF::Uniform(
                UniformPDF::new((0.0_f32.to_radians(), 360.0_f32.to_radians())).unwrap(),
            ),
            UnivariatePDF::Uniform(UniformPDF::new((-0.75, 0.75)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((0.1, 0.25)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((0.2, 1.00)).unwrap()),
            UnivariatePDF::Constant(ConstantPDF::new(1125.0)),
            UnivariatePDF::Uniform(UniformPDF::new((5.0, 25.0)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((1.0, 2.0)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((-5.0, 5.0)).unwrap()),
            UnivariatePDF::Uniform(UniformPDF::new((0.0, 1.0)).unwrap()),
        ]);

        let model = OCModel_CC_NC18(prior_density);

        let sc = ScObs::<ModelObserVec<3>>::from_iterator((0..8).map(|i| {
            (
                None,
                ScConf::TimeDistance(224640.0 + i as Fp * 3600.0 * 2.0, 1.0),
            )
        }));

        let mut data = FEVMEnsbl::new(1);

        let mut output = ModelObserArray::new(&sc, 1);

        data.ensbl.set_column(
            0,
            &SVector::from([0.0, 0.0, 0.0, 0.2, 0.5, 600.0, 20.0, 1.25, 0.5, 0.0]),
        );

        assert!(model
            .fevm_initialize_states_only(sc.as_scconf_slice(), &mut data)
            .is_ok());

        assert!(model
            .fevm_simulate(sc.as_scconf_slice(), &mut data, &mut output)
            .is_ok(),);

        assert!((output[(0, 0)].as_ref().unwrap()[1] - 10.047894).abs() < 1e-4);
        assert!((output[(2, 0)].as_ref().unwrap()[1] - 12.073919).abs() < 1e-4);
        assert!((output[(4, 0)].as_ref().unwrap()[2] - 0.0434046).abs() < 1e-4);

        assert!(OCModel_CC_NC18::<PUnivariatePDF<10>>::get_param_index("B").unwrap() == 6);
    }
}
