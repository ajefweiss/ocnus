use bayesfm::geometry::{Geometry, param_value, quaternion_rot};
use nalgebra::{
    ArrayStorage, Dim, Matrix3, RealField, SMatrix, SVector, U8, UnitQuaternion, Vector3,
    VectorView, VectorView3,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Analytical GCS axis path.
fn agcs_gamma<T>(s: T, nfac: T, hw: T) -> Vector3<T>
where
    T: RealField,
{
    // Convert axial coordinate s to φ.
    let phi = hw * ((s + s) - T::one());

    Vector3::from([phi.cos(), phi.sin(), T::zero()]) * (T::frac_pi_2() * phi / hw).cos().powf(nfac)
}

/// Analytical GCS (un-scaled) curve speed.
fn _agcs_vs<T>(s: T, nfac: T, hw: T) -> T
where
    T: RealField,
{
    let pis = T::pi() * s;
    let npi = nfac * T::pi();

    pis.sin().powf(nfac - T::one())
        * (npi.powi(2)
            + ((T::from_usize(2).unwrap() * hw).powi(2) - npi.powi(2)) * pis.sin().powi(2))
        .sqrt()
}

/// Analytical GCS (un-scaled) curvature.
fn agcs_curv<T>(s: T, nfac: T, hw: T) -> T
where
    T: RealField,
{
    let pis = T::pi() * s;
    let npi = nfac * T::pi();

    let term_1 = T::from_usize(8).unwrap() * hw.powi(2) * (T::one() + nfac)
        - T::from_usize(2).unwrap() * (nfac - T::one()) * npi.powi(2)
        + (nfac - T::one()).powi(2) * nfac * T::pi().powi(2) / pis.sin().powi(2);

    ((-T::from_usize(4).unwrap() * hw.powi(2) + npi.powi(2)).powi(2)
        + nfac * T::pi().powi(2) / pis.sin().powi(2) * (term_1))
        .sqrt()
        * pis.sin().powf(nfac)
}

/// Analytical GCS t vector.
fn agcs_ts<T>(s: T, nfac: T, hw: T) -> Vector3<T>
where
    T: RealField,
{
    // Convert axial coordinate s to φ.
    let phi = hw * ((s + s) - T::one());
    let pis = T::pi() * s;

    let normalization = ((T::from_usize(2).unwrap() * hw * phi.cos() * pis.sin()
        + nfac * T::pi() * phi.sin() * pis.cos())
    .powi(2)
        + (nfac * T::pi() * pis.cos() * phi.cos()
            - T::from_usize(2).unwrap() * hw * pis.sin() * phi.sin())
        .powi(2))
    .sqrt();

    Vector3::new(
        (nfac * T::pi() * pis.cos() * phi.cos()
            - T::from_usize(2).unwrap() * hw * pis.sin() * phi.sin())
            / normalization,
        (T::from_usize(2).unwrap() * hw * pis.sin() * phi.cos()
            + nfac * T::pi() * pis.cos() * phi.sin())
            / normalization,
        T::zero(),
    )
}

/// Analytical GCS n_1/2 vector.
fn agcs_ns<T>(s: T, nfac: T, hw: T, delta: T) -> (Vector3<T>, Vector3<T>)
where
    T: RealField,
{
    let ts = agcs_ts(s, nfac, hw);
    let n2 = Vector3::new(T::zero(), T::zero(), T::one());
    let n1 = -n2.cross(&ts);

    (n1, n2 * delta)
}

/// Coordinate system state type for the AGCSiED geometry.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AGCSState<T>
where
    T: RealField,
{
    /// Apex distance.
    pub rt: T,

    /// Cross-section radius.
    pub rp: T,

    /// Quaternion for orientation.
    pub q: UnitQuaternion<T>,
}

/// AGCSiED geometry.
pub struct AGCSGeometry<T>(PhantomData<T>)
where
    T: RealField;

impl<T> Default for AGCSGeometry<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> Geometry<T, 3, 8> for AGCSGeometry<T>
where
    T: Default + RealField,
{
    const PARAM_NAMES: SVector<&'static str, 8> = SVector::from_array_storage(ArrayStorage(
        [[
            "rot_z", "rot_y", "rot_x", "r_0rs", "d_1au", "hw", "nfac", "cs_delta",
        ]; 1],
    ));

    type CSST = AGCSState<T>;

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Matrix3<T>> {
        let rt = cs_state.rt;
        let rp = cs_state.rp;
        let hw = param_value("hw", &Self::PARAM_NAMES, params);
        let nfac = param_value("nfac", &Self::PARAM_NAMES, params);
        let delta = param_value("cs_delta", &Self::PARAM_NAMES, params);

        let mu = ics[0];
        let nu = ics[1];
        let s = ics[2];

        let k1 = rt * agcs_curv(s, nfac, hw);
        let pis = T::pi() * s;
        let npi = nfac * T::pi();
        let omega = T::two_pi() * nu;

        let quaternion = cs_state.q;

        let (n1, n2) = agcs_ns(s, nfac, hw, delta);

        let cof = pis.sin().powf(nfac) * rt * rp;

        let dmu = n1 * (cof * omega.cos()) + n2 * (cof * omega.sin());

        let dnu = -n1 * (cof * omega.sin() * T::two_pi() * mu)
            + n2 * (cof * omega.cos() * T::two_pi() * mu);

        // Convert axial coordinate s to φ.
        let phi = hw * ((s + s) - T::one());

        let tvx = Vector3::new(
            npi * pis.cos() * phi.cos() - T::from_usize(2).unwrap() * hw * pis.sin() * phi.sin(),
            T::from_usize(2).unwrap() * hw * pis.sin() * phi.cos() + npi * pis.cos() * phi.sin(),
            T::zero(),
        ) * rt
            * pis.sin().powf(nfac - T::one())
            * (T::one() - mu * rp * pis.sin().powf(nfac) * (omega.cos() * k1));

        let ds = tvx
            + (n1 * omega.cos() + n2 * omega.sin())
                * npi
                * mu
                * rp
                * pis.cos()
                * pis.sin().powf(nfac - T::one());

        Some(SMatrix::from_columns(&[
            quaternion.transform_vector(&dmu),
            quaternion.transform_vector(&dnu),
            quaternion.transform_vector(&ds),
        ]))
    }

    /// Compute the determinant of the metric tensor.
    fn sqrt_detg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<T> {
        let rt = cs_state.rt;
        let rp = cs_state.rp;
        let hw = param_value("hw", &Self::PARAM_NAMES, params);
        let nfac = param_value("nfac", &Self::PARAM_NAMES, params);

        let mu = ics[0];
        let nu = ics[1];
        let s = ics[2];

        let k1 = rt * agcs_curv(s, nfac, hw);
        let pis = T::pi() * s;
        let omega = T::two_pi() * nu;

        Some(
            T::two_pi()
                * mu
                * pis.sin().powf(T::from_usize(2).unwrap() * nfac)
                * rt.powi(2)
                * rp.powi(2)
                * (T::one() - mu * rp * rt * pis.sin().powf(nfac) * (omega.cos() * k1)),
        )
    }

    fn initialize_csst<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &mut Self::CSST,
    ) {
        // Extract parameters using their identifiers.
        let distance_0 =
            param_value("r_0rs", &Self::PARAM_NAMES, params) * T::from_usize(695510).unwrap();
        let diameter_1au = param_value("d_1au", &Self::PARAM_NAMES, params);
        let longitude = param_value("rot_z", &Self::PARAM_NAMES, params);
        let latitude = param_value("rot_y", &Self::PARAM_NAMES, params);
        let inclination = param_value("rot_x", &Self::PARAM_NAMES, params);

        let rt = distance_0 / T::from_f64(1.496e8).unwrap();

        assert!(distance_0 > T::zero(), "initial distance must be positive");
        assert!(diameter_1au > T::zero(), "diameter must be positive");

        cs_state.rp =
            diameter_1au * rt.powf(T::from_f64(1.14).unwrap()) / T::from_usize(2).unwrap();
        cs_state.rt = (rt - cs_state.rp) / T::from_usize(2).unwrap();

        cs_state.q = quaternion_rot(longitude, latitude, inclination);
    }

    fn transform_internal_to_external<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U8, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        let rt = cs_state.rt;
        let rp = cs_state.rp;
        let hw = param_value("hw", &Self::PARAM_NAMES, params);
        let nfac = param_value("nfac", &Self::PARAM_NAMES, params);
        let delta = param_value("cs_delta", &Self::PARAM_NAMES, params);

        let mu = ics[0];
        let nu = ics[1];
        let s = ics[2];

        let _pis: T = T::pi() * s;
        let omega = T::two_pi() * nu;

        let quaternion = cs_state.q;

        let (n1, n2) = agcs_ns(s, nfac, hw, delta);

        let gamma = agcs_gamma(s, nfac, hw);
        let gamma_ratio = gamma.norm() / agcs_gamma(T::from_f64(0.5).unwrap(), nfac, hw).norm();

        // let rv = gamma * rt * T::from_usize(2).unwrap()
        //     + (n1 * omega.cos() + n2 * omega.sin()) * mu * rp * gamma_ratio * pis.sin().powf(nfac);

        let rfac_0 = rp * gamma_ratio; //* pis.sin().powf(nfac);
        let rfac = rfac_0 * (T::one() - (-T::from_f64(4.0).unwrap() * gamma_ratio))
            / (T::one() - (-T::from_f64(4.0).unwrap()));

        let rv = gamma * rt * T::from_usize(2).unwrap()
            + (n1 * omega.cos() + n2 * omega.sin()) * mu * rfac;

        Some(quaternion.transform_vector(&rv))
    }

    fn transform_external_to_internal<RStride: Dim, CStride: Dim>(
        _ecs: &VectorView3<T>,
        _params: &VectorView<T, U8, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use bayesfm::geometry::Geometry3D;
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_fr_coords() {
        let params = SVector::<f64, 8>::from([
            0.0_f64.to_radians(),
            0.0_f64.to_radians(),
            0.0_f64.to_radians(),
            20.0,
            0.1,
            45f64.to_radians(),
            0.4,
            1.0,
        ]);

        let mut cs_state = AGCSState::default();

        AGCSGeometry::initialize_csst(&params.fixed_rows::<8>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.56, 0.17, 0.42);

        let ecs = AGCSGeometry::transform_internal_to_external(
            &ics_ref.as_view(),
            &params.fixed_rows::<8>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec = AGCSGeometry::transform_external_to_internal(
            &ecs.as_view(),
            &params.fixed_rows::<8>(0),
            &cs_state,
        )
        .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        AGCSGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<8>(0), 1e-6);
    }

    #[test]
    fn test_fr_functions() {
        assert!(ulps_eq!(
            agcs_ts(0.5, 0.4, 45.0f64.to_radians()).norm(),
            1.0
        ));

        assert!(ulps_eq!(
            agcs_ns(0.5, 0.4, 45.0f64.to_radians(), 1.0).0.norm(),
            1.0
        ));

        assert!(ulps_eq!(
            agcs_ns(0.5, 0.4, 45.0f64.to_radians(), 1.0).1.norm(),
            1.0
        ));

        assert!(ulps_eq!(
            ((agcs_gamma(0.50001, 0.4, 45.0f64.to_radians())
                - agcs_gamma(0.49999, 0.4, 45.0f64.to_radians()))
                / 0.00002)
                .norm(),
            _agcs_vs(0.5, 0.4, 45.0f64.to_radians()),
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            agcs_curv(0.45, 0.4, 45.0f64.to_radians()),
            6.47263,
            max_ulps = 5,
            epsilon = 1e-5
        ));
    }
}
