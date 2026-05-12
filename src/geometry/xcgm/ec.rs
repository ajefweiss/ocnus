use crate::geometry::xcgm::{XCState, impl_xcgm_geometry};
use bayesfm::geometry::{param_value, param_value_or_else, quaternion_rot};
use nalgebra::{Const, Dim, RealField, SVector, Vector3, VectorView};
use std::marker::PhantomData;

/// The elliptic-cylindric contravariant basis vectors.
#[allow(clippy::extra_unused_type_parameters)]
pub fn ec_basis<T, const D: usize, RStride: Dim, CStride: Dim>(
    (mu, nu, _s): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    T: RealField,
{
    let delta = param_value("cs_delta", names, params);
    let radius = param_value("radius", names, params);

    let omega = T::two_pi() * nu;
    let com = omega.clone().cos();
    let som = omega.clone().sin();

    let denom = (omega.clone().cos().powi(2) + (delta.clone() * omega.sin()).powi(2)).sqrt();

    let nu_nom = T::two_pi() * delta.clone() * mu * radius.clone();

    let dmu = Vector3::from([
        delta.clone() * radius.clone() * com.clone() / denom.clone(),
        T::zero(),
        delta.clone() * radius * som.clone() / denom.clone(),
    ]);

    let dnu = Vector3::from([
        -nu_nom.clone() * delta.powi(2) * som / denom.clone().powi(3),
        T::zero(),
        nu_nom * com / denom.powi(3),
    ]);

    let ds = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dmu, dnu, ds]
}

/// The elliptic-cylindric metric determinant
#[allow(clippy::extra_unused_type_parameters)]
pub fn ec_sqrtdetg<T, const D: usize, RStride: Dim, CStride: Dim>(
    (mu, _nu, _s): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> T
where
    T: RealField,
{
    let delta = param_value("cs_delta", names, params);
    let radius = param_value("radius", names, params);

    T::two_pi() * (mu * delta.powi(2) * radius.powi(2))
}

/// The elliptic-cylindric coordinate transformation (ecs -> ics).
pub fn ec_ecs_to_ics<T, const D: usize, RStride: Dim, CStride: Dim>(
    (x, y, z): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: RealField,
{
    let delta = param_value("cs_delta", names, params);
    let radius = param_value("radius", names, params);

    // Compute internal coords (mu, nu).
    let r = (x.clone().powi(2) + z.clone().powi(2)).sqrt();

    let (mu, mut nu) = if r == T::zero() {
        (T::zero(), T::zero())
    } else {
        (
            r.clone() * (x.clone().powi(2) + z.clone().powi(2) * delta.clone().powi(2)).sqrt()
                / r
                / delta
                / radius,
            z.atan2(x) / T::from_usize(2).unwrap() / T::pi(),
        )
    };

    // Force nu into [0, 1].
    while nu < T::zero() {
        nu += T::one()
    }

    while nu > T::one() {
        nu -= T::one()
    }

    Vector3::new(mu, nu, y)
}

/// The elliptic-cylindric coordinate transformation (ics -> ecs).
pub fn ec_ics_to_ecs<T, const D: usize, RStride: Dim, CStride: Dim>(
    (mu, nu, s): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: RealField,
{
    let delta = param_value("cs_delta", names, params);
    let radius = param_value("radius", names, params);

    let omega = T::two_pi() * nu;

    let df = mu * delta.clone() * radius
        / (omega.clone().cos().powi(2) + (delta * omega.clone().sin()).powi(2)).sqrt();

    // Compute cartesian coords (x, y, z).
    let x = omega.clone().cos() * df.clone();
    let y = omega.sin() * df;

    Vector3::new(x, s, y)
}

// Implementation of the elliptic-cylindrical geometry.
impl_xcgm_geometry!(
    ECGeometry,
    "Elliptic-cylindric flux rope geometry.",
    ["phi", "theta", "psi", "y_0", "cs_delta", "radius", "x_0"],
    ec_basis,
    ec_sqrtdetg,
    ec_ecs_to_ics,
    ec_ics_to_ecs
);

#[cfg(test)]
mod tests {
    use super::*;
    use bayesfm::geometry::{Geometry, Geometry3D};
    use nalgebra::{SVector, U1, U3, Vector3};

    #[test]
    fn test_ec_coords() {
        let params = SVector::<f64, 7>::from([
            5.0_f64.to_radians(),
            -3.0_f64.to_radians(),
            5.0_f64.to_radians(),
            0.01,
            1.0,
            0.2,
            0.0,
        ]);

        let mut cs_state = XCState::default();

        ECGeometry::initialize_csst(&params.fixed_rows::<7>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.6, 0.11, 0.5);

        let ecs = ECGeometry::transform_internal_to_external::<U1, U3, _, _>(
            &ics_ref.as_view(),
            &params.fixed_rows::<7>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec = ECGeometry::transform_external_to_internal::<U1, U3, _, _>(
            &ecs.as_view(),
            &params.fixed_rows::<7>(0),
            &cs_state,
        )
        .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        ECGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<7>(0), 1e-6);
    }
}
