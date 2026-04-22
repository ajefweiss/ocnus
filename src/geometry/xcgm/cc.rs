use crate::geometry::xcgm::{XCState, impl_xcgm_geom};
use bayesfm::geometry::{param_value, param_value_or_else, quaternion_rot};
use nalgebra::{Const, Dim, RealField, SVector, Vector3, VectorView};
use std::marker::PhantomData;

/// The circular-cylindric contravariant basis vectors.
#[allow(clippy::extra_unused_type_parameters)]
pub fn cc_basis<T, const D: usize, RStride: Dim, CStride: Dim>(
    (mu, nu, _z): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    T: Copy + RealField,
{
    let radius = param_value("radius", names, params);

    let omega = T::two_pi() * nu;

    let dr = Vector3::from([omega.cos(), T::zero(), omega.sin()]) * radius;
    let dnu = Vector3::from([-omega.sin(), T::zero(), omega.cos()]) * T::two_pi() * mu * radius;
    let dz = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dr, dnu, dz]
}

/// The circular-cylindric metric determinant.
#[allow(clippy::extra_unused_type_parameters)]
pub fn cc_sqrtdetg<T, const D: usize, RStride: Dim, CStride: Dim>(
    (mu, _nu, _z): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> T
where
    T: Copy + RealField,
{
    let radius = param_value("radius", names, params);

    T::two_pi() * (mu * radius.powi(2))
}

/// The circular-cylindric coordinate transformation (ecs -> ics).
pub fn cc_ecs_to_ics<T, const D: usize, RStride: Dim, CStride: Dim>(
    (x, y, z): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: Copy + RealField,
{
    let radius = param_value("radius", names, params);

    // Compute polar coords (mu, omega).
    let mu = (x.powi(2) + z.powi(2)).sqrt() / radius;

    let mut nu = if mu == T::zero() {
        T::zero()
    } else {
        z.atan2(x) / T::two_pi()
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

/// The circular-cylindric coordinate transformation (ics -> ecs).
pub fn cc_ics_to_ecs<T, const D: usize, RStride: Dim, CStride: Dim>(
    (mu, nu, y): (T, T, T),
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: Copy + RealField,
{
    let radius = param_value("radius", names, params);

    let omega = T::two_pi() * nu;

    // Compute cartesian coords (x, y, z).
    let x = omega.cos() * mu * radius;
    let z = omega.sin() * mu * radius;

    Vector3::new(x, y, z)
}

// Implementation of the circular-cylindrical geometry.
impl_xcgm_geom!(
    CCGeometry,
    "Circular-cylindric flux rope geometry.",
    ["phi", "theta", "y_0", "radius", "x_0"],
    cc_basis,
    cc_sqrtdetg,
    cc_ecs_to_ics,
    cc_ics_to_ecs
);

#[cfg(test)]
mod tests {
    use super::*;
    use bayesfm::geometry::{BFMGeometry, BFMGeometry3D};
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_cc_coords() {
        let params =
            SVector::<f64, 5>::from([5.0_f64.to_radians(), -3.0_f64.to_radians(), 0.01, 0.2, 0.0]);

        let mut cs_state = XCState::default();

        CCGeometry::initialize_csst(&params.fixed_rows::<5>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.6, 0.11, 0.5);

        let ecs = CCGeometry::transform_internal_to_external(
            &ics_ref.as_view(),
            &params.fixed_rows::<5>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec = CCGeometry::transform_external_to_internal(
            &ecs.as_view(),
            &params.fixed_rows::<5>(0),
            &cs_state,
        )
        .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        CCGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<5>(0), 1e-6);
    }
}
