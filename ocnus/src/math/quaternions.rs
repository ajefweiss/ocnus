use nalgebra::{Scalar, SimdRealField, Unit, UnitQuaternion, Vector3};
use num_traits::Float;

/// Generate unit quaternion from three successive rotations around the z, y and x-axis.
pub fn quaternion_rot<T>(z_angle: T, y_angle: T, x_angle: T) -> UnitQuaternion<T>
where
    T: 'static + Float + Scalar + SimdRealField,
    <T as nalgebra::SimdValue>::Element: nalgebra::RealField,
{
    let uz = Vector3::<T>::z_axis();
    let uy = Vector3::<T>::y_axis();
    let ux = Vector3::<T>::x_axis();

    let rot_z = UnitQuaternion::from_axis_angle(&uz, z_angle);

    let rot_y = UnitQuaternion::from_axis_angle(
        &Unit::new_unchecked(rot_z.transform_vector(&uy)),
        -y_angle,
    );

    let rot_x = UnitQuaternion::from_axis_angle(
        &Unit::new_unchecked((rot_y * rot_z).transform_vector(&ux)),
        x_angle,
    );

    rot_x * (rot_y * rot_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternions() {
        assert!(quaternion_rot(0.0, 0.0, 0.0) == UnitQuaternion::identity());

        let rad90 = 90.0_f64.to_radians();

        assert!(
            quaternion_rot(rad90, 0.0, 0.0)
                == UnitQuaternion::new(Vector3::new(0.0, 0.0, 1.0) * rad90)
        );
        assert!(
            quaternion_rot(0.0, rad90, 0.0)
                == UnitQuaternion::new(Vector3::new(0.0, -1.0, 0.0) * rad90)
        );
        assert!(
            quaternion_rot(0.0, 0.0, rad90)
                == UnitQuaternion::new(Vector3::new(1.0, 0.0, 0.0) * rad90)
        );
    }
}
