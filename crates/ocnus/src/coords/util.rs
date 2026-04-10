use nalgebra::{Const, Dim, RealField, SVector, Unit, UnitQuaternion, Vector3, VectorView};

/// Retrieve a model parameter index by name.
pub fn param_index<const D: usize>(name: &str, names: &SVector<&'static str, D>) -> Option<usize> {
    names.iter().position(|param| *param == name)
}

/// Retrieve a model parameter value by name.
pub fn param_value<T, const D: usize, RStride: Dim, CStride: Dim>(
    name: &str,
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
) -> T
where
    T: Clone,
{
    let value = param_index(name, names).map(|index| params[index].clone());

    value.unwrap_or_else(|| panic!("model parameter '{}' not found", name))
}

/// Retrieve a model parameter value by name, with default return value.
pub fn param_value_or_else<T, const D: usize, RStride: Dim, CStride: Dim>(
    name: &str,
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    default: T,
) -> T
where
    T: Clone,
{
    let value = param_index(name, names).map(|index| params[index].clone());

    value.unwrap_or_else(|| default.clone())
}

/// Generate unit quaternion from three successive rotations around the z, y and x-axis.
pub fn quaternion_rot<T>(z_angle: T, y_angle: T, x_angle: T) -> UnitQuaternion<T>
where
    T: RealField,
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
        &Unit::new_unchecked((rot_y.clone() * rot_z.clone()).transform_vector(&ux)),
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
