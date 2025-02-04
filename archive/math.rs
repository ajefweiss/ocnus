//! A collection of miscellaneous math routines.

use crate::{Fp, FP_EPSILON};
use nalgebra::{RealField, Unit, UnitQuaternion, Vector3};
use std::cmp::Ordering;

/// Bessel function of the first kind.
pub fn bessel_jn(x: Fp, k: usize) -> Fp {
    match x.total_cmp(&0.0) {
        Ordering::Equal => (matches!(k, 0) as usize) as Fp,
        _ => {
            let result = (1..11).try_fold((x / 2.0).powi(k as i32), |sum, idx| {
                let next = i32::pow(-1, idx as u32) as Fp * (x / 2.0).powi((2 * idx + k) as i32)
                    / (factorial(idx).unwrap() * factorial(idx + k).unwrap()) as Fp;

                match next.partial_cmp(&0.0).unwrap() {
                    std::cmp::Ordering::Greater => match next.partial_cmp(&FP_EPSILON).unwrap() {
                        Ordering::Less => Err(sum + next),
                        _ => Ok(sum + next),
                    },
                    std::cmp::Ordering::Less => match next.partial_cmp(&(-FP_EPSILON)).unwrap() {
                        Ordering::Greater => Err(sum + next),
                        _ => Ok(sum + next),
                    },
                    std::cmp::Ordering::Equal => Err(sum),
                }
            });

            match result {
                Ok(value) => value,
                Err(value) => value,
            }
        }
    }
}

/// A look up table for factorials up to n = 20.
static FACTORIAL_LUT: [usize; 21] = {
    let mut lut = [1; 21];

    let mut n = 2;

    while n < 21 {
        lut[n] = lut[n - 1] * n;
        n += 1;
    }

    lut
};

/// Computes the factorial `n!`.
///
/// This returns `None` if n > 20.
pub fn factorial(n: usize) -> Option<usize> {
    match n.cmp(&21) {
        Ordering::Less => Some(FACTORIAL_LUT[n]),
        _ => None,
    }
}

/// Generate unit quaternion from three successive rotations around the z, y and x-axis.
pub fn quaternion_xyz<T>(z_angle: T, y_angle: T, x_angle: T) -> UnitQuaternion<T>
where
    T: Copy + RealField,
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
    fn test_bessel() {
        assert!((bessel_jn(0.0, 0) - 1.0).abs() < 1e-6);
        assert!(bessel_jn(0.0, 1).abs() < 1e-6);
        assert!(bessel_jn(2.4048, 0).abs() < 1e-4);
        assert!(bessel_jn(5.5201, 0).abs() < 1e-4);
        assert!(bessel_jn(3.8317, 1).abs() < 1e-4);
    }

    #[test]
    fn test_factorials() {
        assert!(factorial(0).unwrap() == 1);
        assert!(factorial(1).unwrap() == 1);
        assert!(factorial(2).unwrap() == 2);
        assert!(factorial(3).unwrap() == 6);
        assert!(factorial(7).unwrap() == 5040);
        assert!(factorial(21).is_none());
    }

    #[test]
    fn test_quaternions() {
        assert!(quaternion_xyz(0.0, 0.0, 0.0) == UnitQuaternion::identity());

        let rad90 = 90.0_f64.to_radians();

        assert!(
            quaternion_xyz(rad90, 0.0, 0.0)
                == UnitQuaternion::new(Vector3::new(0.0, 0.0, 1.0) * rad90)
        );
        assert!(
            quaternion_xyz(0.0, rad90, 0.0)
                == UnitQuaternion::new(Vector3::new(0.0, -1.0, 0.0) * rad90)
        );
        assert!(
            quaternion_xyz(0.0, 0.0, rad90)
                == UnitQuaternion::new(Vector3::new(1.0, 0.0, 0.0) * rad90)
        );
    }
}
