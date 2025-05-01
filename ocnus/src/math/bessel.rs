use crate::math::factorial;
use nalgebra::RealField;
use std::cmp::Ordering;

/// Bessel function of the first kind.
pub fn bessel_jn<T>(x: T, k: usize) -> T
where
    T: Copy + RealField,
{
    match x.partial_cmp(&T::zero()) {
        Some(ord) => match ord {
            Ordering::Equal => T::from_usize(matches!(k, 0) as usize).unwrap(),
            _ => {
                let sum = (1..11).try_fold(
                    (x / (T::one() * T::from_usize(2).unwrap())).powi(k as i32),
                    |acc, idx| {
                        let next = T::from_i32(i32::pow(-1, idx as u32)).unwrap()
                            * (x / T::from_usize(2).unwrap()).powi((2 * idx + k) as i32)
                            / T::from_usize(factorial(idx).unwrap() * factorial(idx + k).unwrap())
                                .unwrap();

                        match next.partial_cmp(&T::zero()).unwrap() {
                            std::cmp::Ordering::Greater => {
                                match next.partial_cmp(&T::from_f64(1e-6).unwrap()).unwrap() {
                                    Ordering::Less => Err(acc + next),
                                    _ => Ok(acc + next),
                                }
                            }
                            std::cmp::Ordering::Less => {
                                match next.partial_cmp(&T::from_f64(-1e-6).unwrap()).unwrap() {
                                    Ordering::Greater => Err(acc + next),
                                    _ => Ok(acc + next),
                                }
                            }
                            std::cmp::Ordering::Equal => Err(acc),
                        }
                    },
                );

                match sum {
                    Ok(result) => result,
                    Err(result) => result,
                }
            }
        },
        None => (-T::one()).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bessel() {
        assert!((bessel_jn(0.0f32, 0) - 1.0).abs() < 1e-6);
        assert!((bessel_jn(0.0f32, 1)).abs() < 1e-6);
        assert!((bessel_jn(2.404826f32, 0)).abs() < 1e-6);
        assert!((bessel_jn(5.520071f32, 0)).abs() < 1e-6);
        assert!((bessel_jn(3.831705f32, 1)).abs() < 1e-6);
    }
}
