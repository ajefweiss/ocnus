use crate::math::factorial;
use std::cmp::Ordering;

/// Bessel function of the first kind.
pub fn bessel_jn(x: f64, k: usize) -> f64 {
    match x.total_cmp(&0.0) {
        Ordering::Equal => (matches!(k, 0) as usize) as f64,
        _ => {
            let sum = (1..11).try_fold((x / 2.0).powi(k as i32), |acc, idx| {
                let next = i32::pow(-1, idx as u32) as f64 * (x / 2.0).powi((2 * idx + k) as i32)
                    / (factorial(idx).unwrap() * factorial(idx + k).unwrap()) as f64;

                match next.partial_cmp(&0.0).unwrap() {
                    std::cmp::Ordering::Greater => match next.partial_cmp(&f64::EPSILON).unwrap() {
                        Ordering::Less => Err(acc + next),
                        _ => Ok(acc + next),
                    },
                    std::cmp::Ordering::Less => match next.partial_cmp(&(-f64::EPSILON)).unwrap() {
                        Ordering::Greater => Err(acc + next),
                        _ => Ok(acc + next),
                    },
                    std::cmp::Ordering::Equal => Err(acc),
                }
            });

            match sum {
                Ok(result) => result,
                Err(result) => result,
            }
        }
    }
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
}
