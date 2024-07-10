use crate::{
    fXX,
    math::{T, factorial, powi},
};
use std::cmp::Ordering;

/// Bessel function of the first kind.
pub fn bessel_jn<T>(x: T, k: usize) -> T
where
    T: fXX,
{
    match x.total_cmp(&T::zero()) {
        Ordering::Equal => T::from_usize(matches!(k, 0) as usize).unwrap(),
        _ => {
            let sum = (1..11).try_fold(powi!(x / (T::one() * T!(2.0)), k as i32), |acc, idx| {
                let next = T::from_i32(i32::pow(-1, idx as u32)).unwrap()
                    * powi!(x / (T!(2.0)), (2 * idx + k) as i32)
                    / T::from_usize(factorial(idx).unwrap() * factorial(idx + k).unwrap()).unwrap();

                match next.partial_cmp(&T::zero()).unwrap() {
                    std::cmp::Ordering::Greater => match next.partial_cmp(&T::epsilon()).unwrap() {
                        Ordering::Less => Err(acc + next),
                        _ => Ok(acc + next),
                    },
                    std::cmp::Ordering::Less => match next.partial_cmp(&(-T::epsilon())).unwrap() {
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
    use crate::math::{abs, bessel_jn};

    #[test]
    fn test_bessel() {
        assert!(abs!(bessel_jn(0.0, 0) - 1.0) < 1e-6);
        assert!(abs!(bessel_jn(0.0, 1)) < 1e-6);
        assert!(abs!(bessel_jn(2.4048, 0)) < 1e-4);
        assert!(abs!(bessel_jn(5.5201, 0)) < 1e-4);
        assert!(abs!(bessel_jn(3.8317, 1)) < 1e-4);
    }
}
