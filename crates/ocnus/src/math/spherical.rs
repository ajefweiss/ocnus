// use nalgebra::RealField;

// /// Returns the real spherical harmonic function of order (m, l), normalized as in geodesy.
// #[inline]
// pub fn sph_yml<T>(m: usize, l: isize) -> impl Fn(T, T) -> T
// where
//     T: RealField,
// {
//     match (m, l) {
//         (0, 0) => sph_y00,

//         (1, -1) => sph_y11s,
//         (1, 0) => sph_y10,
//         (1, 1) => sph_y11c,

//         (2, -2) => sph_y22s,
//         (2, -1) => sph_y21s,
//         (2, 0) => sph_y20,
//         (2, 1) => sph_y21c,
//         (2, 2) => sph_y22c,

//         (3, -3) => sph_y33s,
//         (3, -2) => sph_y32s,
//         (3, -1) => sph_y31s,
//         (3, 0) => sph_y30,
//         (3, 1) => sph_y31c,
//         (3, 2) => sph_y32c,
//         (3, 3) => sph_y33c,

//         _ => unimplemented!(),
//     }
// }

// pub fn sph_y00<T>(_theta: T, _varphi: T) -> T
// where
//     T: RealField,
// {
//     T::one()
// }

// pub fn sph_y11s<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(3.0).unwrap()).sqrt() * theta.sin() * varphi.sin()
// }

// pub fn sph_y10<T>(theta: T, _varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(3.0).unwrap()).sqrt() * theta.cos()
// }

// pub fn sph_y11c<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(3.0).unwrap()).sqrt() * theta.sin() * varphi.cos()
// }

// pub fn sph_y22s<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
//         * theta.sin().powi(2)
//         * (T::from_usize(2).unwrap() * varphi).sin()
// }

// pub fn sph_y21s<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
//         * (T::from_usize(2).unwrap() * theta).sin()
//         * varphi.sin()
// }

// pub fn sph_y20<T>(theta: T, _varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(5.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
//         * (T::from_usize(3).unwrap() * theta.cos().powi(2) - T::one())
// }

// pub fn sph_y21c<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
//         * (T::from_usize(2).unwrap() * theta).sin()
//         * varphi.cos()
// }

// pub fn sph_y22c<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
//         * theta.sin().powi(2)
//         * (T::from_usize(2).unwrap() * varphi).cos()
// }

// pub fn sph_y33s<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(35.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
//         * theta.sin().powi(3)
//         * (T::from_usize(3).unwrap() * varphi).sin()
// }

// pub fn sph_y32s<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(105.0).unwrap() / T::from_usize(4).unwrap()).sqrt()
//         * theta.clone().sin().powi(2)
//         * theta.cos()
//         * (T::from_usize(2).unwrap() * varphi).sin()
// }

// pub fn sph_y31s<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(21.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
//         * theta.clone().sin()
//         * (T::from_usize(5).unwrap() * theta.cos().powi(2) - T::one())
//         * varphi.sin()
// }

// pub fn sph_y30<T>(theta: T, _varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(7.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
//         * (T::from_usize(5).unwrap() * theta.clone().cos().powi(3)
//             - T::from_usize(3).unwrap() * theta.cos())
// }

// pub fn sph_y31c<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(21.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
//         * theta.clone().sin()
//         * (T::from_usize(5).unwrap() * theta.cos().powi(2) - T::one())
//         * varphi.cos()
// }

// pub fn sph_y32c<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(105.0).unwrap() / T::from_usize(4).unwrap()).sqrt()
//         * theta.clone().sin().powi(2)
//         * theta.cos()
//         * (T::from_usize(2).unwrap() * varphi).cos()
// }

// pub fn sph_y33c<T>(theta: T, varphi: T) -> T
// where
//     T: RealField,
// {
//     (T::from_f64(35.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
//         * theta.sin().powi(3)
//         * (T::from_usize(3).unwrap() * varphi).cos()
// }

use nalgebra::RealField;

/// Returns the real spherical harmonic function of order (m, l), normalized as in geodesy.
#[inline]
pub fn sph_yml<T>(m: usize, l: isize) -> impl Fn(T, T) -> T
where
    T: RealField,
{
    match (m, l) {
        (0, 0) => sph_y00,

        (1, -1) => sph_y11s,
        (1, 0) => sph_y10,
        (1, 1) => sph_y11c,

        (2, -2) => sph_y22s,
        (2, -1) => sph_y21s,
        (2, 0) => sph_y20,
        (2, 1) => sph_y21c,
        (2, 2) => sph_y22c,

        (3, -3) => sph_y33s,
        (3, -2) => sph_y32s,
        (3, -1) => sph_y31s,
        (3, 0) => sph_y30,
        (3, 1) => sph_y31c,
        (3, 2) => sph_y32c,
        (3, 3) => sph_y33c,

        (4, -4) => sph_y44s,
        (4, -3) => sph_y43s,
        (4, -2) => sph_y42s,
        (4, -1) => sph_y41s,
        (4, 0) => sph_y40,
        (4, 1) => sph_y41c,
        (4, 2) => sph_y42c,
        (4, 3) => sph_y43c,
        (4, 4) => sph_y44c,

        (5, -5) => sph_y55s,
        (5, -4) => sph_y54s,
        (5, -3) => sph_y53s,
        (5, -2) => sph_y52s,
        (5, -1) => sph_y51s,
        (5, 0) => sph_y50,
        (5, 1) => sph_y51c,
        (5, 2) => sph_y52c,
        (5, 3) => sph_y53c,
        (5, 4) => sph_y54c,
        (5, 5) => sph_y55c,

        _ => unimplemented!(),
    }
}

pub fn sph_y00<T>(_theta: T, _varphi: T) -> T
where
    T: RealField,
{
    T::one()
}

pub fn sph_y11s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(3.0).unwrap()).sqrt() * theta.sin() * varphi.sin()
}

pub fn sph_y10<T>(theta: T, _varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(3.0).unwrap()).sqrt() * theta.cos()
}

pub fn sph_y11c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(3.0).unwrap()).sqrt() * theta.sin() * varphi.cos()
}

pub fn sph_y22s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
        * theta.sin().powi(2)
        * (T::from_usize(2).unwrap() * varphi).sin()
}

pub fn sph_y21s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
        * (T::from_usize(2).unwrap() * theta).sin()
        * varphi.sin()
}

pub fn sph_y20<T>(theta: T, _varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(5.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
        * (T::from_usize(3).unwrap() * theta.cos().powi(2) - T::one())
}

pub fn sph_y21c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
        * (T::from_usize(2).unwrap() * theta).sin()
        * varphi.cos()
}

pub fn sph_y22c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(15.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
        * theta.sin().powi(2)
        * (T::from_usize(2).unwrap() * varphi).cos()
}

pub fn sph_y33s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(35.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.sin().powi(3)
        * (T::from_usize(3).unwrap() * varphi).sin()
}

pub fn sph_y32s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(105.0).unwrap() / T::from_usize(4).unwrap()).sqrt()
        * theta.clone().sin().powi(2)
        * theta.cos()
        * (T::from_usize(2).unwrap() * varphi).sin()
}

pub fn sph_y31s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(21.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.clone().sin()
        * (T::from_usize(5).unwrap() * theta.cos().powi(2) - T::one())
        * varphi.sin()
}

pub fn sph_y30<T>(theta: T, _varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(7.0).unwrap()).sqrt() / T::from_usize(2).unwrap()
        * (T::from_usize(5).unwrap() * theta.clone().cos().powi(3)
            - T::from_usize(3).unwrap() * theta.cos())
}

pub fn sph_y31c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(21.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.clone().sin()
        * (T::from_usize(5).unwrap() * theta.cos().powi(2) - T::one())
        * varphi.cos()
}

pub fn sph_y32c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(105.0).unwrap() / T::from_usize(4).unwrap()).sqrt()
        * theta.clone().sin().powi(2)
        * theta.cos()
        * (T::from_usize(2).unwrap() * varphi).cos()
}

pub fn sph_y33c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(35.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.sin().powi(3)
        * (T::from_usize(3).unwrap() * varphi).cos()
}

// l=4 functions

pub fn sph_y44s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(315.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * theta.sin().powi(4)
        * (T::from_usize(4).unwrap() * varphi).sin()
}

pub fn sph_y43s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(315.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.clone().sin().powi(3)
        * theta.cos()
        * (T::from_usize(3).unwrap() * varphi).sin()
}

pub fn sph_y42s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(45.0).unwrap() / T::from_usize(16).unwrap()).sqrt()
        * theta.clone().sin().powi(2)
        * (T::from_usize(7).unwrap() * theta.cos().powi(2) - T::one())
        * (T::from_usize(2).unwrap() * varphi).sin()
}

pub fn sph_y41s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(45.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.clone().sin()
        * (T::from_usize(7).unwrap() * theta.clone().cos().powi(3)
            - T::from_usize(3).unwrap() * theta.cos())
        * varphi.sin()
}

pub fn sph_y40<T>(theta: T, _varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(9.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * (T::from_usize(35).unwrap() * theta.clone().cos().powi(4)
            - T::from_usize(30).unwrap() * theta.clone().cos().powi(2)
            + T::from_usize(3).unwrap())
}

pub fn sph_y41c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(45.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.clone().sin()
        * (T::from_usize(7).unwrap() * theta.clone().cos().powi(3)
            - T::from_usize(3).unwrap() * theta.cos())
        * varphi.cos()
}

pub fn sph_y42c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(45.0).unwrap() / T::from_usize(16).unwrap()).sqrt()
        * theta.clone().sin().powi(2)
        * (T::from_usize(7).unwrap() * theta.cos().powi(2) - T::one())
        * (T::from_usize(2).unwrap() * varphi).cos()
}

pub fn sph_y43c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(315.0).unwrap() / T::from_usize(8).unwrap()).sqrt()
        * theta.clone().sin().powi(3)
        * theta.cos()
        * (T::from_usize(3).unwrap() * varphi).cos()
}

pub fn sph_y44c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(315.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * theta.sin().powi(4)
        * (T::from_usize(4).unwrap() * varphi).cos()
}

// l=5 functions

pub fn sph_y55s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(693.0).unwrap() / T::from_usize(128).unwrap()).sqrt()
        * theta.sin().powi(5)
        * (T::from_usize(5).unwrap() * varphi).sin()
}

pub fn sph_y54s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(3465.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * theta.clone().sin().powi(4)
        * theta.cos()
        * (T::from_usize(4).unwrap() * varphi).sin()
}

pub fn sph_y53s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(385.0).unwrap() / T::from_usize(128).unwrap()).sqrt()
        * theta.clone().sin().powi(3)
        * (T::from_usize(9).unwrap() * theta.cos().powi(2) - T::one())
        * (T::from_usize(3).unwrap() * varphi).sin()
}

pub fn sph_y52s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(1155.0).unwrap() / T::from_usize(16).unwrap()).sqrt()
        * theta.clone().sin().powi(2)
        * (T::from_usize(3).unwrap() * theta.clone().cos().powi(3) - theta.cos())
        * (T::from_usize(2).unwrap() * varphi).sin()
}

pub fn sph_y51s<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(165.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * theta.clone().sin()
        * (T::from_usize(21).unwrap() * theta.clone().cos().powi(4)
            - T::from_usize(14).unwrap() * theta.clone().cos().powi(2)
            + T::one())
        * varphi.sin()
}

pub fn sph_y50<T>(theta: T, _varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(11.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * (T::from_usize(63).unwrap() * theta.clone().cos().powi(5)
            - T::from_usize(70).unwrap() * theta.clone().cos().powi(3)
            + T::from_usize(15).unwrap() * theta.cos())
}

pub fn sph_y51c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(165.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * theta.clone().sin()
        * (T::from_usize(21).unwrap() * theta.clone().cos().powi(4)
            - T::from_usize(14).unwrap() * theta.clone().cos().powi(2)
            + T::one())
        * varphi.cos()
}

pub fn sph_y52c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(1155.0).unwrap() / T::from_usize(16).unwrap()).sqrt()
        * theta.clone().sin().powi(2)
        * (T::from_usize(3).unwrap() * theta.clone().cos().powi(3) - theta.cos())
        * (T::from_usize(2).unwrap() * varphi).cos()
}

pub fn sph_y53c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(385.0).unwrap() / T::from_usize(128).unwrap()).sqrt()
        * theta.clone().sin().powi(3)
        * (T::from_usize(9).unwrap() * theta.cos().powi(2) - T::one())
        * (T::from_usize(3).unwrap() * varphi).cos()
}

pub fn sph_y54c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(3465.0).unwrap() / T::from_usize(64).unwrap()).sqrt()
        * theta.clone().sin().powi(4)
        * theta.cos()
        * (T::from_usize(4).unwrap() * varphi).cos()
}

pub fn sph_y55c<T>(theta: T, varphi: T) -> T
where
    T: RealField,
{
    (T::from_f64(693.0).unwrap() / T::from_usize(128).unwrap()).sqrt()
        * theta.sin().powi(5)
        * (T::from_usize(5).unwrap() * varphi).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;

    fn ortho_integral(f: impl Fn(f64, f64) -> f64, g: impl Fn(f64, f64) -> f64) -> f64 {
        let n_theta = 100;
        let n_varphi = 100;
        let d_theta = std::f64::consts::PI / n_theta as f64;
        let d_varphi = 2.0 * std::f64::consts::PI / n_varphi as f64;

        let mut integral = 0.0;

        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * d_theta;
            for j in 0..n_varphi {
                let varphi = (j as f64 + 0.5) * d_varphi;
                integral += f(theta, varphi) * g(theta, varphi) * theta.sin() * d_theta * d_varphi;
            }
        }
        integral
    }

    #[test]
    fn test_spherical_harmonics() {
        for l1 in 0..6 {
            for m1 in -l1..(l1 + 1) {
                for l2 in 0..6 {
                    for m2 in -l2..(l2 + 1) {
                        let integral =
                            ortho_integral(sph_yml(l1 as usize, m1), sph_yml(l2 as usize, m2));

                        if l1 == l2 && m1 == m2 {
                            assert!(ulps_eq!(
                                integral,
                                4.0 * std::f64::consts::PI,
                                max_ulps = 5,
                                epsilon = 1e-2
                            ));
                        } else {
                            assert!(ulps_eq!(integral, 0.0, max_ulps = 5, epsilon = 1e-2));
                        }
                    }
                }
            }
        }
    }
}
