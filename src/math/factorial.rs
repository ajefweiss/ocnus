use std::cmp::Ordering;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorials() {
        assert!(factorial(0).unwrap() == 1);
        assert!(factorial(1).unwrap() == 1);
        assert!(factorial(2).unwrap() == 2);
        assert!(factorial(3).unwrap() == 6);
        assert!(factorial(7).unwrap() == 5040);
        assert!(factorial(21).is_none());
    }
}
