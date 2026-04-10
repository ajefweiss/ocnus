use std::{iter::Sum, ops::Div};

/// Return a vector with normalized values from a slice.
pub fn normalize<T>(slice: &[T]) -> Vec<T>
where
    T: Clone + Div<T, Output = T> + Sum,
{
    let total = slice.iter().cloned().sum::<T>();

    Vec::from_iter(slice.iter().map(|value| value.clone() / total.clone()))
}
