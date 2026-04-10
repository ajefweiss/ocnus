use itertools::Itertools;
use num_traits::AsPrimitive;

/// Return quantile values of a given slice.
pub fn quantiles<T>(values: &[T], quantiles: &[T]) -> Vec<T>
where
    T: AsPrimitive<f64> + PartialOrd,
{
    let valus_sorted = values
        .iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .collect::<Vec<T>>();

    let mut results = Vec::with_capacity(quantiles.len());

    for quantile in quantiles {
        results.push(valus_sorted[(values.len() as f64 * quantile.as_()) as usize]);
    }

    results
}

/// Return the larget `n` values of a slice.
pub fn largest_n<T>(values: &[T], n: usize) -> Vec<T>
where
    T: Clone + PartialOrd,
{
    let valus_sorted = values
        .iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .cloned()
        .collect::<Vec<T>>();

    Vec::from(&valus_sorted[(values.len() - n)..])
}
