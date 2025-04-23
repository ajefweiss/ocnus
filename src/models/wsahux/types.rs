use derive_more::{Deref, DerefMut};
use nalgebra::RealField;
use nalgebra::{DMatrix, Scalar};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

/// WSA-HUX input data structure.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WSAInputData<T>
where
    T: RealField,
{
    /// Longitudes.
    pub lon_1d: DMatrix<T>,

    /// Latitudes
    pub lat_1d: DMatrix<T>,

    /// Polarity map,
    pub pols: DMatrix<T>,

    /// Corohonal hole distance map.
    pub dmap: DMatrix<T>,

    /// Expansion factor map.
    pub efs: DMatrix<T>,

    /// Selected latitude indices.
    #[serde(default)]
    pub lat_indices: Vec<usize>,
}

impl<T> WSAInputData<T>
where
    T: RealField,
{
    /// The shape of the underlying input matrices.
    pub fn shape(&self) -> (usize, usize) {
        (self.lon_1d.len(), self.lat_1d.len())
    }
}

/// A full WSA-HUX state data structure.
/// The const generic `R` is the maximum radius in solar radii.
#[allow(missing_docs)]
#[derive(Clone, Debug, Serialize)]
pub struct WSAState<T, const R: usize>
where
    T: Clone + Scalar,
{
    pub angle: T,
    pub wsahux: Vec<WSASlice<T, R>>,
}

impl<T, const R: usize> WSAState<T, R>
where
    T: AsPrimitive<usize> + RealField,
{
    /// Initialize [`WSAState`] from given input date.
    pub fn initialize(&mut self, dr: T, input: &WSAInputData<T>) {
        let shape = input.shape();

        self.angle = T::zero();

        self.wsahux = if input.lat_indices.is_empty() {
            vec![WSASlice::new(dr, shape.0); shape.1]
        } else {
            vec![WSASlice::new(dr, shape.0); input.lat_indices.len()]
        };
    }
}

impl<T, const R: usize> Default for WSAState<T, R>
where
    T: AsPrimitive<usize> + Default + RealField,
{
    fn default() -> Self {
        Self {
            angle: T::default(),
            wsahux: vec![WSASlice::new(T::from_usize(R).unwrap(), 1); 1],
        }
    }
}

/// A single solar wind latitudinal slice, as used in the WSA-HUX model.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, Serialize)]
pub struct WSASlice<T, const R: usize>(pub DMatrix<T>)
where
    T: Clone + Scalar;

impl<T, const R: usize> WSASlice<T, R>
where
    T: AsPrimitive<usize> + RealField,
{
    /// Create a new [`WSASlice`].
    pub fn new(dr: T, lon: usize) -> Self {
        Self(DMatrix::zeros(
            (T::from_f64(R as f64).unwrap() / dr).as_(),
            lon,
        ))
    }
}
