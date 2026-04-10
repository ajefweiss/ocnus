//! # Curvilinear coordinate systems and model geometries.
//!
//! This module introduces the [`Coordinates`] trait, which is the trait that is shared by all curvilinear coordinate systems that define a model geometry.
//! The trait provides bi-directional coordinate transformation functions, and methods that compute the covariant and contravariant basis vectors.
//! Note that the basis vectors of the coordinate systems are not necessarily orthonormal.
//! Therefore, one must properly account for using co- and contravariant basis vectors. Simple geometries may nonetheless have orthogonal basis vectors.

//! Currently implemented coordinate systems & model geometries:
//! - [`CCGeometry`] A circular-cylindrical geometry with internal coords (r, ϕ, z) for flux
//!   rope models.
//! - [`ECGeometry`] An elliptic-cylindrical geometry with internal coords (μ, ν, z) for flux
//!   rope models.
//! - [`TTGeometry`] A tapered-toroidal geometry with an elliptical cross-section and internal
//!   coords (μ, ν, ) for flux rope models.
//! - [`SPHGeometry`] A spherical geometry with internal coordiantes (r, ϕ, θ) for spheromak models.
//! - [`SPHUGeometry`] A unit sphere with internal coordiantes (r, ϕ, θ) for global heliospheric models.
//!
//! Each geometry is associated with a fixed coordinate system state type, which enables the description of time-varying coordatinate systems.
//! The coordinate system state types must be initialized from the coordinate system parameters using an implementation of [`Coordinates::initialize_cs`].

mod cartesian;
mod linear;
mod sphgm;
mod util;

pub use cartesian::*;
pub use linear::LinearGeometry;
pub use sphgm::{SPHGeometry, SPHState, SPHUGeometry};
pub use util::*;

use nalgebra::{Const, DMatrix, Dim, RealField, SVector, SVectorView, Vector3, VectorView};
use rayon::prelude::*;
use std::iter::Sum;

/// A trait that is shared by all D-dimensional coordinate systems describing a model geometry.
pub trait Coordinates<T, const D: usize, const P: usize>
where
    T: RealField,
{
    /// Coordinate system parameter names.
    const PARAMS: SVector<&'static str, P>;

    /// Number of coordinates.
    const NDIMS: usize = D;

    /// Number of coordinate system parameters.
    const NPARAMS: usize = P;

    /// Associated coordinate system state type.
    type CSST;

    /// Returns the local contravariant basis vectors.
    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<[SVector<T, D>; D]>;

    /// Returns the local contravariant basis vectors and returns the normalized vectors.
    fn contravariant_basis_normalized<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<[SVector<T, D>; D]> {
        let mut basis = Self::contravariant_basis::<RStride, CStride>(ics, params, cs_state)?;

        basis.iter_mut().for_each(|v| {
            *v = v.normalize();
        });

        Some(basis)
    }

    /// Returns the local covariant basis vectors.
    fn covariant_basis<RStride: Dim, CStride: Dim>(
        _ics: &SVectorView<T, D>,
        _params: &SVectorView<T, P>,
        _state: &Self::CSST,
    ) -> Option<[SVector<T, D>; D]> {
        unimplemented!("covariant basis vectors are currently not implemented")
    }

    /// Create a vector from contravariant components.
    fn contravariant_vector<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, D>,
        components: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>
    where
        SVector<T, D>: Sum,
    {
        let basis = Self::contravariant_basis::<RStride, CStride>(ics, params, cs_state)?;

        Some(
            basis
                .iter()
                .zip(components.iter())
                .map(|(basis, component)| basis * component.clone())
                .sum(),
        )
    }

    /// Create a vector from contravariant components, using the normalized basis vectors.
    fn contravariant_vector_normalized<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, D>,
        components: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>
    where
        SVector<T, D>: Sum,
    {
        let basis =
            Self::contravariant_basis_normalized::<RStride, CStride>(ics, params, cs_state)?;

        Some(
            basis
                .iter()
                .zip(components.iter())
                .map(|(basis, component)| basis * component.clone())
                .sum(),
        )
    }

    /// Returns the square root of the determinant of the metric tensor.
    fn sqrtdetg<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<T>;

    /// Initialize the coordinate system state.
    ///
    /// This function may panic for invalid parameter inputs.
    fn initialize_cs<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &mut Self::CSST,
    );

    /// Transform external coords `ecs` into the internal coords `ics`.
    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>;

    /// Transform internal coords `ics` into cartesian coords `ecs`.
    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>;
}

/// A trait that is shared by all 3-dimensional coordinate systems describing a model geometry.
pub trait Coordinates3D<T, const P: usize>: Coordinates<T, 3, P>
where
    T: RealField,
{
    /// Returns three coordinate matrices that describe an iso-surface (for constant mu).
    fn iso_surface_mu<const RCS: usize, RStride: Dim, CStride: Dim>(
        mu: T,
        nu_matrix: DMatrix<T>,
        s_matrix: DMatrix<T>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<[DMatrix<T>; 3]>
    where
        Self::CSST: Sync,
    {
        let vectors = nu_matrix
            .par_column_iter()
            .zip(s_matrix.par_column_iter())
            .chunks(RCS)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .flat_map(|(row_nu, row_s)| {
                        row_nu.iter().zip(row_s.iter()).map(|(nu, s)| {
                            Self::transform_ics_to_ecs(
                                &Vector3::from([mu.clone(), nu.clone(), s.clone()]).as_view(),
                                params,
                                cs_state,
                            )
                        })
                    })
                    .collect::<Vec<Option<SVector<T, 3>>>>()
            })
            .collect::<Option<Vec<SVector<T, 3>>>>()?;

        let rc = nu_matrix.nrows();
        let cc = nu_matrix.ncols();

        let xx = DMatrix::from_iterator(rc, cc, vectors.iter().map(|vec| vec.x.clone()));
        let yy = DMatrix::from_iterator(rc, cc, vectors.iter().map(|vec| vec.y.clone()));
        let zz = DMatrix::from_iterator(rc, cc, vectors.iter().map(|vec| vec.z.clone()));

        Some([xx, yy, zz])
    }

    /// Test the trait functions for a specific implementation.
    fn test_implementation<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, 3>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        delta_h: T,
    ) where
        Self::CSST: Default,
    {
        use approx::ulps_eq;
        use nalgebra::Vector3;

        let mut cs_state = Self::CSST::default();

        Self::initialize_cs::<RStride, CStride>(params, &mut cs_state);

        let ics_1p =
            ics + Vector3::<T>::x_axis().into_inner() * delta_h.clone() / T::from_usize(2).unwrap();
        let ics_1m =
            ics - Vector3::<T>::x_axis().into_inner() * delta_h.clone() / T::from_usize(2).unwrap();

        let ics_2p =
            ics + Vector3::<T>::y_axis().into_inner() * delta_h.clone() / T::from_usize(2).unwrap();
        let ics_2m =
            ics - Vector3::<T>::y_axis().into_inner() * delta_h.clone() / T::from_usize(2).unwrap();

        let ics_3p =
            ics + Vector3::<T>::z_axis().into_inner() * delta_h.clone() / T::from_usize(2).unwrap();
        let ics_3m =
            ics - Vector3::<T>::z_axis().into_inner() * delta_h.clone() / T::from_usize(2).unwrap();

        let basis = Self::contravariant_basis(
            &ics.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_1p = Self::transform_ics_to_ecs(
            &ics_1p.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_1m = Self::transform_ics_to_ecs(
            &ics_1m.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_2p = Self::transform_ics_to_ecs(
            &ics_2p.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_2m = Self::transform_ics_to_ecs(
            &ics_2m.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_3p = Self::transform_ics_to_ecs(
            &ics_3p.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_3m = Self::transform_ics_to_ecs(
            &ics_3m.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        assert!(ulps_eq!(
            basis[0],
            (ecs_1p - ecs_1m) / delta_h.clone(),
            max_ulps = 5,
            epsilon = T::from_f64(1e-5).unwrap(),
        ));
        assert!(ulps_eq!(
            basis[1],
            (ecs_2p - ecs_2m) / delta_h.clone(),
            max_ulps = 5,
            epsilon = T::from_f64(1e-5).unwrap()
        ));
        assert!(ulps_eq!(
            basis[2],
            (ecs_3p - ecs_3m) / delta_h.clone(),
            max_ulps = 5,
            epsilon = T::from_f64(1e-5).unwrap()
        ));

        let sqrtdetg_basis = (basis[0].cross(&basis[1]).dot(&basis[2])).abs();
        let sqrtdetg_analy = Self::sqrtdetg(
            &ics.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        assert!(approx::ulps_eq!(sqrtdetg_basis, sqrtdetg_analy));
    }
}

// Blank implementation for 3D coordinates.
impl<T, OC, const P: usize> Coordinates3D<T, P> for OC
where
    T: RealField,
    Self: Coordinates<T, 3, P>,
{
}
