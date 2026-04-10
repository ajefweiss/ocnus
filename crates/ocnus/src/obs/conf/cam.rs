use crate::obs::{
    Obs, ObsTime,
    conf::{ObsCam, ObsPosition},
};
use derive_more::Deref;
use nalgebra::{RealField, SMatrix, SVector, Scalar, Unit, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use wcs::{ImgXY, LonLat, WCS, WCSParams};

/// A struct for storing a 3D observation configuration with a camera defined by the WCS.
#[derive(Debug, Deref, Deserialize, Serialize)]
pub struct CamConf<T>
where
    T: Scalar,
{
    timestamp: T,
    position: Vector3<T>,

    #[deref]
    /// WCS params object that describes the camera.
    params: WCSParams,

    /// The polarization filter angle (optional).
    polangle: Option<T>,
}

impl<T> CamConf<T>
where
    T: RealField,
{
    /// Returns a matrix consisting of corner vectors that define the FOV of the observation camera.
    /// Note: These calculations do not take into account the position of the observer (i.e. assumes a fixed position at the origin).
    pub fn fovs(&self) -> SMatrix<T, 3, 4> {
        let wcs = WCS::new(&self.params).unwrap();
        let position = &self.position;

        let image_res_x = wcs.img_dimensions()[0] as f64;
        let image_res_y = wcs.img_dimensions()[1] as f64;

        let mut x0 = wcs.unproj_lonlat(&ImgXY::new(0.0, 0.0)).unwrap();
        let mut x1 = wcs.unproj_lonlat(&ImgXY::new(image_res_x, 0.0)).unwrap();
        let mut x2 = wcs.unproj_lonlat(&ImgXY::new(0.0, image_res_y)).unwrap();
        let mut x3 = wcs
            .unproj_lonlat(&ImgXY::new(image_res_x, image_res_y))
            .unwrap();

        // Correct for left-handedness
        x0 = LonLat::new(f64::two_pi() - x0.lon(), x0.lat());
        x1 = LonLat::new(f64::two_pi() - x1.lon(), x1.lat());
        x2 = LonLat::new(f64::two_pi() - x2.lon(), x2.lat());
        x3 = LonLat::new(f64::two_pi() - x3.lon(), x3.lat());

        // Project external coordinates onto the celestian sphere, as seen by the observed.
        // Build a quaternion to de-rotate the external coordinates into a system with lon/lat.
        let ux = Vector3::<T>::x_axis();

        let rot_axis = if (position.normalize().clone() - ux.clone().into_inner()).norm()
            < T::from_f64(5e-2).unwrap()
        {
            Vector3::<T>::z_axis()
        } else {
            Unit::new_normalize(ux.clone().cross(position))
        };

        let rot_angle = (ux.dot(&-position) / position.norm()).acos();

        let derot_q = UnitQuaternion::from_axis_angle(&rot_axis, rot_angle).conjugate();

        let derot_x0 = {
            let xyz = x0.to_xyz();

            derot_q.transform_vector(&Vector3::from([
                T::from_f64(xyz.x()).unwrap(),
                T::from_f64(xyz.y()).unwrap(),
                T::from_f64(xyz.z()).unwrap(),
            ]))
        };

        let derot_x1 = {
            let xyz = x1.to_xyz();

            derot_q.transform_vector(&Vector3::from([
                T::from_f64(xyz.x()).unwrap(),
                T::from_f64(xyz.y()).unwrap(),
                T::from_f64(xyz.z()).unwrap(),
            ]))
        };

        let derot_x2 = {
            let xyz = x2.to_xyz();

            derot_q.transform_vector(&Vector3::from([
                T::from_f64(xyz.x()).unwrap(),
                T::from_f64(xyz.y()).unwrap(),
                T::from_f64(xyz.z()).unwrap(),
            ]))
        };

        let derot_x3 = {
            let xyz = x3.to_xyz();

            derot_q.transform_vector(&Vector3::from([
                T::from_f64(xyz.x()).unwrap(),
                T::from_f64(xyz.y()).unwrap(),
                T::from_f64(xyz.z()).unwrap(),
            ]))
        };

        SMatrix::from_columns(&[derot_x0, derot_x1, derot_x2, derot_x3])
    }

    /// Creates a new [`CamConf`] object.
    pub fn new(timestamp: T, position: Vector3<T>, params: WCSParams, polangle: Option<T>) -> Self {
        Self {
            timestamp,
            position,
            params,
            polangle,
        }
    }
}

// TODO: get this into the WCS crate
impl<T> Clone for CamConf<T>
where
    T: Scalar,
{
    fn clone(&self) -> Self {
        CamConf {
            timestamp: self.timestamp.clone(),
            position: self.position.clone(),
            params: serde_json5::from_str(&serde_json5::to_string(&self.params).unwrap()).unwrap(),
            polangle: self.polangle.clone(),
        }
    }
}

// TODO: get this into the WCS crate
impl<T> PartialEq for CamConf<T>
where
    T: Scalar + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
            && self.position == other.position
            && self.params.ctype1 == other.params.ctype1
            && self.params.ctype2 == other.params.ctype2
            && self.params.naxis == other.params.naxis
            && self.params.naxis1 == other.params.naxis1
            && self.params.epoch == other.params.epoch
            && self.params.crpix1 == other.params.crpix1
            && self.params.crpix2 == other.params.crpix2
            && self.polangle == other.polangle
    }
}

impl<T> From<(T, Vector3<T>, WCSParams, Option<T>)> for CamConf<T>
where
    T: Scalar,
{
    fn from(value: (T, Vector3<T>, WCSParams, Option<T>)) -> Self {
        Self {
            timestamp: value.0,
            position: value.1,
            params: value.2,
            polangle: value.3,
        }
    }
}

impl<T> ObsTime<T> for CamConf<T>
where
    T: PartialEq + Scalar,
{
    fn timestamp(&self) -> T {
        self.timestamp.clone()
    }
}

impl<T> ObsPosition<T, 3> for CamConf<T>
where
    T: Scalar,
{
    fn position(&self) -> SVector<T, 3> {
        self.position.clone()
    }
}

impl<T> ObsCam<T> for CamConf<T>
where
    T: Scalar,
{
    fn polarization(&self) -> Option<T> {
        self.polangle.clone()
    }

    fn wcs(&self) -> &WCSParams {
        &self.params
    }
}

impl<T> Obs<T, CamConf<T>>
where
    T: Scalar,
{
    /// Returns the position of the observation.
    pub fn positions(&self) -> Vec<SVector<T, 3>> {
        self.conf.iter().map(|conf| conf.position()).collect()
    }
}
