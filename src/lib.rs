#[macro_use]
extern crate approx;

use approx::AbsDiffEq;
use hdf5::H5Type;
use ndarray::{Array, Dimension, Ix1, Ix2, Ix3};
use num_complex::Complex;
use num_traits::{
    cast::{AsPrimitive, FromPrimitive},
    Float,
};
use std::path::Path;

mod base;
mod utils;
mod uvh5;

pub use self::uvh5::UVH5;

pub use self::base::{
    ArrayMetaData, BltOrder, BltOrders, CatTypes, Catalog, EqConvention, Orientation, PhaseType,
    SiderealVal, UVMeta, UnphasedVal, VisUnit,
};
pub use self::utils::{
    antnums_to_baseline, baseline_to_antnums, ecef_from_enu, ecef_from_rot_ecef, enu_from_ecef,
    latlonalt_from_xyz, rot_ecef_from_ecef, xyz_from_latlonalt,
};

fn compare_complex_arrays<T, U>(
    array1: &Array<Complex<T>, U>,
    array2: &Array<Complex<T>, U>,
) -> bool
where
    T: Float + AbsDiffEq + AbsDiffEq<T> + AbsDiffEq<Epsilon = T>,
    U: Dimension,
{
    array1
        .mapv(|x| x.re)
        .abs_diff_eq(&array2.mapv(|x| x.re), T::from(1e-6).unwrap())
        && array1
            .mapv(|x| x.im)
            .abs_diff_eq(&array2.mapv(|x| x.im), T::from(1e-6).unwrap())
}

#[derive(Debug, Clone)]
pub struct UVData<T, S>
where
    T: Float + AbsDiffEq,
    S: Float + AbsDiffEq,
{
    pub meta: UVMeta,
    pub meta_arrays: ArrayMetaData,
    pub data_array: Option<Array<Complex<T>, Ix3>>,
    pub nsample_array: Option<Array<S, Ix3>>,
    pub flag_array: Option<Array<bool, Ix3>>,
}

impl<T, S> PartialEq for UVData<T, S>
where
    T: Float + AbsDiffEq + AbsDiffEq<T> + AbsDiffEq<Epsilon = T>,
    S: Float + AbsDiffEq + AbsDiffEq<S> + AbsDiffEq<Epsilon = S>,
{
    fn eq(&self, other: &UVData<T, S>) -> bool {
        match self.meta == other.meta {
            true => {}
            false => return false,
        }

        match self.meta_arrays == other.meta_arrays {
            true => {}
            false => return false,
        }

        match &self.data_array {
            Some(data1) => match &other.data_array {
                Some(data2) => match compare_complex_arrays(data1, data2) {
                    true => {}
                    false => return false,
                },

                None => return false,
            },
            None => match &other.data_array {
                None => {}
                Some(_) => return false,
            },
        }

        match &self.nsample_array {
            Some(nsample1) => match &other.nsample_array {
                Some(nsample2) => match nsample1.abs_diff_eq(nsample2, S::from(1e-6).unwrap()) {
                    true => {}
                    false => return false,
                },

                None => return false,
            },
            None => match &other.nsample_array {
                None => {}
                Some(_) => return false,
            },
        }

        match self.flag_array == other.flag_array {
            true => {}
            false => return false,
        }

        true
    }
}

impl<T, S> UVData<T, S>
where
    T: Float + AbsDiffEq,
    S: Float + AbsDiffEq,
{
    pub fn new(meta: UVMeta, metadata_only: bool) -> UVData<T, S> {
        let meta_arrays = ArrayMetaData::new(&meta);
        let (data_array, nsample_array, flag_array) = match metadata_only {
            true => (None, None, None),
            false => {
                let data_array = Some(Array::<Complex<T>, Ix3>::zeros((
                    meta.nblts as usize,
                    meta.ntimes as usize,
                    meta.npols as usize,
                )));
                let nsample_array = Some(Array::<S, Ix3>::zeros((
                    meta.nblts as usize,
                    meta.ntimes as usize,
                    meta.npols as usize,
                )));
                let flag_array = Some(Array::<bool, Ix3>::from_elem(
                    (
                        meta.nblts as usize,
                        meta.ntimes as usize,
                        meta.npols as usize,
                    ),
                    false,
                ));
                (data_array, nsample_array, flag_array)
            }
        };
        UVData::<T, S> {
            meta,
            meta_arrays,
            data_array,
            nsample_array,
            flag_array,
        }
    }

    pub fn telescope_location_latlonalt(&self) -> (f64, f64, f64) {
        utils::latlonalt_from_xyz(self.meta.telescope_location)
    }

    pub fn telescope_location_latlonalt_degrees(&self) -> (f64, f64, f64) {
        let lla: (f64, f64, f64) = utils::latlonalt_from_xyz(self.meta.telescope_location);
        (lla.0.to_degrees(), lla.1.to_degrees(), lla.2)
    }

    pub fn get_enu_antpos(&self) -> Array<f64, Ix2> {
        let (lat, lon, alt) = self.telescope_location_latlonalt_degrees();
        let tele_loc: Array<f64, Ix1> = Array::from_vec(self.meta.telescope_location.to_vec());
        let xyz: Array<f64, Ix2> = self.meta_arrays.antenna_positions.clone() + tele_loc;
        enu_from_ecef(&xyz, lat, lon, alt)
    }
}

impl From<UVMeta> for UVData<f64, f32> {
    fn from(meta: UVMeta) -> UVData<f64, f32> {
        UVData::<f64, f32>::new(meta, true)
    }
}

impl<T, S> From<(UVMeta, bool)> for UVData<T, S>
where
    T: Float + AbsDiffEq,
    S: Float + AbsDiffEq,
{
    fn from((meta, metadata_only): (UVMeta, bool)) -> UVData<T, S> {
        UVData::<T, S>::new(meta, metadata_only)
    }
}

impl<T, S> From<UVH5<T, S>> for UVData<T, S>
where
    T: Float + AsPrimitive<f64> + FromPrimitive + H5Type + AbsDiffEq,
    S: Float + H5Type + AbsDiffEq,
{
    fn from(uvh5: UVH5<T, S>) -> UVData<T, S> {
        UVData {
            meta: uvh5.meta,
            meta_arrays: uvh5.meta_arrays,
            data_array: uvh5.data_array,
            nsample_array: uvh5.nsample_array,
            flag_array: uvh5.flag_array,
        }
    }
}

impl<T, S> From<UVData<T, S>> for UVH5<T, S>
where
    T: Float + FromPrimitive + AsPrimitive<f64> + H5Type + AbsDiffEq,
    S: Float + H5Type + AbsDiffEq,
{
    fn from(uvd: UVData<T, S>) -> UVH5<T, S> {
        UVH5 {
            meta: uvd.meta,
            meta_arrays: uvd.meta_arrays,
            data_array: uvd.data_array,
            nsample_array: uvd.nsample_array,
            flag_array: uvd.flag_array,
        }
    }
}

impl<T, S> UVData<T, S>
where
    T: Float + AsPrimitive<f64> + FromPrimitive + H5Type + AbsDiffEq,
    S: Float + H5Type + AbsDiffEq,
{
    pub fn read_uvh5<P: AsRef<Path>>(path: P, read_data: bool) -> hdf5::Result<UVData<T, S>> {
        Ok(UVData::<T, S>::from(UVH5::<T, S>::from_file::<P>(
            path, read_data,
        )?))
    }

    pub fn write_uvh5<P: AsRef<Path>>(self, path: P, overwrite: bool) -> hdf5::Result<()> {
        UVH5::<T, S>::from(self).to_file::<P>(path, overwrite)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{compare_complex_arrays, UVData};
    use ndarray::{array, Array, Ix1, Ix2};
    use num_complex::Complex;
    use std::path::Path;

    #[test]
    fn test_complex_eq() {
        let array1: Array<Complex<f64>, Ix1> = array![
            Complex::<f64> {
                re: 1.0f64,
                im: 2.0f64
            },
            Complex::<f64> {
                re: 3.0f64,
                im: 4.0f64
            }
        ];

        assert!(compare_complex_arrays(&array1, &array1))
    }

    #[test]
    fn test_complex_neq() {
        let array1: Array<Complex<f64>, Ix1> = array![
            Complex::<f64> {
                re: 1.0f64,
                im: 2.0f64
            },
            Complex::<f64> {
                re: 3.0f64,
                im: 4.0f64
            }
        ];

        let array2: Array<Complex<f64>, Ix1> = array![
            Complex::<f64> {
                re: 7.0f64,
                im: 8.0f64
            },
            Complex::<f64> {
                re: 9.0f64,
                im: 10.0f64
            }
        ];

        assert!(!compare_complex_arrays(&array1, &array2))
    }

    #[test]
    fn enu_antpos() {
        let ref_antpos: Array<f64, Ix2> = array![
            [-105.03530155, -110.72205287, 0.9381712],
            [-90.42745888, -110.66626434, 0.92839577],
            [-75.81961648, -110.61047581, 0.91858693],
            [-112.38751909, -98.10314446, 0.78825341],
            [-97.77967573, -98.04735522, 0.71849469],
            [-83.17183304, -97.99156613, 0.65870255],
            [-68.56399298, -97.93578013, 0.79887699],
            [-119.73977458, -85.47423123, 0.64830222],
            [-105.13193283, -85.41844352, 0.66856021]
        ];
        let data_file =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/test_multiphase.uvh5");
        let uvd = UVData::<f64, f32>::read_uvh5(data_file, false).expect("Cannot read.");
        let enu = uvd.get_enu_antpos();
        assert!(enu.abs_diff_eq(&ref_antpos, 1e-6))
    }
}
