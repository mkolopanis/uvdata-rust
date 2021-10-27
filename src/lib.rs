#[cfg(test)]
#[macro_use]
extern crate approx;

use hdf5::H5Type;
use ndarray::{Array, Ix3};
use num_complex::Complex;
use num_traits::{cast::FromPrimitive, Float};
use std::path::Path;

mod base;
mod utils;
mod uvh5;

pub use self::uvh5::UVH5;

pub use self::base::{
    ArrayMetaData, BltOrder, BltOrders, CatTypes, Catalog, EqConvention, Orientation, PhaseType,
    SiderealVal, UVMeta, UnphasedVal, VisUnit,
};
pub use self::utils::{antnums_to_baseline, latlonalt_from_xyz, xyz_from_latlonalt};

#[derive(Debug, PartialEq, Clone)]
pub struct UVData<T, S>
where
    T: Float,
    S: Float,
{
    pub meta: UVMeta,
    pub meta_arrays: ArrayMetaData,
    pub data_array: Option<Array<Complex<T>, Ix3>>,
    pub nsample_array: Option<Array<S, Ix3>>,
    pub flag_array: Option<Array<bool, Ix3>>,
}

impl<T, S> UVData<T, S>
where
    T: Float,
    S: Float,
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
}

impl From<UVMeta> for UVData<f64, f32> {
    fn from(meta: UVMeta) -> UVData<f64, f32> {
        UVData::<f64, f32>::new(meta, true)
    }
}

impl<T, S> From<(UVMeta, bool)> for UVData<T, S>
where
    T: Float,
    S: Float,
{
    fn from((meta, metadata_only): (UVMeta, bool)) -> UVData<T, S> {
        UVData::<T, S>::new(meta, metadata_only)
    }
}

impl<T, S> From<UVH5<T, S>> for UVData<T, S>
where
    T: Float + FromPrimitive + H5Type,
    S: Float + H5Type,
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

impl<T, S> UVData<T, S>
where
    T: Float + FromPrimitive + H5Type,
    S: Float + H5Type,
{
    pub fn read_uvh5<P: AsRef<Path>>(path: P, read_data: bool) -> hdf5::Result<UVData<T, S>> {
        Ok(UVData::<T, S>::from(UVH5::<T, S>::from_file::<P>(
            path, read_data,
        )?))
    }
}
