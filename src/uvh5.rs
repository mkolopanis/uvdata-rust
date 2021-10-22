use hdf5;
use hdf5::types::FixedAscii;
use ndarray::{s, Array, Ix1, Ix2, Ix3, Ix4};
use num::Complex;
use std::path::Path;

use super::base::{
    ArrayMetaData, BltOrder, BltOrders, CatTypes, Catalog, EqConvention, Orientation, PhaseType,
    SiderealVal, UVMeta, UnphasedVal, VisUnit,
};
use super::utils;

#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Complexh5 {
    r: f64,
    i: f64,
}

const MAX_HIST_LENGTH: usize = 20_000;

impl Into<Complex<f64>> for Complexh5 {
    fn into(self) -> Complex<f64> {
        Complex::<f64>::new(self.r.into(), self.i.into())
    }
}

impl Into<Complex<f32>> for Complexh5 {
    fn into(self) -> Complex<f32> {
        Complex::<f32>::new(self.r as f32, self.i as f32)
    }
}

fn read_scalar<T: hdf5::H5Type>(header: &hdf5::Group, param: String) -> hdf5::Result<Option<T>> {
    match header.link_exists(&param.to_string()) {
        true => Ok(Some(header.dataset(param.as_str())?.read_scalar::<T>()?)),
        false => Ok(None),
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct UVH5 {
    pub meta: UVMeta,
    pub meta_arrays: ArrayMetaData,
    pub data_array: Option<Array<Complex<f64>, Ix3>>,
    pub nsample_array: Option<Array<f32, Ix3>>,
    pub flag_array: Option<Array<bool, Ix3>>,
}

impl UVH5 {
    pub fn from_file<P: AsRef<Path>>(fname: P, read_data: bool) -> hdf5::Result<UVH5> {
        let h5file = hdf5::File::open(fname)?;

        // read metadata
        let header = h5file.group("/Header")?;
        let lat = header.dataset("latitude")?.read_scalar::<f64>()?;
        let lon = header.dataset("longitude")?.read_scalar::<f64>()?;
        let alt = header.dataset("altitude")?.read_scalar::<f64>()?;
        let telescope_location = utils::xyz_from_latlonalt::<f64>(lat, lon, alt);

        let instrument = header
            .dataset("instrument")?
            .read_scalar::<FixedAscii<200>>()?
            .to_string();
        let telescope_name = header
            .dataset("telescope_name")?
            .read_scalar::<FixedAscii<200>>()?
            .to_string();

        let history: String = header
            .dataset("history")?
            .read_scalar::<FixedAscii<MAX_HIST_LENGTH>>()?
            .to_string();

        let vis_units: VisUnit = match header.link_exists(&"vis_units".to_string()) {
            true => match header
                .dataset("vis_units")?
                .read_scalar::<FixedAscii<200>>()?
                .to_lowercase()
                .as_str()
            {
                "uncalib" => VisUnit::Uncalib,
                "jy" => VisUnit::Jansky,
                "k str" => VisUnit::Kelvinstr,
                unit => return Err(format!("Unknown Vis Unit: {:}", unit).into()),
            },
            false => VisUnit::Uncalib,
        };

        let unknown: FixedAscii<200> = match FixedAscii::<200>::from_ascii("unknown") {
            Ok(text) => text,
            Err(_) => return Err("We're in trouble here".into()),
        };
        let dut1: Option<f32> = read_scalar::<f32>(&header, "dut1".to_string())?;
        let earth_omega: Option<f32> = read_scalar::<f32>(&header, "earth_omega".to_string())?;
        let gst0: Option<f32> = read_scalar::<f32>(&header, "gst0".to_string())?;
        let rdate: Option<String> =
            match read_scalar::<FixedAscii<200>>(&header, "rdate".to_string())? {
                Some(val) => Some(String::from(val)),
                None => None,
            };
        let timesys: Option<String> =
            match read_scalar::<FixedAscii<200>>(&header, "timesys".to_string())? {
                Some(val) => Some(String::from(val)),
                None => None,
            };

        let x_orientation: Option<FixedAscii<200>> =
            read_scalar::<FixedAscii<200>>(&header, "x_orientation".to_string())?;

        let x_orientation: Orientation =
            match x_orientation.unwrap_or(unknown).to_lowercase().as_str() {
                "east" => Orientation::East,
                "north" => Orientation::North,
                _ => Orientation::Unknown,
            };

        let blt_order: Option<FixedAscii<200>> =
            read_scalar::<FixedAscii<200>>(&header, "blt_order".to_string())?;

        let blt_order: BltOrder = match blt_order.unwrap_or(unknown).to_lowercase().as_str() {
            "bda, " => BltOrder {
                major: BltOrders::Bda,
                minor: BltOrders::Bda,
            },
            "baseline, time" => BltOrder {
                major: BltOrders::Baseline,
                minor: BltOrders::Time,
            },
            "baseline, ant1" => BltOrder {
                major: BltOrders::Baseline,
                minor: BltOrders::Ant1,
            },
            "baseline, ant2" => BltOrder {
                major: BltOrders::Baseline,
                minor: BltOrders::Ant2,
            },
            "time, baseline" => BltOrder {
                major: BltOrders::Time,
                minor: BltOrders::Baseline,
            },
            "time, ant1" => BltOrder {
                major: BltOrders::Time,
                minor: BltOrders::Ant1,
            },
            "time, ant2" => BltOrder {
                major: BltOrders::Time,
                minor: BltOrders::Ant2,
            },
            "ant1, ant2" => BltOrder {
                major: BltOrders::Ant1,
                minor: BltOrders::Ant2,
            },
            "ant1, time" => BltOrder {
                major: BltOrders::Ant1,
                minor: BltOrders::Time,
            },
            "ant1, baseline" => BltOrder {
                major: BltOrders::Ant1,
                minor: BltOrders::Baseline,
            },
            "ant2, ant1" => BltOrder {
                major: BltOrders::Ant2,
                minor: BltOrders::Ant1,
            },
            "ant2, time" => BltOrder {
                major: BltOrders::Ant2,
                minor: BltOrders::Time,
            },
            "ant2, baseline" => BltOrder {
                major: BltOrders::Ant2,
                minor: BltOrders::Baseline,
            },
            "unknown" => BltOrder {
                major: BltOrders::Unknown,
                minor: BltOrders::Unknown,
            },
            _ => return Err("Found unknown blt ordering".into()),
        };

        let antenna_diameters: Option<Array<f32, Ix1>> =
            match header.link_exists(&"antenna_diameters".to_string()) {
                true => Some(header.dataset("antenna_diameters")?.read::<f32, Ix1>()?),
                false => None,
            };
        let uvplane_reference_time: Option<i32> =
            read_scalar::<i32>(&header, "uvplane_reference_time".to_string())?;

        let eq_coeffs: Option<Array<f32, Ix2>> = match header.link_exists(&"eq_coeffs".to_string())
        {
            true => Some(header.dataset("eq_coeffs")?.read::<f32, Ix2>()?),
            false => None,
        };

        let eq_coeffs_convention: Option<FixedAscii<200>> =
            read_scalar::<FixedAscii<200>>(&header, "eq_coeffs_convention".to_string())?;

        let eq_coeffs_convention: EqConvention = match eq_coeffs_convention
            .unwrap_or(unknown)
            .to_lowercase()
            .as_str()
        {
            "divide" => EqConvention::Divide,
            "multiply" => EqConvention::Multiply,
            _ => EqConvention::Unknown,
        };

        let phase_type: Option<FixedAscii<200>> =
            read_scalar::<FixedAscii<200>>(&header, "phase_type".to_string())?;

        let phase_type: PhaseType = match phase_type.unwrap_or(unknown).to_lowercase().as_str() {
            "drift" => PhaseType::Drift,
            "phased" => PhaseType::Phased,
            _ => PhaseType::Drift,
        };

        let object_name: String =
            match read_scalar::<FixedAscii<200>>(&header, "object_name".to_string()) {
                Ok(name) => name.unwrap_or(unknown).to_lowercase().to_string(),
                Err(_) => "unknown".to_string(),
            };

        let nants_data: u32 = read_scalar::<u32>(&header, "Nants_data".to_string())?.unwrap();

        let nants_telescope: u32 =
            read_scalar::<u32>(&header, "Nants_telescope".to_string())?.unwrap();
        let nbls: u32 = read_scalar::<u32>(&header, "Nbls".to_string())?.unwrap();
        let nblts: u32 = read_scalar::<u32>(&header, "Nblts".to_string())?.unwrap();
        let nspws: u32 = read_scalar::<u32>(&header, "Nblts".to_string())?.unwrap();
        let npols: u8 = read_scalar::<u8>(&header, "Npols".to_string())?.unwrap();
        let ntimes: u32 = read_scalar::<u32>(&header, "Ntimes".to_string())?.unwrap();
        let nfreqs: u32 = read_scalar::<u32>(&header, "Nfreqs".to_string())?.unwrap();
        let nphases: u32 = match read_scalar::<u32>(&header, "Nphases".to_string())? {
            Some(int) => int,
            None => 1,
        };

        let meta = UVMeta {
            nbls,
            nblts,
            nspws,
            npols,
            ntimes,
            nfreqs,
            nphases,
            nants_data,
            blt_order,
            vis_units,
            nants_telescope,
            phase_type,
            x_orientation,
            instrument,
            telescope_name,
            telescope_location,
            eq_coeffs_convention,
            dut1,
            gst0,
            rdate,
            earth_omega,
            timesys,
            uvplane_reference_time,
            history,
        };
        // read all the meta arrays

        let spw_array: Array<u32, Ix1> = header.dataset("spw_array")?.read::<u32, Ix1>()?;
        let uvw_array: Array<f64, Ix2> = header.dataset("uvw_array")?.read::<f64, Ix2>()?;
        let time_array: Array<f64, Ix1> = header.dataset("time_array")?.read::<f64, Ix1>()?;
        let lst_array: Array<f64, Ix1> = header.dataset("lst_array")?.read::<f64, Ix1>()?;
        let ant_1_array: Array<u32, Ix1> = header.dataset("ant_1_array")?.read::<u32, Ix1>()?;
        let ant_2_array: Array<u32, Ix1> = header.dataset("ant_2_array")?.read::<u32, Ix1>()?;
        let baseline_array: Array<u32, Ix1> =
            utils::antnums_to_baseline(&ant_1_array, &ant_2_array, false);
        let freq_dset = header.dataset("freq_array")?;
        let freq_array: Array<f64, Ix1> = match freq_dset.ndim() {
            1 => freq_dset.read::<f64, Ix1>()?,
            2 => {
                // need to squeeze out the spw axis
                // we have defined uvdata to only work
                // with future array shapes
                freq_dset.read::<f64, Ix2>()?.slice_move(s![0, ..])
            }
            ndim => return Err(format!("Incompatible dimensions of freq array: {:}", ndim).into()),
        };

        let spw_id_array: Array<u32, Ix1> = match header.link_exists(&"spw_id_array".to_string()) {
            true => header.dataset("spw_id_array")?.read::<u32, Ix1>()?,
            false => Array::<u32, Ix1>::zeros(meta.nfreqs as usize),
        };
        let polarization_array: Array<i8, Ix1> =
            header.dataset("polarization_array")?.read::<i8, Ix1>()?;
        let integration_time: Array<f64, Ix1> =
            header.dataset("integration_time")?.read::<f64, Ix1>()?;

        let cwidth_dset = header.dataset("channel_width")?;
        let channel_width: Array<f64, Ix1> = match cwidth_dset.ndim() {
            0 => Array::<f64, Ix1>::from_elem(
                meta.nfreqs as usize,
                cwidth_dset.read_scalar::<f64>()?,
            ),
            1 => cwidth_dset.read::<f64, Ix1>()?,
            ndim => {
                return Err(format!("Incompatible dimensions of Channel width: {:}", ndim).into())
            }
        };

        let antenna_numbers: Array<u32, Ix1> =
            header.dataset("antenna_numbers")?.read::<u32, Ix1>()?;
        let antenna_positions: Array<f64, Ix2> =
            header.dataset("antenna_positions")?.read::<f64, Ix2>()?;
        let phase_center_catalog = match header.link_exists("phase_center_catalog") {
            true => {
                let phase_group: hdf5::Group = header.group("phase_center_catalog")?;
                let phase_names: Vec<String> = phase_group.member_names()?;
                let mut cat: Catalog = Catalog::new();
                for name in phase_names {
                    let json_str = phase_group
                        .dataset(&name.as_str())?
                        .read_scalar::<FixedAscii<20_000>>()?;

                    let cat_val: CatTypes = match serde_json::from_str(json_str.as_str()) {
                        Ok(CatTypes::Unphased(val)) => CatTypes::Unphased(val),
                        Ok(CatTypes::Sidereal(val)) => CatTypes::Sidereal(val),
                        Ok(CatTypes::Ephem(val)) => CatTypes::Ephem(val),
                        Err(err) => return Err(format!("Json Err {}", err).into()),
                    };
                    cat.insert(name.into(), cat_val);
                }
                cat
            }
            false => {
                // if not multi-phased deal with each phase and
                // add into catalog
                let mut cat = Catalog::new();
                match phase_type {
                    PhaseType::Drift => {
                        cat.insert(
                            "zenith".to_string(),
                            CatTypes::Unphased(UnphasedVal {
                                cat_id: 0,
                                cat_type: "unphased".to_string(),
                            }),
                        );
                    }
                    PhaseType::Phased => {
                        let cat_frame: String = match read_scalar::<FixedAscii<200>>(
                            &header,
                            "phase_center_frame".to_string(),
                        ) {
                            Ok(name) => name.unwrap_or(unknown).to_lowercase().to_string(),
                            Err(_) => "unknown".to_string(),
                        };
                        cat.insert(
                            object_name,
                            CatTypes::Sidereal(SiderealVal {
                                cat_id: 0,
                                cat_type: "sidereal".to_string(),
                                cat_lon: read_scalar::<f64>(
                                    &header,
                                    "phase_center_ra".to_string(),
                                )?
                                .unwrap(),
                                cat_lat: read_scalar::<f64>(
                                    &header,
                                    "phase_center_dec".to_string(),
                                )?
                                .unwrap(),
                                cat_frame,
                                cat_epoch: read_scalar::<f64>(
                                    &header,
                                    "phase_center_epoch".to_string(),
                                )?
                                .unwrap(),
                                cat_pm_ra: None,
                                cat_pm_dec: None,
                                cat_dist: None,
                                cat_vrad: None,
                                info_source: Some("UVData".to_string()),
                            }),
                        );
                    }
                    _ => (),
                }
                cat
            }
        };
        let meta_arrays = ArrayMetaData {
            spw_array,
            uvw_array,
            time_array,
            lst_array,
            ant_1_array,
            ant_2_array,
            baseline_array,
            freq_array,
            spw_id_array,
            polarization_array,
            integration_time,
            channel_width,
            antenna_numbers,
            antenna_positions,
            eq_coeffs,
            antenna_diameters,
            phase_center_catalog,
        };
        // optional data read
        let (data_array, nsample_array, flag_array) = match read_data {
            true => {
                let dgroup = h5file.group("/Data")?;
                let visdata = dgroup.dataset("visdata")?;
                let flagdata = dgroup.dataset("flags")?;
                let nsampledata = dgroup.dataset("nsamples")?;
                match visdata.ndim() {
                    3 => {
                        let data: Array<Complex<f64>, Ix3> =
                            visdata.read::<Complexh5, Ix3>()?.mapv(|x| x.into());
                        let flags: Array<bool, Ix3> = flagdata.read::<bool, Ix3>()?;
                        let samps: Array<f32, Ix3> = nsampledata.read::<f32, Ix3>()?;
                        (Some(data), Some(samps), Some(flags))
                    }
                    4 => {
                        // need to squeeze out the spw axis
                        // we have defined uvdata to only work
                        // with future array shapes
                        let data: Array<Complex<f64>, Ix3> = visdata
                            .read::<Complexh5, Ix4>()?
                            .slice_move(s![.., 0, .., ..])
                            .mapv(|x| x.into());
                        let flags: Array<bool, Ix3> =
                            flagdata.read::<bool, Ix4>()?.slice_move(s![.., 0, .., ..]);
                        let samps: Array<f32, Ix3> = nsampledata
                            .read::<f32, Ix4>()?
                            .slice_move(s![.., 0, .., ..]);
                        (Some(data), Some(samps), Some(flags))
                    }
                    ndim => {
                        return Err(
                            format!("Incompatible dimensions of data array: {:}", ndim).into()
                        )
                    }
                }
            }
            false => (None, None, None),
        };
        h5file.close();
        let uvh5 = UVH5 {
            meta,
            meta_arrays,
            data_array,
            nsample_array,
            flag_array,
        };
        // let uvh5
        Ok(uvh5)
    }
}
