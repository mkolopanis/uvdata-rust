use hdf5::{types::FixedAscii, H5Type};
use ndarray::{Array, Axis, Ix1, Ix2, Ix3, Ix4};
use num_complex::Complex;
use num_traits::{
    cast::{AsPrimitive, FromPrimitive},
    Float,
};
use std::{path::Path, str::FromStr};

use super::base::{
    ArrayMetaData, BltOrder, CatTypes, Catalog, EqConvention, Orientation, PhaseType, SiderealVal,
    UVMeta, UnphasedVal, VisUnit,
};
use super::utils;

const VERSION_STR: &str = env!("CARGO_PKG_VERSION");
fn print_version_str() -> String {
    format!("{}{}.", " Read/Written with uvdata-rust ", VERSION_STR)
}

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
struct Complexh5 {
    r: f64,
    i: f64,
}

const MAX_HIST_LENGTH: usize = 20_000;

impl<T: Float + AsPrimitive<f64>> From<Complex<T>> for Complexh5 {
    fn from(comp: Complex<T>) -> Self {
        Self {
            r: comp.re.as_(),
            i: comp.im.as_(),
        }
    }
}
impl<T: Float + FromPrimitive> From<Complexh5> for Complex<T> {
    fn from(comp: Complexh5) -> Self {
        Self {
            re: FromPrimitive::from_f64(comp.r).unwrap(),
            im: FromPrimitive::from_f64(comp.i).unwrap(),
        }
    }
}

fn read_scalar<T: hdf5::H5Type>(header: &hdf5::Group, param: &str) -> hdf5::Result<Option<T>> {
    match header.link_exists(param) {
        true => Ok(Some(header.dataset(param)?.read_scalar::<T>()?)),
        false => Ok(None),
    }
}

fn write_scalar<T: hdf5::H5Type>(group: &hdf5::Group, param: &str, val: &T) -> hdf5::Result<()> {
    group.new_dataset::<T>().create(param)?.write_scalar(val)
}

#[derive(Debug, PartialEq, Clone)]
pub struct UVH5<T, S>
where
    T: Float + FromPrimitive + AsPrimitive<f64> + H5Type,
    S: Float + H5Type,
{
    pub meta: UVMeta,
    pub meta_arrays: ArrayMetaData,
    pub data_array: Option<Array<Complex<T>, Ix3>>,
    pub nsample_array: Option<Array<S, Ix3>>,
    pub flag_array: Option<Array<bool, Ix3>>,
}

impl<T, S> UVH5<T, S>
where
    T: Float + AsPrimitive<f64> + FromPrimitive + H5Type,
    S: Float + H5Type,
{
    pub fn from_file<P: AsRef<Path>>(fname: P, read_data: bool) -> hdf5::Result<UVH5<T, S>> {
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

        let mut history: String = header
            .dataset("history")?
            .read_scalar::<FixedAscii<MAX_HIST_LENGTH>>()?
            .to_string();

        // append the version string if it is not already there.
        if !history
            .replace(" ", "")
            .replace("\n", "")
            .contains(&print_version_str().replace(" ", "").replace("\n", ""))
        {
            history.push_str(&print_version_str());
        }

        let vis_units: VisUnit = match header.link_exists(&"vis_units".to_string()) {
            true => VisUnit::from_str(
                &header
                    .dataset("vis_units")?
                    .read_scalar::<FixedAscii<200>>()?,
            )?,
            false => VisUnit::Uncalib,
        };

        let unknown: FixedAscii<200> = match FixedAscii::<200>::from_ascii("unknown") {
            Ok(text) => text,
            Err(_) => return Err("We're in trouble here".into()),
        };
        let dut1: Option<f32> = read_scalar::<f32>(&header, "dut1")?;
        let earth_omega: Option<f32> = read_scalar::<f32>(&header, "earth_omega")?;
        let gst0: Option<f32> = read_scalar::<f32>(&header, "gst0")?;
        let rdate: Option<String> =
            read_scalar::<FixedAscii<200>>(&header, "rdate")?.map(String::from);
        let timesys: Option<String> =
            read_scalar::<FixedAscii<200>>(&header, "timesys")?.map(String::from);

        let x_orientation: Orientation = Orientation::from_str(
            &read_scalar::<FixedAscii<200>>(&header, "x_orientation")?.unwrap_or(unknown),
        )?;

        let blt_order: BltOrder = BltOrder::from_str(
            &read_scalar::<FixedAscii<200>>(&header, "blt_order")?.unwrap_or(unknown),
        )?;

        let antenna_diameters: Option<Array<f32, Ix1>> =
            match header.link_exists(&"antenna_diameters".to_string()) {
                true => Some(header.dataset("antenna_diameters")?.read::<f32, Ix1>()?),
                false => None,
            };
        let uvplane_reference_time: Option<i32> =
            read_scalar::<i32>(&header, "uvplane_reference_time")?;

        let eq_coeffs: Option<Array<f32, Ix2>> = match header.link_exists(&"eq_coeffs".to_string())
        {
            true => Some(header.dataset("eq_coeffs")?.read::<f32, Ix2>()?),
            false => None,
        };

        let eq_coeffs_convention: EqConvention = EqConvention::from_str(
            &read_scalar::<FixedAscii<200>>(&header, "eq_coeffs_convention")?.unwrap_or(unknown),
        )?;

        let phase_type: PhaseType = PhaseType::from_str(
            &read_scalar::<FixedAscii<200>>(&header, "phase_type")?.unwrap_or(unknown),
        )?;

        let object_name: String = match read_scalar::<FixedAscii<200>>(&header, "object_name") {
            Ok(name) => name.unwrap_or(unknown).to_lowercase(),
            Err(_) => "unknown".to_string(),
        };

        let nants_data: u32 = read_scalar::<u32>(&header, "Nants_data")?.unwrap();

        let nants_telescope: u32 = read_scalar::<u32>(&header, "Nants_telescope")?.unwrap();

        let nblts: u32 = read_scalar::<u32>(&header, "Nblts")?.unwrap();
        let nspws: u32 = read_scalar::<u32>(&header, "Nspws")?.unwrap();
        let npols: u8 = read_scalar::<u8>(&header, "Npols")?.unwrap();
        let ntimes: u32 = read_scalar::<u32>(&header, "Ntimes")?.unwrap();
        let nfreqs: u32 = read_scalar::<u32>(&header, "Nfreqs")?.unwrap();
        let nphases: u32 = read_scalar::<u32>(&header, "Nphases")?.unwrap_or(1);

        // compute Nbls
        let ant_1_array: Array<u32, Ix1> = header.dataset("ant_1_array")?.read::<u32, Ix1>()?;
        let ant_2_array: Array<u32, Ix1> = header.dataset("ant_2_array")?.read::<u32, Ix1>()?;
        let baseline_array: Array<u32, Ix1> =
            utils::antnums_to_baseline(&ant_1_array, &ant_2_array, false);
        let nbls = baseline_array
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len() as u32;

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
            object_name: object_name.clone(),
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

        let antenna_names: Array<String, Ix1> = header
            .dataset("antenna_names")?
            .read::<FixedAscii<50>, Ix1>()?
            .mapv(|x| x.into());
        let freq_dset = header.dataset("freq_array")?;
        let freq_array: Array<f64, Ix1> = match freq_dset.ndim() {
            1 => freq_dset.read::<f64, Ix1>()?,
            2 => {
                // need to squeeze out the spw axis
                // we have defined uvdata to only work
                // with future array shapes
                freq_dset.read::<f64, Ix2>()?.remove_axis(Axis(0))
            }
            ndim => return Err(format!("Incompatible dimensions of freq array: {:}", ndim).into()),
        };

        let spw_id_array: Array<u32, Ix1> =
            match header.link_exists(&"flex_spw_id_array".to_string()) {
                true => header.dataset("flex_spw_id_array")?.read::<u32, Ix1>()?,
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

        let (phase_center_catalog, phase_center_id_array) = match header
            .link_exists("phase_center_catalog")
        {
            true => {
                let phase_group: hdf5::Group = header.group("phase_center_catalog")?;
                let phase_names: Vec<String> = phase_group.member_names()?;
                let mut cat: Catalog = Catalog::new();
                for name in phase_names {
                    let json_str = phase_group
                        .dataset(name.as_str())?
                        .read_scalar::<FixedAscii<20_000>>()?;

                    let cat_val: CatTypes = match serde_json::from_str(json_str.as_str()) {
                        Ok(CatTypes::Unphased(val)) => CatTypes::Unphased(val),
                        Ok(CatTypes::Sidereal(val)) => CatTypes::Sidereal(val),
                        Ok(CatTypes::Ephem(val)) => CatTypes::Ephem(val),
                        Err(err) => return Err(format!("Json Err {}", err).into()),
                    };
                    cat.insert(name, cat_val);
                }
                let id_array = header
                    .dataset("phase_center_id_array")?
                    .read::<u32, Ix1>()?;
                (cat, id_array)
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
                        let cat_frame: String =
                            match read_scalar::<FixedAscii<200>>(&header, "phase_center_frame") {
                                Ok(name) => name.unwrap_or(unknown).to_lowercase(),
                                Err(_) => "unknown".to_string(),
                            };
                        cat.insert(
                            object_name,
                            CatTypes::Sidereal(SiderealVal {
                                cat_id: 0,
                                cat_type: "sidereal".to_string(),
                                cat_lon: read_scalar::<f64>(&header, "phase_center_ra")?.unwrap(),
                                cat_lat: read_scalar::<f64>(&header, "phase_center_dec")?.unwrap(),
                                cat_frame,
                                cat_epoch: read_scalar::<f64>(&header, "phase_center_epoch")?
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
                (cat, Array::<u32, Ix1>::zeros(meta.nblts as usize))
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
            antenna_names,
            antenna_positions,
            eq_coeffs,
            antenna_diameters,
            phase_center_catalog,
            phase_center_id_array,
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
                        let data: Array<Complex<T>, Ix3> =
                            visdata.read::<Complexh5, Ix3>()?.mapv(|x| x.into());
                        let flags: Array<bool, Ix3> = flagdata.read::<bool, Ix3>()?;
                        let samps: Array<S, Ix3> = nsampledata.read::<S, Ix3>()?;
                        (Some(data), Some(samps), Some(flags))
                    }
                    4 => {
                        // need to squeeze out the spw axis
                        // we have defined uvdata to only work
                        // with future array shapes
                        let data: Array<Complex<T>, Ix3> = visdata
                            .read::<Complexh5, Ix4>()?
                            .remove_axis(Axis(1))
                            .mapv(|x| x.into());
                        let flags: Array<bool, Ix3> =
                            flagdata.read::<bool, Ix4>()?.remove_axis(Axis(1));
                        let samps: Array<S, Ix3> =
                            nsampledata.read::<S, Ix4>()?.remove_axis(Axis(1));
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
        h5file.close()?;
        let uvh5 = UVH5::<T, S> {
            meta,
            meta_arrays,
            data_array,
            nsample_array,
            flag_array,
        };

        Ok(uvh5)
    }
    pub fn to_file<P: AsRef<Path>>(self, fname: P, overwrite: bool) -> hdf5::Result<()> {
        match self.data_array {
            Some(_) => {}
            None => return Err("Unable to write metadata only objects to UVH5 files.".into()),
        }
        let h5file: hdf5::File = match overwrite {
            true => hdf5::File::create(fname)?,
            false => hdf5::File::create_excl(fname)?,
        };

        let header = h5file.create_group("/Header")?;

        // write out all the fields of meta
        write_scalar::<u32>(&header, "Nblts", &self.meta.nblts)?;
        write_scalar::<u32>(&header, "Nspws", &self.meta.nspws)?;
        write_scalar::<u8>(&header, "Npols", &self.meta.npols)?;
        write_scalar::<u32>(&header, "Ntimes", &self.meta.ntimes)?;
        write_scalar::<u32>(&header, "Nfreqs", &self.meta.nfreqs)?;
        // handle nphases in a bit
        // write_scalar::<u32>(&header, "Nbls", &self.meta.nbls)?;
        write_scalar::<u32>(&header, "Nants_data", &self.meta.nants_data)?;

        // only write out blt_order if it is known
        match self.meta.blt_order.to_string().as_ref() {
            "unknown, unknown" => {}
            order => write_scalar::<FixedAscii<20>>(
                &header,
                "blt_order",
                &FixedAscii::<20>::from_ascii(order).expect("Unable to write blt_order"),
            )?,
        };

        write_scalar::<u32>(&header, "Nants_telescope", &self.meta.nants_telescope)?;

        write_scalar::<FixedAscii<7>>(
            &header,
            "vis_units",
            &FixedAscii::<7>::from_ascii(&self.meta.vis_units.to_string().to_lowercase())
                .expect("Unable to write vis_units"),
        )?;

        write_scalar::<FixedAscii<5>>(
            &header,
            "x_orientation",
            &FixedAscii::<5>::from_ascii(&self.meta.x_orientation.to_string().to_lowercase())
                .expect("Unable to write x_orientation"),
        )?;

        write_scalar::<FixedAscii<200>>(
            &header,
            "instrument",
            &FixedAscii::<200>::from_ascii(&self.meta.instrument)
                .expect("Unable to write instrument"),
        )?;

        write_scalar::<FixedAscii<200>>(
            &header,
            "telescope_name",
            &FixedAscii::<200>::from_ascii(&self.meta.telescope_name)
                .expect("Unable to write telescope_name"),
        )?;

        let (latitude, longitude, altitude) =
            utils::latlonalt_from_xyz(self.meta.telescope_location);

        write_scalar::<f64>(&header, "latitude", &latitude.to_degrees())?;
        write_scalar::<f64>(&header, "longitude", &longitude.to_degrees())?;
        write_scalar::<f64>(&header, "altitude", &altitude)?;

        write_scalar::<FixedAscii<200>>(
            &header,
            "object_name",
            &FixedAscii::<200>::from_ascii(&self.meta.object_name)
                .expect("Unable to write object_name"),
        )?;

        // only write out eq_coeffs_conventionf if it is known
        match self
            .meta
            .eq_coeffs_convention
            .to_string()
            .to_lowercase()
            .as_ref()
        {
            "unknown" => {}
            conv => write_scalar::<FixedAscii<8>>(
                &header,
                "eq_coeffs_convention",
                &FixedAscii::<8>::from_ascii(conv).expect("Unable to write eq_coeffs_convention"),
            )?,
        };

        if let Some(dut1) = self.meta.dut1 {
            write_scalar::<f32>(&header, "dut1", &dut1).expect("Unable to write dut1");
        }

        if let Some(gst0) = self.meta.gst0 {
            write_scalar::<f32>(&header, "gst0", &gst0).expect("Unable to write gst0");
        }

        if let Some(rdate) = self.meta.rdate {
            write_scalar::<FixedAscii<200>>(
                &header,
                "rdate",
                &FixedAscii::<200>::from_ascii(&rdate).expect("Unable to write rdate"),
            )
            .expect("Unable to write rdate");
        }
        if let Some(earth_omega) = self.meta.earth_omega {
            write_scalar::<f32>(&header, "earth_omega", &earth_omega)
                .expect("Unable to write earth_omega");
        }
        if let Some(timesys) = self.meta.timesys {
            write_scalar::<FixedAscii<200>>(
                &header,
                "timesys",
                &FixedAscii::<200>::from_ascii(&timesys).expect("Unable to write timesys"),
            )
            .expect("Unable to write timesys.");
        };
        if let Some(ref_time) = self.meta.uvplane_reference_time {
            write_scalar::<i32>(&header, "uvplane_reference_time", &ref_time)
                .expect("Unable to write uvplane_reference_time");
        }

        let mut hist_out = self.meta.history.clone();
        // append the version string if it is not already there.
        if !hist_out
            .replace(" ", "")
            .replace("\n", "")
            .contains(&print_version_str().replace(" ", "").replace("\n", ""))
        {
            hist_out.push_str(&print_version_str());
        }

        write_scalar::<FixedAscii<MAX_HIST_LENGTH>>(
            &header,
            "history",
            &FixedAscii::<MAX_HIST_LENGTH>::from_ascii(&hist_out).expect("Unable to write history"),
        )?;

        // write out fields of meta_arrays

        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.spw_array)
            .create("spw_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.uvw_array)
            .create("uvw_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.time_array)
            .create("time_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.lst_array)
            .create("lst_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.ant_1_array)
            .create("ant_1_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.ant_2_array)
            .create("ant_2_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.freq_array)
            .create("freq_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.spw_id_array)
            .create("flex_spw_id_array")?;

        write_scalar(&header, "flex_spw", &true)?;

        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.polarization_array)
            .create("polarization_array")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.integration_time)
            .create("integration_time")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.channel_width)
            .create("channel_width")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.antenna_numbers)
            .create("antenna_numbers")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.antenna_names.mapv(|val| {
                FixedAscii::<50>::from_ascii(&val).expect("Unable to write antenna_names")
            }))
            .create("antenna_names")?;
        header
            .new_dataset_builder()
            .with_data(&self.meta_arrays.antenna_positions)
            .create("antenna_positions")?;

        if let Some(eq_coeffs) = self.meta_arrays.eq_coeffs {
            header
                .new_dataset_builder()
                .with_data(&eq_coeffs)
                .create("eq_coeffs")
                .expect("Unable to write equalization coefficients.");
        }

        if let Some(ant_diams) = self.meta_arrays.antenna_diameters {
            header
                .new_dataset_builder()
                .with_data(&ant_diams)
                .create("antenna_diameters")
                .expect("Unable to write antenna_diameters.");
        }

        match self.meta.nphases {
            1 => {
                match self.meta_arrays.phase_center_catalog.into_iter().next() {
                    Some((_, CatTypes::Unphased(_))) => {
                        write_scalar::<FixedAscii<6>>(
                            &header,
                            "phase_type",
                            &FixedAscii::<6>::from_ascii(&"drift")
                                .expect("Unable to write phase_type"),
                        )?;
                    }
                    Some((_, CatTypes::Sidereal(catalog))) => {
                        write_scalar::<FixedAscii<6>>(
                            &header,
                            "phase_type",
                            &FixedAscii::<6>::from_ascii(
                                &self.meta.phase_type.to_string().to_lowercase(),
                            )
                            .expect("Unable to write phase_type"),
                        )?;
                        write_scalar::<FixedAscii<200>>(
                            &header,
                            "phase_center_frame",
                            &FixedAscii::<200>::from_ascii(&catalog.cat_frame.to_lowercase())
                                .expect("Cannot convert phase type to ascii."),
                        )
                        .expect("Cannot write out phase_center_frame.");
                        write_scalar(&header, "phase_center_ra", &catalog.cat_lat)?;
                        write_scalar(&header, "phase_center_dec", &catalog.cat_lon)?;
                        write_scalar(&header, "phase_center_epoch", &catalog.cat_epoch)?;
                        // need to calculate some things here, app_ra, app_dec, phase_center_frame_pa
                        // catalog.cat_pm_ra.map(|val|  write_scalar(&header, "phase_center_frame_pa", &catalog.cat_epoch)?)
                    }
                    Some((name, CatTypes::Ephem(catalog))) => {
                        write_scalar::<u32>(&header, "Nphase", &1)?;
                        let cat_group = header.create_group("phase_center_catalog")?;
                        let dumped_val = FixedAscii::<MAX_HIST_LENGTH>::from_ascii(
                            &serde_json::to_string(&CatTypes::Ephem(catalog))
                                .expect("Cannot convert catalog value to string."),
                        )
                        .expect("Unable to write out catalog values.");
                        write_scalar(&cat_group, &name, &dumped_val)?
                    }
                    other => {
                        return Err(format!("Invalid phase center catalog entry {:?}", other).into())
                    }
                }
            }
            val => {
                write_scalar::<FixedAscii<6>>(
                    &header,
                    "phase_type",
                    &FixedAscii::<6>::from_ascii(&self.meta.phase_type.to_string().to_lowercase())
                        .expect("Unable to write phase_type"),
                )?;
                // handle the catalog
                write_scalar::<u32>(&header, "Nphase", &val)?;
                let cat_group = header.create_group("phase_center_catalog")?;
                for (name, catval) in self.meta_arrays.phase_center_catalog.iter() {
                    let dumped_val = FixedAscii::<MAX_HIST_LENGTH>::from_ascii(
                        &serde_json::to_string(catval)
                            .expect("Cannot convert catalog value to string."),
                    )
                    .expect("Unable to write out catalog values.");
                    write_scalar(&cat_group, name, &dumped_val)?
                }
                header
                    .new_dataset_builder()
                    .with_data(&self.meta_arrays.phase_center_id_array)
                    .create("phase_center_id_array")?;
            }
        };

        let dgroup = h5file.create_group("/Data")?;

        let h5_data: Array<Complexh5, Ix3> = self.data_array.unwrap().mapv(|x| x.into());

        dgroup
            .new_dataset_builder()
            .with_data(&h5_data)
            .create("visdata")?;

        dgroup
            .new_dataset_builder()
            .with_data(&self.flag_array.unwrap())
            .lzf()
            .create("flags")?;

        dgroup
            .new_dataset_builder()
            .with_data(&self.nsample_array.unwrap())
            .lzf()
            .create("nsamples")?;

        h5file.close()?;

        Ok(())
    }
}
