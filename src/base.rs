use approx::abs_diff_eq;
use ndarray::{Array, Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, str::FromStr};

// TODO: make and enum of the different catalog types
// and catalog structs themselves probably too
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct UnphasedVal {
    pub cat_id: u32,
    pub cat_type: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SiderealVal {
    pub cat_id: u32,
    pub cat_type: String,
    pub cat_lon: f64,
    pub cat_lat: f64,
    pub cat_frame: String,
    pub cat_epoch: f64,
    pub cat_pm_ra: Option<f64>,
    pub cat_pm_dec: Option<f64>,
    pub cat_dist: Option<f64>,
    pub cat_vrad: Option<f64>,
    pub info_source: Option<String>,
}

impl PartialEq<SiderealVal> for SiderealVal {
    fn eq(&self, other: &SiderealVal) -> bool {
        match self.cat_id == other.cat_id {
            true => {}
            false => return false,
        }

        match self.cat_type == other.cat_type {
            true => {}
            false => return false,
        }

        match abs_diff_eq!(self.cat_lon, other.cat_lon, epsilon = 1e-6) {
            true => {}
            false => return false,
        }

        match abs_diff_eq!(self.cat_lat, other.cat_lat, epsilon = 1e-6) {
            true => {}
            false => return false,
        }

        match self.cat_frame == other.cat_frame {
            true => {}
            false => return false,
        }

        match abs_diff_eq!(self.cat_epoch, other.cat_epoch, epsilon = 1e-6) {
            true => {}
            false => return false,
        }

        match &self.cat_pm_ra {
            Some(pm_ra1) => match &other.cat_pm_ra {
                Some(pm_ra2) => match abs_diff_eq!(pm_ra1, pm_ra2, epsilon = 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.cat_pm_ra {
                None => {}
                Some(_) => return false,
            },
        }

        match &self.cat_pm_dec {
            Some(pm_dec1) => match &other.cat_pm_dec {
                Some(pm_dec2) => match abs_diff_eq!(pm_dec1, pm_dec2, epsilon = 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.cat_pm_dec {
                None => {}
                Some(_) => return false,
            },
        }

        match &self.cat_dist {
            Some(dist1) => match &other.cat_dist {
                Some(dist2) => match abs_diff_eq!(dist1, dist2, epsilon = 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.cat_dist {
                None => {}
                Some(_) => return false,
            },
        }

        match &self.cat_vrad {
            Some(vrad1) => match &other.cat_vrad {
                Some(vrad2) => match abs_diff_eq!(vrad1, vrad2, epsilon = 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.cat_vrad {
                None => {}
                Some(_) => return false,
            },
        }

        match self.info_source == other.info_source {
            true => {}
            false => return false,
        }
        true
    }
}
impl Eq for SiderealVal {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EphemVal {
    pub cat_id: u32,
    pub cat_type: String,
    pub cat_lon: Array<f64, Ix1>,
    pub cat_lat: Array<f64, Ix1>,
    pub cat_frame: String,
    pub cat_epoch: f64,
    pub cat_dist: Option<Array<f64, Ix1>>,
    pub cat_vrad: Option<Array<f64, Ix1>>,
    pub info_source: Option<String>,
}

impl PartialEq<EphemVal> for EphemVal {
    fn eq(&self, other: &EphemVal) -> bool {
        match self.cat_id == other.cat_id {
            true => {}
            false => return false,
        }
        match self.cat_type == other.cat_type {
            true => {}
            false => return false,
        }

        match self.cat_lon.abs_diff_eq(&other.cat_lon, 1e-6) {
            true => {}
            false => return false,
        }

        match self.cat_lat.abs_diff_eq(&other.cat_lat, 1e-6) {
            true => {}
            false => return false,
        }

        match self.cat_frame == other.cat_frame {
            true => {}
            false => return false,
        }

        match abs_diff_eq!(self.cat_epoch, other.cat_epoch, epsilon = 1e-6) {
            true => {}
            false => return false,
        }
        match &self.cat_dist {
            Some(dist1) => match &other.cat_dist {
                Some(dist2) => match dist1.abs_diff_eq(dist2, 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.cat_dist {
                None => {}
                Some(_) => return false,
            },
        }

        match &self.cat_vrad {
            Some(vrad1) => match &other.cat_vrad {
                Some(vrad2) => match vrad1.abs_diff_eq(vrad2, 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.cat_vrad {
                None => {}
                Some(_) => return false,
            },
        }

        match self.info_source == other.info_source {
            true => {}
            false => return false,
        }
        true
    }
}
impl Eq for EphemVal {}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
#[serde(untagged)]
pub enum CatTypes {
    Unphased(UnphasedVal),
    Sidereal(SiderealVal),
    Ephem(EphemVal),
}

pub type Catalog = BTreeMap<String, CatTypes>;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum VisUnit {
    Uncalib,
    Jansky,
    Kelvinstr,
}
impl FromStr for VisUnit {
    type Err = String;

    fn from_str(input: &str) -> Result<VisUnit, Self::Err> {
        match input
            .trim_matches(char::is_whitespace)
            .to_lowercase()
            .as_str()
        {
            "uncalib" => Ok(VisUnit::Uncalib),
            "jy" => Ok(VisUnit::Jansky),
            "k str" => Ok(VisUnit::Kelvinstr),
            unit => Err(format!("Unknown Visibility Unit: {}.", unit)),
        }
    }
}

impl std::fmt::Display for VisUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PhaseType {
    Drift,
    Phased,
    Multi,
}

impl FromStr for PhaseType {
    type Err = String;

    fn from_str(input: &str) -> Result<PhaseType, Self::Err> {
        match input
            .trim_matches(char::is_whitespace)
            .to_lowercase()
            .as_str()
        {
            "drift" => Ok(PhaseType::Drift),
            "phased" => Ok(PhaseType::Phased),
            "multi" => Ok(PhaseType::Multi),
            other => Err(format!("Unknown phase type: {}.", other)),
        }
    }
}
impl std::fmt::Display for PhaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EqConvention {
    Divide,
    Multiply,
    Unknown,
}

impl FromStr for EqConvention {
    type Err = String;

    fn from_str(input: &str) -> Result<EqConvention, Self::Err> {
        match input
            .trim_matches(char::is_whitespace)
            .to_lowercase()
            .as_str()
        {
            "divide" => Ok(EqConvention::Divide),
            "multiply" => Ok(EqConvention::Multiply),
            "unknown" => Ok(EqConvention::Unknown),
            other => Err(format!("Unknown Equalization Convention: {}.", other)),
        }
    }
}
impl std::fmt::Display for EqConvention {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Orientation {
    East,
    North,
    Unknown,
}

impl FromStr for Orientation {
    type Err = String;

    fn from_str(input: &str) -> Result<Orientation, Self::Err> {
        match input
            .trim_matches(char::is_whitespace)
            .to_lowercase()
            .as_str()
        {
            "east" => Ok(Orientation::East),
            "north" => Ok(Orientation::North),
            "unknown" => Ok(Orientation::Unknown),
            other => Err(format!("Unknown Equalization Convention: {}.", other)),
        }
    }
}
impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BltOrders {
    Ant1,
    Ant2,
    Time,
    Baseline,
    Bda,
    Unknown,
}
impl std::fmt::Display for BltOrders {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BltOrder {
    pub major: BltOrders,
    pub minor: BltOrders,
}
impl std::fmt::Display for BltOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:}, {:}",
            self.major.to_string().to_lowercase(),
            self.minor.to_string().to_lowercase()
        )
    }
}

impl FromStr for BltOrder {
    type Err = String;

    fn from_str(input: &str) -> Result<BltOrder, Self::Err> {
        match input
            .trim_matches(char::is_whitespace)
            .to_lowercase()
            .as_str()
        {
            "bda," => Ok(BltOrder {
                major: BltOrders::Bda,
                minor: BltOrders::Bda,
            }),
            "baseline, time" => Ok(BltOrder {
                major: BltOrders::Baseline,
                minor: BltOrders::Time,
            }),
            "baseline, ant1" => Ok(BltOrder {
                major: BltOrders::Baseline,
                minor: BltOrders::Ant1,
            }),
            "baseline, ant2" => Ok(BltOrder {
                major: BltOrders::Baseline,
                minor: BltOrders::Ant2,
            }),
            "time, baseline" => Ok(BltOrder {
                major: BltOrders::Time,
                minor: BltOrders::Baseline,
            }),
            "time, ant1" => Ok(BltOrder {
                major: BltOrders::Time,
                minor: BltOrders::Ant1,
            }),
            "time, ant2" => Ok(BltOrder {
                major: BltOrders::Time,
                minor: BltOrders::Ant2,
            }),
            "ant1, ant2" => Ok(BltOrder {
                major: BltOrders::Ant1,
                minor: BltOrders::Ant2,
            }),
            "ant1, time" => Ok(BltOrder {
                major: BltOrders::Ant1,
                minor: BltOrders::Time,
            }),
            "ant1, baseline" => Ok(BltOrder {
                major: BltOrders::Ant1,
                minor: BltOrders::Baseline,
            }),
            "ant2, ant1" => Ok(BltOrder {
                major: BltOrders::Ant2,
                minor: BltOrders::Ant1,
            }),
            "ant2, time" => Ok(BltOrder {
                major: BltOrders::Ant2,
                minor: BltOrders::Time,
            }),
            "ant2, baseline" => Ok(BltOrder {
                major: BltOrders::Ant2,
                minor: BltOrders::Baseline,
            }),
            "unknown" => Ok(BltOrder {
                major: BltOrders::Unknown,
                minor: BltOrders::Unknown,
            }),
            other => Err(format!("Unknown Blt Ordering: {}.", other)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UVMeta {
    pub nbls: u32,
    pub nblts: u32,
    pub nspws: u32,
    pub npols: u8,
    pub ntimes: u32,
    pub nfreqs: u32,
    pub nphases: u32,
    pub nants_data: u32,
    pub blt_order: BltOrder,
    pub vis_units: VisUnit,
    pub nants_telescope: u32,
    pub phase_type: PhaseType,
    pub x_orientation: Orientation,
    pub instrument: String,
    pub telescope_name: String,
    pub telescope_location: [f64; 3],
    pub object_name: String,
    pub eq_coeffs_convention: EqConvention,
    pub dut1: Option<f32>,
    pub gst0: Option<f32>,
    pub rdate: Option<String>,
    pub earth_omega: Option<f32>,
    pub timesys: Option<String>,
    pub uvplane_reference_time: Option<i32>,
    pub history: String,
}

impl PartialEq<UVMeta> for UVMeta {
    fn eq(&self, other: &UVMeta) -> bool {
        match self.nbls == other.nbls {
            true => {}
            false => return false,
        }

        match self.nblts == other.nblts {
            true => {}
            false => return false,
        }

        match self.nspws == other.nspws {
            true => {}
            false => return false,
        }

        match self.npols == other.npols {
            true => {}
            false => return false,
        }

        match self.ntimes == other.ntimes {
            true => {}
            false => return false,
        }

        match self.nfreqs == other.nfreqs {
            true => {}
            false => return false,
        }

        match self.nphases == other.nphases {
            true => {}
            false => return false,
        }

        match self.nants_data == other.nants_data {
            true => {}
            false => return false,
        }

        match self.blt_order == other.blt_order {
            true => {}
            false => return false,
        }

        match self.vis_units == other.vis_units {
            true => {}
            false => return false,
        }

        match self.nants_telescope == other.nants_telescope {
            true => {}
            false => return false,
        }

        match self.phase_type == other.phase_type {
            true => {}
            false => return false,
        }

        match self.x_orientation == other.x_orientation {
            true => {}
            false => return false,
        }

        match self.instrument == other.instrument {
            true => {}
            false => return false,
        }

        match self.telescope_name == other.telescope_name {
            true => {}
            false => return false,
        }

        for (x1, x2) in self
            .telescope_location
            .iter()
            .zip(other.telescope_location.iter())
        {
            match abs_diff_eq!(x1, x2, epsilon = 1e-6) {
                true => {}
                false => return false,
            }
        }

        match self.object_name == other.object_name {
            true => {}
            false => return false,
        }

        match self.eq_coeffs_convention == other.eq_coeffs_convention {
            true => {}
            false => return false,
        }

        match self.dut1 {
            Some(dut1) => match other.dut1 {
                Some(dut2) => match abs_diff_eq!(dut1, dut2) {
                    true => {}
                    false => {
                        return false;
                    }
                },
                None => {
                    return false;
                }
            },
            None => match other.dut1 {
                None => {}
                Some(_) => {
                    return false;
                }
            },
        }

        match self.gst0 {
            Some(gst1) => match other.gst0 {
                Some(gst2) => match abs_diff_eq!(gst1, gst2) {
                    true => {}
                    false => {
                        return false;
                    }
                },
                None => {
                    return false;
                }
            },
            None => match other.gst0 {
                None => {}
                Some(_) => {
                    return false;
                }
            },
        }

        match self.rdate == other.rdate {
            true => {}
            false => return false,
        }
        // check handling here
        match self.earth_omega {
            Some(omega1) => match other.earth_omega {
                Some(omega2) => match abs_diff_eq!(omega1, omega2) {
                    true => {}
                    false => {
                        return false;
                    }
                },
                None => {
                    return false;
                }
            },
            None => match other.earth_omega {
                None => {}
                Some(_) => {
                    return false;
                }
            },
        }

        match self.timesys == other.timesys {
            true => {}
            false => return false,
        }

        match self.uvplane_reference_time == other.uvplane_reference_time {
            true => {}
            false => return false,
        }

        match self.history == other.history {
            true => {}
            false => return false,
        }
        true
    }
}
impl Eq for UVMeta {}

impl UVMeta {
    pub fn new() -> UVMeta {
        UVMeta {
            nbls: 0,
            nblts: 0,
            npols: 0,
            nspws: 0,
            ntimes: 0,
            nfreqs: 0,
            nphases: 1,
            nants_data: 0,
            nants_telescope: 0,
            blt_order: BltOrder {
                major: BltOrders::Unknown,
                minor: BltOrders::Unknown,
            },
            x_orientation: Orientation::Unknown,
            phase_type: PhaseType::Drift,
            vis_units: VisUnit::Uncalib,
            instrument: "Unknown".to_string(),
            telescope_name: "Unknown".to_string(),
            telescope_location: [0f64; 3],
            object_name: "Unknown".to_string(),
            eq_coeffs_convention: EqConvention::Unknown,
            dut1: None,
            gst0: None,
            rdate: None,
            earth_omega: None,
            timesys: None,
            uvplane_reference_time: None,
            history: "".to_string(),
        }
    }
}

impl Default for UVMeta {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ArrayMetaData {
    pub spw_array: Array<u32, Ix1>,
    pub uvw_array: Array<f64, Ix2>,
    pub time_array: Array<f64, Ix1>,
    pub lst_array: Array<f64, Ix1>,
    pub ant_1_array: Array<u32, Ix1>,
    pub ant_2_array: Array<u32, Ix1>,
    pub baseline_array: Array<u32, Ix1>,
    pub freq_array: Array<f64, Ix1>,
    pub spw_id_array: Array<u32, Ix1>,
    pub polarization_array: Array<i8, Ix1>,
    pub integration_time: Array<f64, Ix1>,
    pub channel_width: Array<f64, Ix1>,
    pub antenna_numbers: Array<u32, Ix1>,
    pub antenna_names: Array<String, Ix1>,
    pub antenna_positions: Array<f64, Ix2>,
    pub eq_coeffs: Option<Array<f32, Ix2>>,
    pub antenna_diameters: Option<Array<f32, Ix1>>,
    pub phase_center_catalog: Catalog,
    pub phase_center_id_array: Array<u32, Ix1>,
}
impl PartialEq<ArrayMetaData> for ArrayMetaData {
    fn eq(&self, other: &ArrayMetaData) -> bool {
        match self.spw_array == other.spw_array {
            true => {}
            false => return false,
        }

        match self.spw_array == other.spw_array {
            true => {}
            false => return false,
        }
        match self.uvw_array.abs_diff_eq(&other.uvw_array, 1e-6) {
            true => {}
            false => return false,
        }

        match self.time_array.abs_diff_eq(&other.time_array, 1e-6) {
            true => {}
            false => return false,
        }

        match self.lst_array.abs_diff_eq(&other.lst_array, 1e-6) {
            true => {}
            false => return false,
        }

        match self.ant_1_array == other.ant_1_array {
            true => {}
            false => return false,
        }

        match self.ant_2_array == other.ant_2_array {
            true => {}
            false => return false,
        }

        match self.baseline_array == other.baseline_array {
            true => {}
            false => return false,
        }

        match self.freq_array.abs_diff_eq(&other.freq_array, 1e-6) {
            true => {}
            false => return false,
        }

        match self.spw_id_array == other.spw_id_array {
            true => {}
            false => return false,
        }

        match self.polarization_array == other.polarization_array {
            true => {}
            false => return false,
        }

        match self
            .integration_time
            .abs_diff_eq(&other.integration_time, 1e-6)
        {
            true => {}
            false => return false,
        }

        match self.channel_width.abs_diff_eq(&other.channel_width, 1e-6) {
            true => {}
            false => return false,
        }

        match self.antenna_numbers == other.antenna_numbers {
            true => {}
            false => return false,
        }

        match self.antenna_names == other.antenna_names {
            true => {}
            false => return false,
        }

        match self
            .antenna_positions
            .abs_diff_eq(&other.antenna_positions, 1e-6)
        {
            true => {}
            false => return false,
        }

        match &self.eq_coeffs {
            Some(coeffs1) => match &other.eq_coeffs {
                Some(coeffs2) => match coeffs1.abs_diff_eq(coeffs2, 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.eq_coeffs {
                None => {}
                Some(_) => return false,
            },
        }

        match &self.antenna_diameters {
            Some(diams1) => match &other.antenna_diameters {
                Some(diams2) => match diams1.abs_diff_eq(diams2, 1e-6) {
                    true => {}
                    false => return false,
                },
                None => return false,
            },
            None => match &other.antenna_diameters {
                None => {}
                Some(_) => return false,
            },
        }

        match self.phase_center_catalog == other.phase_center_catalog {
            true => {}
            false => return false,
        }

        match self.phase_center_id_array == other.phase_center_id_array {
            true => {}
            false => return false,
        }

        true
    }
}
impl Eq for ArrayMetaData {}

impl ArrayMetaData {
    pub fn new(meta: &UVMeta) -> ArrayMetaData {
        let mut cat = Catalog::new();
        for phase in 0..meta.nphases {
            cat.insert(
                format!("zenith_{}", phase).to_string(),
                CatTypes::Unphased(UnphasedVal {
                    cat_id: phase,
                    cat_type: "unphased".to_string(),
                }),
            );
        }
        ArrayMetaData {
            spw_array: Array::<u32, Ix1>::zeros(meta.nspws as usize),
            uvw_array: Array::<f64, Ix2>::zeros((meta.nblts as usize, 3)),
            time_array: Array::<f64, Ix1>::zeros(meta.nblts as usize),
            lst_array: Array::<f64, Ix1>::zeros(meta.nblts as usize),
            ant_1_array: Array::<u32, Ix1>::zeros(meta.nblts as usize),
            ant_2_array: Array::<u32, Ix1>::zeros(meta.nblts as usize),
            baseline_array: Array::<u32, Ix1>::zeros(meta.nblts as usize),
            freq_array: Array::<f64, Ix1>::zeros(meta.nfreqs as usize),
            spw_id_array: Array::<u32, Ix1>::zeros(meta.nfreqs as usize),
            polarization_array: Array::<i8, Ix1>::zeros(meta.npols as usize),
            integration_time: Array::<f64, Ix1>::zeros(meta.nblts as usize),
            channel_width: Array::<f64, Ix1>::zeros(meta.nfreqs as usize),
            antenna_numbers: Array::<u32, Ix1>::zeros(meta.nants_telescope as usize),
            antenna_names: Array::<f32, Ix1>::range(0.0, meta.nblts as f32, 1.0)
                .mapv(|x| x.to_string()),
            antenna_positions: Array::<f64, Ix2>::zeros((meta.nants_telescope as usize, 3)),
            eq_coeffs: None,
            antenna_diameters: None,
            phase_center_catalog: cat,
            phase_center_id_array: Array::<u32, Ix1>::zeros(meta.nblts as usize),
        }
    }
}
