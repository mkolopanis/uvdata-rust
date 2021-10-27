use ndarray::{Array, Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, str::FromStr};

// TODO: make and enum of the different catalog types
// and catalog structs themselves probably too
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct UnphasedVal {
    pub cat_id: u32,
    pub cat_type: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
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

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
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

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
pub enum CatTypes {
    Unphased(UnphasedVal),
    Sidereal(SiderealVal),
    Ephem(EphemVal),
}

pub type Catalog = HashMap<String, CatTypes>;

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
        write!(f, "{:}, {:}", self.major, self.minor)
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

#[derive(Debug, PartialEq, Clone)]
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

#[derive(Debug, PartialEq, Clone)]
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
