use ndarray::{Array, Ix1, Ix2};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum VisUnit {
    Uncalib,
    Jansky,
    Kelvinstr,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PhaseType {
    Drift,
    Phased,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EqConvention {
    Divide,
    Multiply,
    Unknown,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Orientation {
    East,
    North,
    Unknown,
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BltOrder {
    pub major: BltOrders,
    pub minor: BltOrders,
}

#[derive(Debug, PartialEq, Clone)]
pub struct UVMeta {
    pub nbls: u32,
    pub nblts: u32,
    pub nspws: u32,
    pub npols: u8,
    pub ntimes: u32,
    pub nfreqs: u32,
    pub nants_data: u32,
    pub blt_order: BltOrder,
    pub vis_units: VisUnit,
    pub nants_telescope: u32,
    pub phase_type: PhaseType,
    pub x_orientation: Orientation,
    pub instrument: String,
    pub telescope_name: String,
    pub telescope_location: [f64; 3],
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
    pub antenna_positions: Array<f64, Ix2>,
    pub eq_coeffs: Option<Array<f32, Ix2>>,
    pub antenna_diameters: Option<Array<f32, Ix1>>,
}

impl ArrayMetaData {
    pub fn new(meta: &UVMeta) -> ArrayMetaData {
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
            antenna_positions: Array::<f64, Ix2>::zeros((meta.nants_telescope as usize, 3)),
            eq_coeffs: None,
            antenna_diameters: None,
        }
    }
}
