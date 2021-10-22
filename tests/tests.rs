use ndarray::Array3;
use num::Complex;
use std::path::Path;
use uvdata::*;

#[test]
fn init_metadata_false() {
    let meta = UVMeta {
        nbls: 3,
        nblts: 15,
        ntimes: 5,
        nfreqs: 12,
        npols: 4,
        nspws: 1,
        nphases: 1,
        nants_data: 5,
        nants_telescope: 12,
        blt_order: BltOrder {
            major: BltOrders::Unknown,
            minor: BltOrders::Unknown,
        },
        phase_type: PhaseType::Drift,
        vis_units: VisUnit::Jansky,
        x_orientation: Orientation::Unknown,
        instrument: "Test".to_owned(),
        telescope_name: "Test".to_owned(),
        telescope_location: [0.0, 0.0, 0.0],
        eq_coeffs_convention: EqConvention::Unknown,
        dut1: None,
        gst0: None,
        rdate: None,
        earth_omega: None,
        timesys: None,
        uvplane_reference_time: None,
        history: "".to_string(),
    };
    let test_data = Array3::<Complex<f64>>::zeros((
        meta.nblts as usize,
        meta.ntimes as usize,
        meta.npols as usize,
    ));
    let test_nsample = Array3::<f32>::zeros((
        meta.nblts as usize,
        meta.ntimes as usize,
        meta.npols as usize,
    ));
    let test_flag = Array3::<bool>::from_elem(
        (
            meta.nblts as usize,
            meta.ntimes as usize,
            meta.npols as usize,
        ),
        false,
    );
    let uvd = UVData::<f64, f32>::new(meta, false);
    assert_eq!(uvd.data_array.unwrap(), test_data);
    assert_eq!(uvd.nsample_array.unwrap(), test_nsample);
    assert_eq!(uvd.flag_array.unwrap(), test_flag);
}

#[test]
fn from_uvmeta_bool() {
    let meta = UVMeta {
        nbls: 3,
        nblts: 15,
        ntimes: 5,
        nfreqs: 12,
        npols: 4,
        nspws: 1,
        nphases: 1,
        nants_data: 5,
        nants_telescope: 12,
        blt_order: BltOrder {
            major: BltOrders::Unknown,
            minor: BltOrders::Unknown,
        },
        phase_type: PhaseType::Drift,
        vis_units: VisUnit::Jansky,
        x_orientation: Orientation::Unknown,
        instrument: "Test".to_owned(),
        telescope_name: "Test".to_owned(),
        telescope_location: [0.0, 0.0, 0.0],
        eq_coeffs_convention: EqConvention::Unknown,
        dut1: None,
        gst0: None,
        rdate: None,
        earth_omega: None,
        timesys: None,
        uvplane_reference_time: None,
        history: "".to_string(),
    };
    let test_data = Array3::<Complex<f64>>::zeros((
        meta.nblts as usize,
        meta.ntimes as usize,
        meta.npols as usize,
    ));
    let test_nsample = Array3::<f32>::zeros((
        meta.nblts as usize,
        meta.ntimes as usize,
        meta.npols as usize,
    ));
    let test_flag = Array3::<bool>::from_elem(
        (
            meta.nblts as usize,
            meta.ntimes as usize,
            meta.npols as usize,
        ),
        false,
    );
    let uvd = UVData::<f64, f32>::from((meta, false));
    assert_eq!(uvd.data_array.unwrap(), test_data);
    assert_eq!(uvd.nsample_array.unwrap(), test_nsample);
    assert_eq!(uvd.flag_array.unwrap(), test_flag);
}

#[test]
fn from_uvmeta() {
    let meta = UVMeta {
        nbls: 3,
        nblts: 15,
        ntimes: 5,
        nfreqs: 12,
        npols: 4,
        nspws: 1,
        nphases: 1,
        nants_data: 5,
        nants_telescope: 12,
        blt_order: BltOrder {
            major: BltOrders::Unknown,
            minor: BltOrders::Unknown,
        },
        phase_type: PhaseType::Drift,
        vis_units: VisUnit::Jansky,
        x_orientation: Orientation::Unknown,
        instrument: "Test".to_owned(),
        telescope_name: "Test".to_owned(),
        telescope_location: [0.0, 0.0, 0.0],
        eq_coeffs_convention: EqConvention::Unknown,
        dut1: None,
        gst0: None,
        rdate: None,
        earth_omega: None,
        timesys: None,
        uvplane_reference_time: None,
        history: "".to_string(),
    };
    let uvd = UVData::<f64, f32>::from(meta);
    assert!(uvd.data_array.is_none());
    assert!(uvd.nsample_array.is_none());
    assert!(uvd.flag_array.is_none());
}

#[test]
fn init_metadata_false_f32() {
    let meta = UVMeta {
        nbls: 3,
        nblts: 15,
        ntimes: 5,
        nfreqs: 12,
        npols: 4,
        nspws: 1,
        nphases: 1,
        nants_data: 5,
        nants_telescope: 12,
        blt_order: BltOrder {
            major: BltOrders::Unknown,
            minor: BltOrders::Unknown,
        },
        phase_type: PhaseType::Drift,
        vis_units: VisUnit::Jansky,
        x_orientation: Orientation::Unknown,
        instrument: "Foo".to_string(),
        telescope_name: "Test".to_owned(),
        telescope_location: [0.0, 0.0, 0.0],
        eq_coeffs_convention: EqConvention::Unknown,
        dut1: None,
        gst0: None,
        rdate: None,
        earth_omega: None,
        timesys: None,
        uvplane_reference_time: None,
        history: "".to_string(),
    };
    let test_data = Array3::<Complex<f32>>::zeros((
        meta.nblts as usize,
        meta.ntimes as usize,
        meta.npols as usize,
    ));
    let test_nsample = Array3::<f32>::zeros((
        meta.nblts as usize,
        meta.ntimes as usize,
        meta.npols as usize,
    ));
    let test_flag = Array3::<bool>::from_elem(
        (
            meta.nblts as usize,
            meta.ntimes as usize,
            meta.npols as usize,
        ),
        false,
    );
    let uvd = UVData::<f32, f32>::new(meta, false);

    assert_eq!(uvd.data_array.unwrap(), test_data);
    assert_eq!(uvd.nsample_array.unwrap(), test_nsample);
    assert_eq!(uvd.flag_array.unwrap(), test_flag);
}
#[test]
fn init_metadata_true() {
    let meta = UVMeta {
        nbls: 3,
        nblts: 15,
        ntimes: 5,
        nfreqs: 12,
        npols: 4,
        nspws: 1,
        nphases: 1,
        nants_data: 5,
        nants_telescope: 12,
        blt_order: BltOrder {
            major: BltOrders::Unknown,
            minor: BltOrders::Unknown,
        },
        phase_type: PhaseType::Drift,
        vis_units: VisUnit::Jansky,
        x_orientation: Orientation::Unknown,
        instrument: "Test".to_owned(),
        telescope_name: "Test".to_owned(),
        telescope_location: [0.0, 0.0, 0.0],
        eq_coeffs_convention: EqConvention::Unknown,
        dut1: None,
        gst0: None,
        rdate: None,
        earth_omega: None,
        timesys: None,
        uvplane_reference_time: None,
        history: "".to_string(),
    };
    let uvd = UVData::<f64, f32>::new(meta, true);
    assert!(uvd.data_array.is_none());
    assert!(uvd.nsample_array.is_none());
    assert!(uvd.flag_array.is_none());
}

#[test]
fn test_read_files() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    for fname in data_dir.read_dir().expect("No data found") {
        if let Ok(fname) = fname {
            match UVData::<f64, f32>::read_uvh5(fname.path(), true) {
                Ok(_) => assert!(true),
                Err(_) => assert!(false),
            };
        }
    }
}
