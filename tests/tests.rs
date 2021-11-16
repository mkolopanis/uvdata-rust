#[cfg(test)]
#[macro_use]
extern crate approx;

use ndarray::Array3;
use num_complex::Complex;
use std::{fs, path::Path};
use tempdir::TempDir;
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
        object_name: "Unknown".to_string(),
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
        object_name: "Unknown".to_string(),
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
        object_name: "Unknown".to_string(),
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
        object_name: "Unknown".to_string(),
        eq_coeffs_convention: EqConvention::Unknown,
        dut1: None,
        gst0: None,
        rdate: None,
        earth_omega: None,
        timesys: None,
        uvplane_reference_time: None,
        history: "".to_string(),
    };
    let test_data = Array3::<Complex<f32>>::from_elem(
        (
            meta.nblts as usize,
            meta.ntimes as usize,
            meta.npols as usize,
        ),
        Complex { re: 2.0, im: -3.2 },
    );
    let test_nsample = Array3::<f32>::from_elem(
        (
            meta.nblts as usize,
            meta.ntimes as usize,
            meta.npols as usize,
        ),
        3.1415,
    );
    let test_flag = Array3::<bool>::from_elem(
        (
            meta.nblts as usize,
            meta.ntimes as usize,
            meta.npols as usize,
        ),
        false,
    );
    let mut uvd = UVData::<f32, f32>::new(meta, false);
    uvd.data_array = Some(test_data.clone());
    uvd.nsample_array = Some(test_nsample.clone());
    uvd.flag_array = Some(test_flag.clone());

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
        object_name: "Unknown".to_string(),
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
    data_dir
        .read_dir()
        .expect("No data found")
        .filter_map(Result::ok)
        .filter(|fname| fname.path().extension().unwrap() == "uvh5")
        .for_each(|fname| {
            match UVData::<f64, f32>::read_uvh5(fname.path(), true) {
                Ok(_) => assert!(true),
                Err(_) => assert!(false),
            };
        })
}

#[test]
fn test_roundtrip_files() {
    let outdir = TempDir::new("roundtrip_test").expect("Unable to create temporary test directory");
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let files = fs::read_dir(data_dir).expect("No data.");
    files
        .filter_map(Result::ok)
        // .take_while(|f| f.is_ok())
        // .filter_map(|f| f.ok())
        .filter(|fname| fname.path().extension().unwrap() == "uvh5")
        .for_each(|fname| {
            println!("filename {:?}", fname);
            let uvd = UVData::<f64, f32>::read_uvh5(fname.path(), true)
                .expect(format!("Unable to read file {:?}", fname).as_str());
            let uvd1 = uvd.clone();
            let outpath = outdir.path().clone().join(format!(
                "out_{}.uvh5",
                fname.path().file_stem().and_then(|x| x.to_str()).unwrap()
            ));
            uvd.write_uvh5(&outpath, true)
                .expect(format!("Unable to write {:?}", outpath).as_str());
            let mut uvd2 = UVData::<f64, f32>::read_uvh5(&outpath, true)
                .expect(format!("Unable to read file {:?}", outpath).as_str());

            // histories are probably the same but let's just make sure.
            uvd2.meta.history = uvd1.meta.history.clone();

            assert_eq!(uvd1.meta, uvd2.meta);

            assert_eq!(uvd1.meta_arrays, uvd2.meta_arrays);

            assert_eq!(uvd1, uvd2);
        })
}

#[test]
fn test_latlonalt_fn() {
    let mut meta = UVMeta::new();
    let ref_latlonalt = [-26.7f64, 116.7f64, 377.8f64];
    let ref_xyz = [-2562123.42683, 5094215.40141, -2848728.58869];
    meta.telescope_location = ref_xyz;

    let uvd: UVData<f64, f32> = UVData::<f64, f32>::from(meta);
    let telescope_lla = uvd.telescope_location_latlonalt();
    for (x1, x2) in [telescope_lla.0, telescope_lla.1, telescope_lla.2]
        .iter()
        .zip(
            [
                ref_latlonalt[0].to_radians(),
                ref_latlonalt[1].to_radians(),
                ref_latlonalt[2],
            ]
            .iter(),
        )
    {
        assert_abs_diff_eq!(x1, x2, epsilon = 1e-3)
    }
    let telescope_lla_degrees = uvd.telescope_location_latlonalt_degrees();
    for (x1, x2) in [
        telescope_lla_degrees.0,
        telescope_lla_degrees.1,
        telescope_lla_degrees.2,
    ]
    .iter()
    .zip(ref_latlonalt.iter())
    {
        assert_abs_diff_eq!(x1, x2, epsilon = 1e-3)
    }
}
