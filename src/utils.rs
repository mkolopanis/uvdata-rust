use ndarray::{azip, Array, Ix1, Ix2};
use num_traits::{cast::FromPrimitive, Float, PrimInt};

const GPS_A: f64 = 6378137f64;
const GPS_B: f64 = 6356752.31424518;
const E2: f64 = 6.69437999014e-3;
const EP2: f64 = 6.73949674228e-3;

pub fn xyz_from_latlonalt<T: Float + FromPrimitive>(lat: T, lon: T, alt: T) -> [T; 3] {
    let gps_a: T = T::from_f64(GPS_A).unwrap();
    let gps_b: T = T::from_f64(GPS_B).unwrap();
    let e2: T = T::from_f64(E2).unwrap();
    let one: T = T::from_f64(1.0f64).unwrap();

    let sin_lat: T = lat.to_radians().sin();
    let cos_lat: T = lat.to_radians().cos();

    let sin_lon: T = lon.to_radians().sin();
    let cos_lon: T = lon.to_radians().cos();

    let b_div_a2: T = (gps_b / gps_a).powi(2);
    let gps_n: T = gps_a / (one - e2 * sin_lat.powi(2)).sqrt();
    let mut xyz: [T; 3] = [T::zero(); 3];

    xyz[0] = (gps_n + alt) * cos_lat * cos_lon;
    xyz[1] = (gps_n + alt) * cos_lat * sin_lon;

    xyz[2] = (b_div_a2 * gps_n + alt) * sin_lat;
    xyz
}

pub fn latlonalt_from_xyz<T: Float + FromPrimitive>(xyz: [T; 3]) -> (T, T, T) {
    // see wikipedia geodetic_datum and Datum transformations of
    // GPS positions PDF in docs/references folder
    let gps_a: T = T::from_f64(GPS_A).unwrap();
    let gps_b: T = T::from_f64(GPS_B).unwrap();
    let e2: T = T::from_f64(E2).unwrap();
    let ep2: T = T::from_f64(EP2).unwrap();
    let one: T = T::from_f64(1.0f64).unwrap();

    let gps_p = (xyz[0].powi(2) + xyz[1].powi(2)).sqrt();
    let gps_theta = (xyz[2] * gps_a).atan2(gps_p * gps_b);

    let lat = (xyz[2] + ep2 * gps_b * gps_theta.sin().powi(3))
        .atan2(gps_p - e2 * gps_a * gps_theta.cos().powi(3));

    let lon = xyz[1].atan2(xyz[0]);

    let alt = (gps_p / lat.cos()) - gps_a / (one - e2 * lat.sin().powi(2)).sqrt();

    (lat, lon, alt)
}

pub fn antnums_to_baseline<T: PrimInt + FromPrimitive>(
    ant1: &Array<T, Ix1>,
    ant2: &Array<T, Ix1>,
    attempt256: bool,
) -> Array<T, Ix1> {
    let mut baselines: Array<T, Ix1> = Array::<T, Ix1>::zeros(ant1.len());
    let one: T = T::one();
    match attempt256 {
        true => {
            azip!((bl in &mut baselines, &a1 in ant1, &a2 in ant2) {
                *bl = T::from_u32(256u32).unwrap()
                    * (a1 + one)
                    + (a2 + one)
            });
        }
        false => {
            let two: T = T::one() + T::one();
            azip!((bl in &mut baselines, &a1 in ant1, &a2 in ant2) {
                *bl = T::from_u32(2048u32).unwrap()
                    * (a1 + one)
                    + (a2 + one)
                    + two.pow(16)
            });
        }
    };
    baselines
}

pub fn baseline_to_antnums<T: PrimInt + FromPrimitive>(
    baselines: &Array<T, Ix1>,
    use256: bool,
) -> (Array<T, Ix1>, Array<T, Ix1>) {
    let one = T::one();
    let modulus = match use256 {
        false => T::from_u32(2048u32).unwrap(),
        true => T::from_u32(256u32).unwrap(),
    };
    let bls = match use256 {
        true => baselines.clone(),
        false => {
            let two_16: T = (one + one).pow(16);
            baselines.mapv(|x| x - two_16)
        }
    };
    let ant2: Array<T, Ix1> = bls.mapv(|x| (x % modulus) - one);
    let ant1: Array<T, Ix1> = bls
        .iter()
        .zip(ant2.iter())
        .map(|(&bl, &a2)| (bl - (a2 + one)) / modulus - one)
        .collect();
    (ant1, ant2)
}

pub fn enu_from_ecef<T>(xyz: &Array<T, Ix2>, lat: T, lon: T, alt: T) -> Array<T, Ix2>
where
    T: Float + FromPrimitive,
{
    let xyz_center: [T; 3] = xyz_from_latlonalt(lat, lon, alt);

    let sin_lat = lat.to_radians().sin();
    let cos_lat = lat.to_radians().cos();

    let sin_lon = lon.to_radians().sin();
    let cos_lon = lon.to_radians().cos();

    let mut enu = xyz.clone();
    for mut _enu in enu.outer_iter_mut() {
        let x_use = _enu[0] - xyz_center[0];
        let y_use = _enu[1] - xyz_center[1];
        let z_use = _enu[2] - xyz_center[2];

        _enu[0] = -sin_lon * x_use + cos_lon * y_use;
        _enu[1] = -sin_lat * cos_lon * x_use - sin_lat * sin_lon * y_use + cos_lat * z_use;
        _enu[2] = cos_lat * cos_lon * x_use + cos_lat * sin_lon * y_use + sin_lat * z_use;
    }
    enu
}

pub fn ecef_from_enu<T>(enu: &Array<T, Ix2>, lat: T, lon: T, alt: T) -> Array<T, Ix2>
where
    T: Float + FromPrimitive,
{
    let xyz_center: [T; 3] = xyz_from_latlonalt(lat, lon, alt);
    let sin_lat = lat.to_radians().sin();
    let cos_lat = lat.to_radians().cos();

    let sin_lon = lon.to_radians().sin();
    let cos_lon = lon.to_radians().cos();

    let mut ecef = enu.clone();
    for (mut _xyz, _enu) in ecef.outer_iter_mut().zip(enu.outer_iter()) {
        _xyz[0] = -sin_lat * cos_lon * _enu[1] - sin_lon * _enu[0]
            + cos_lat * cos_lon * _enu[2]
            + xyz_center[0];
        _xyz[1] = -sin_lat * sin_lon * _enu[1]
            + cos_lon * _enu[0]
            + cos_lat * sin_lon * _enu[2]
            + xyz_center[1];
        _xyz[2] = cos_lat * _enu[1] + sin_lat * _enu[2] + xyz_center[2];
    }
    ecef
}

#[cfg(test)]
mod test {

    use super::{
        antnums_to_baseline, baseline_to_antnums, ecef_from_enu, enu_from_ecef, latlonalt_from_xyz,
        xyz_from_latlonalt,
    };
    use ndarray::{array, stack, Array, Axis};

    #[test]
    fn xyz_from_lla() {
        let ref_latlonalt = [-26.7f64, 116.7f64, 377.8f64];
        let ref_xyz = [-2562123.42683, 5094215.40141, -2848728.58869];

        let out_xyz = xyz_from_latlonalt(ref_latlonalt[0], ref_latlonalt[1], ref_latlonalt[2]);
        for (x1, x2) in ref_xyz.iter().zip(out_xyz.iter()) {
            assert_abs_diff_eq!(x1, x2, epsilon = 1e-3);
        }
    }

    #[test]
    fn lla_from_xyz() {
        let ref_latlonalt = [-26.7f64, 116.7f64, 377.8f64];
        let ref_xyz = [-2562123.42683, 5094215.40141, -2848728.58869];

        let lla_out: (f64, f64, f64) = latlonalt_from_xyz(ref_xyz);
        let lla_out: [f64; 3] = [lla_out.0.to_degrees(), lla_out.1.to_degrees(), lla_out.2];
        for (x1, x2) in lla_out.iter().zip(ref_latlonalt.iter()) {
            assert_abs_diff_eq!(x1, x2, epsilon = 1e-3);
        }
    }
    #[test]
    fn anums_to_bls() {
        let ant_1 = array![10u32, 280u32];
        let ant_2 = array![20u32, 310u32];
        assert_eq!(
            array![88085u32, 641335u32],
            antnums_to_baseline(&ant_1, &ant_2, false)
        );
    }
    #[test]
    fn anums_to_bls_256() {
        let ant_1 = array![0u32, 3u32];
        let ant_2 = array![0u32, 6u32];
        assert_eq!(
            array![257u32, 1031u32],
            antnums_to_baseline(&ant_1, &ant_2, true)
        );
    }

    #[test]
    fn bls_to_antnums() {
        let bls = array![88085u32, 641335u32];
        let ant_1 = array![10u32, 280u32];
        let ant_2 = array![20u32, 310u32];
        let anums = baseline_to_antnums(&bls, false);
        assert_eq!(ant_1, anums.0);
        assert_eq!(ant_2, anums.1)
    }

    #[test]
    fn bls_to_antnums256() {
        let bls = array![257u32, 1031u32];
        let ant_1 = array![0u32, 3u32];
        let ant_2 = array![0u32, 6u32];
        let anums = baseline_to_antnums(&bls, true);
        assert_eq!(ant_1, anums.0);
        assert_eq!(ant_2, anums.1)
    }

    #[test]
    fn ecef_to_enu() {
        let center_lat = -30.7215261207;
        let center_lon = 21.4283038269;
        let center_alt = 1051.7;
        let lats = [
            -30.72218216,
            -30.72138101,
            -30.7212785,
            -30.7210011,
            -30.72159853,
            -30.72206199,
            -30.72174614,
            -30.72188775,
            -30.72183915,
            -30.72100138,
        ];
        let lons = [
            21.42728211,
            21.42811727,
            21.42814544,
            21.42795736,
            21.42686739,
            21.42918772,
            21.42785662,
            21.4286408,
            21.42750933,
            21.42896567,
        ];
        let alts = [
            1052.25, 1051.35, 1051.2, 1051., 1051.45, 1052.04, 1051.68, 1051.87, 1051.77, 1051.06,
        ];
        let east = [
            -97.87631659,
            -17.87126443,
            -15.17316938,
            -33.19049252,
            -137.60520964,
            84.67346748,
            -42.84049408,
            32.28083937,
            -76.1094745,
            63.40285935,
        ];
        let north = [
            -72.7437482,
            16.09066646,
            27.45724573,
            58.21544651,
            -8.02964511,
            -59.41961437,
            -24.39698388,
            -40.09891961,
            -34.70965816,
            58.18410876,
        ];
        let up = [
            0.54883333,
            -0.35004539,
            -0.50007736,
            -0.70035299,
            -0.25148791,
            0.33916067,
            -0.02019057,
            0.16979185,
            0.06945155,
            -0.64058124,
        ];

        let mut xyz = stack![Axis(0), lats, lons, alts].reversed_axes();
        for mut _xyz in xyz.outer_iter_mut() {
            _xyz.assign(&Array::from_vec(
                xyz_from_latlonalt(_xyz[0], _xyz[1], _xyz[2]).to_vec(),
            ));
        }
        let enu = enu_from_ecef(&xyz, center_lat, center_lon, center_alt);
        let enu_ref = stack![Axis(0), east, north, up].reversed_axes();
        assert!(enu.abs_diff_eq(&enu_ref, 1e-3))
    }

    #[test]
    fn enu_to_ecef() {
        let center_lat = -30.7215261207;
        let center_lon = 21.4283038269;
        let center_alt = 1051.7;
        let x = [
            5109327.46674067,
            5109339.76407785,
            5109344.06370947,
            5109365.11297147,
            5109372.115673,
            5109266.94314734,
            5109329.89620962,
            5109295.13656657,
            5109337.21810468,
            5109329.85680612,
        ];

        let y = [
            2005130.57953031,
            2005221.35184577,
            2005225.93775268,
            2005214.8436201,
            2005105.42364036,
            2005302.93158317,
            2005190.65566222,
            2005257.71335575,
            2005157.78980089,
            2005304.7729239,
        ];

        let z = [
            -3239991.24516348,
            -3239914.4185286,
            -3239904.57048431,
            -3239878.02656316,
            -3239935.20415493,
            -3239979.68381865,
            -3239949.39266985,
            -3239962.98805772,
            -3239958.30386264,
            -3239878.08403833,
        ];

        let east = [
            -97.87631659,
            -17.87126443,
            -15.17316938,
            -33.19049252,
            -137.60520964,
            84.67346748,
            -42.84049408,
            32.28083937,
            -76.1094745,
            63.40285935,
        ];
        let north = [
            -72.7437482,
            16.09066646,
            27.45724573,
            58.21544651,
            -8.02964511,
            -59.41961437,
            -24.39698388,
            -40.09891961,
            -34.70965816,
            58.18410876,
        ];
        let up = [
            0.54883333,
            -0.35004539,
            -0.50007736,
            -0.70035299,
            -0.25148791,
            0.33916067,
            -0.02019057,
            0.16979185,
            0.06945155,
            -0.64058124,
        ];
        let enu = stack![Axis(0), east, north, up].reversed_axes();
        let xyz = ecef_from_enu(&enu, center_lat, center_lon, center_alt);
        let ref_xyz = stack![Axis(0), x, y, z].reversed_axes();

        assert!(xyz.abs_diff_eq(&ref_xyz, 1e-6))
    }
}
