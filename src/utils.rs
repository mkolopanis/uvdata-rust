use ndarray::{azip, Array, Ix1};
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
            })
        }
        false => {
            let two: T = T::one() + T::one();
            azip!((bl in &mut baselines, &a1 in ant1, &a2 in ant2) {
                *bl = T::from_u32(2048u32).unwrap()
                    * (a1 + one)
                    + (a2 + one)
                    + two.pow(16)
            })
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
    let ant2 = bls.mapv(|x| (x % modulus) - one);
    let ant1 = bls
        .iter()
        .zip(ant2.iter())
        .map(|(&bl, &a2)| (bl - (a2 + one)) / modulus - one)
        .collect();
    (ant1, ant2)
}

#[cfg(test)]
mod test {

    use super::{antnums_to_baseline, baseline_to_antnums, latlonalt_from_xyz, xyz_from_latlonalt};
    use ndarray::array;

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
    fn bls_to_anums() {
        let bls = array![88085u32, 641335u32];
        let ant_1 = array![10u32, 280u32];
        let ant_2 = array![20u32, 310u32];
        let anums = baseline_to_antnums(&bls, false);
        assert_eq!(ant_1, anums.0);
        assert_eq!(ant_2, anums.1)
    }

    #[test]
    fn bls_to_anums256() {
        let bls = array![257u32, 1031u32];
        let ant_1 = array![0u32, 3u32];
        let ant_2 = array![0u32, 6u32];
        let anums = baseline_to_antnums(&bls, true);
        assert_eq!(ant_1, anums.0);
        assert_eq!(ant_2, anums.1)
    }
}
