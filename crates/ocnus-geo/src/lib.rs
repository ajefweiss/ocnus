//! Ocnus - Atmospheric and Geospace Models

#[derive(Debug, Default)]
#[repr(C)]
/// NRLMSIS Input
pub struct NRLMSISINPUT {
    /// year, currently ignored
    pub year: i32,
    /// day of year
    pub doy: i32,
    /// seconds in day (UT)
    pub sec: f64,
    /// altitude in kilometers
    pub alt: f64,
    /// geodetic latitude
    pub g_lat: f64,
    /// geodetic longitude
    pub g_long: f64,
    /// local apparent solar time (hours), see note below
    pub lst: f64,
    /// 81 day average of F10.7 flux (centered on doy)
    pub f107a: f64,
    /// daily F10.7 flux for previous day
    pub f107: f64,
    /// magnetic index(daily)
    pub ap: f64,
    /// see above
    pub ap_a: [f64; 7],
}

#[derive(Debug, Default)]
#[repr(C)]
/// NRLMSIS Flags
pub struct NRLMSISFLAGS {
    /// switches
    pub switches: [i32; 24],
    /// sw
    pub sw: [f64; 24],
    /// swc
    pub swc: [f64; 24],
}

#[derive(Debug, Default)]
#[repr(C)]
/// NRLMSIS Output
pub struct NRLMSISOUTPUT {
    /// densities
    pub d: [f64; 9],
    /// temperatures
    pub t: [f64; 2],
}

unsafe extern "C" {
    /// A
    pub fn gtd7(input: &NRLMSISINPUT, flags: &NRLMSISFLAGS, output: &mut NRLMSISOUTPUT);

    /// A
    pub fn gts7(input: &NRLMSISINPUT, flags: &NRLMSISFLAGS, output: &mut NRLMSISOUTPUT);
}

pub mod models;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msis() {
        let input = NRLMSISINPUT {
            year: 2026,
            doy: 5,
            sec: 0.0,
            alt: 1000.0,
            g_lat: 0.0,
            g_long: 0.0,
            lst: 0.0 / 3600.0 + 0.0 / 15.0,
            f107a: 157.4,
            f107: 154.58,
            ap: 15.0,
            ap_a: [15.0, 15.0, 15.0, 7.0, 3.0, 3.38, 19.5],
        };

        let flags = NRLMSISFLAGS {
            switches: [
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
            sw: [
                0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            swc: [
                0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        };

        let mut output = NRLMSISOUTPUT {
            d: [0.0; 9],
            t: [0.0; 2],
        };

        unsafe {
            gts7(&input, &flags, &mut output);

            dbg!(&output);
        }
    }
}
