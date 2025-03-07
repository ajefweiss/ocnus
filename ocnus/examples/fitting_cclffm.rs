use chrono::Local;
use env_logger::Builder;
use nalgebra::DMatrix;
use ocnus::{
    ScObs, ScObsConf, ScObsSeries,
    fevm::{CCLFFModel, FEVM, FEVMData, FEVMDataPairs, FEVMNoiseZero, FEVMNullState},
    geometry::XCState,
    obser::ObserVec,
    stats::{PDFConstant, PDFUniform, PDFUnivariates},
};
use std::io::prelude::*;

fn main() {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Debug)
        .init();

    let prior = PDFUnivariates::new([
        PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
        PDFUniform::new_uvpdf((0.0, 6.00)).unwrap(),
        PDFUniform::new_uvpdf((-0.75, 0.75)).unwrap(),
        PDFUniform::new_uvpdf((0.05, 0.35)).unwrap(),
        PDFConstant::new_uvpdf(1125.0),
        PDFUniform::new_uvpdf((5.0, 25.0)).unwrap(),
        PDFUniform::new_uvpdf((-2.4, 2.4)).unwrap(),
        PDFUniform::new_uvpdf((0.7, 1.0)).unwrap(),
    ]);

    let model = CCLFFModel(prior.clone());

    #[allow(clippy::excessive_precision)]
    let refobs = [
        ObserVec::from([-2.67159853, -13.38264243, -7.20006469]),
        // ObserVec::from([-3.13531505, -13.98423491, -6.64658269]),
        ObserVec::from([-5.34451816, -14.85328723, -1.73480069]),
        // ObserVec::from([-3.38360946, -15.34720671, -1.71139834]),
        ObserVec::from([-4.00239361, -14.94369, 1.2604304]),
        // ObserVec::from([-4.21540068, -13.93568105, 4.73878965]),
        ObserVec::from([-2.7273795, -14.29364075, 4.8267508]),
        // ObserVec::from([-5.41469694, -13.9772912, 4.14112123]),
        ObserVec::from([-4.28371012, -13.89409455, 5.13879915]),
        // ObserVec::from([-4.30711573, -12.61217154, 5.78382821]),
    ];

    let sc = ScObsSeries::<ObserVec<3>>::from_iterator((0..refobs.len()).map(|i| {
        ScObs::new(
            10800.0 + i as f32 * 3600.0 * 2.0,
            ScObsConf::Distance(1.0),
            Some(refobs[i].clone()),
        )
    }));

    let mut fevmdp = model
        .fevm_data(
            &sc,
            4096,
            2_usize.pow(18),
            None::<&PDFUnivariates<8>>,
            None::<&FEVMNoiseZero>,
            42,
        )
        .unwrap();

    // let mut abc_data = CCLFF_Model::pf_builder()
    //     .seed_init(43)
    //     .ensbl_size(8 * 1024)
    //     .sim_ensbl_size(64 * 1024)
    //     .exploration_factor(1.5)
    //     .build()
    //     .unwrap();

    // let mut threshold = 10.0;
    // let mut result_rmse;
    // let mut last_i = 0;

    // for i in 0..10 {
    //     result_rmse = model
    //         .abc_filter(&sc, &mut abc_data, Some(&covm), threshold)
    //         .unwrap();

    //     result_rmse.sort_by(|a, b| a.partial_cmp(b).unwrap());

    //     // New threshold is lower 25.0%.
    //     threshold = *result_rmse
    //         .iter()
    //         .try_fold(0, |acc, next| {
    //             if acc < (abc_data.ensbl_size as f32 * 0.33) as usize {
    //                 Ok(acc + 1)
    //             } else {
    //                 Err(next)
    //             }
    //         })
    //         .unwrap_err();

    //     let list_as_json = serde_json::to_string(&abc_data).unwrap();

    //     // let mut file =
    //     //     std::fs::File::create(format!("C:\\Users\\Andreas\\Data\\{:03}.particles", i)).expect("could not create file!");

    //     let mut file = std::fs::File::create(format!(
    //         "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
    //         i
    //     ))
    //     .expect("could not create file!");

    //     file.write_all(list_as_json.as_bytes())
    //         .expect("Cannot write to the file!");

    //     last_i = i;
    // }

    // for j in 0..10 {
    //     model.mvll_filter(&sc, &mut abc_data, &covm).unwrap();

    //     let list_as_json = serde_json::to_string(&abc_data).unwrap();

    //     // let mut file = std::fs::File::create(format!("C:\\Users\\Andreas\\Data\\{:03}.particles", last_i + j + 1))
    //     //     .expect("could not create file!");

    //     let mut file = std::fs::File::create(format!(
    //         "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
    //         1 + last_i + j
    //     ))
    //     .expect("could not create file!");

    //     file.write_all(list_as_json.as_bytes())
    //         .expect("Cannot write to the file!");
    // }
}
