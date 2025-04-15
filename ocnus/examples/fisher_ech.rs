// use chrono::Local;
// use core::f64;
// use directories::UserDirs;
// use dyn_fmt::AsStrFormatExt;
// use env_logger::Builder;
// use log::warn;
// use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};
// use ocnus::{
//     coords::XCState,
//     OcnusFEVM::{
//         OcnusFEVM, FEVMEnsbl, FEVMError, FEVMNoise, FEVMNullState, FisherInformation,
//         filters::{
//             ABCParticleFilter, ABCParticleFilterMode, ParticleFilter, ParticleFilterError,
//             ParticleFilterSettingsBuilder, SIRParticleFilter, root_mean_square_filter,
//         },
//         models::ECHModel,
//     },
//     obser::{ObserVec, ScObs, ScObsConf, ScObsSeries},
//     prodef::{Constant1D, CovMatrix, Reciprocal1D, Uniform1D, UnivariateND},
// };
// use std::{io::prelude::*, path::Path};

// fn main() {
//     Builder::new()
//         .format(|buf, record| {
//             writeln!(
//                 buf,
//                 "{} [{}] - {}",
//                 Local::now().format("%Y-%m-%dT%H:%M:%S.%f"),
//                 record.level(),
//                 record.args()
//             )
//         })
//         .filter(None, log::LevelFilter::Info)
//         .init();

//     let prior = UnivariateND::new([
//         Uniform1D::new_uvpdf((-1.0, 1.0)).unwrap(),
//         Uniform1D::new_uvpdf((1.5, 4.5)).unwrap(),
//         Constant1D::new_uvpdf(0.0),
//         Uniform1D::new_uvpdf((-0.5, 0.5)).unwrap(),
//         Constant1D::new_uvpdf(1.0),
//         Reciprocal1D::new_uvpdf((0.1, 0.3)).unwrap(),
//         Uniform1D::new_uvpdf((0.7, 0.9)).unwrap(),
//         Constant1D::new_uvpdf(1125.0),
//         Uniform1D::new_uvpdf((10.0, 25.0)).unwrap(),
//         Constant1D::new_uvpdf(1.0),
//         Uniform1D::new_uvpdf((3.0, 12.0)).unwrap(),
//         Constant1D::new_uvpdf(0.0),
//     ]);

//     let model = ECHModel::new(prior);

//     let sc: ScObsSeries<f64, ObserVec<f64, 3>> = ScObsSeries::<_, ObserVec<_, 3>>::from_iterator(
//         (0..20).map(|i| ScObs::new(i as f64 * 1.0 * 3600.0, ScObsConf::Distance(1.0), None)),
//     );

//     let mut fevmd_0 = FEVMEnsbl {
//         params: Matrix::<f64, _, Dyn, VecStorage<f64, Const<12>, Dyn>>::from_iterator(
//             1,
//             [
//                 -15_f64.to_radians(),
//                 177.20819_f64.to_radians(),
//                 0.0_f64.to_radians(),
//                 -0.04,
//                 1.0,
//                 0.2,
//                 0.75,
//                 1125.0,
//                 16.0,
//                 1.0,
//                 6.0,
//                 0.0,
//             ],
//         ),
//         fm_states: vec![FEVMNullState::default(); 1],
//         cs_states: vec![XCState::default(); 1],
//         weights: vec![1.0],
//     };

//     const NOISE_VAR: f64 = 1.0;

//     let corrfunc = |d| if d == 0.0 { NOISE_VAR } else { 0.0 };

//     let result = model
//         .fischer_information_matrix(&sc, &fevmd_0, &corrfunc)
//         .unwrap();

//     let fim = result[0];

//     let fim_clean = fim;

//     println!("{:.5}", fim_clean);
//     println!("{:.5}", fim_clean.pseudo_inverse(0.0001).unwrap());

//     // Create synthetic output.
//     let mut output = DMatrix::<ObserVec<_, 3>>::zeros(sc.len(), 1);

//     model
//         .fevm_initialize_states_only(&sc, &mut fevmd_0)
//         .unwrap();
//     model
//         .fevm_simulate(&sc, &mut fevmd_0, &mut output, None)
//         .unwrap();

//     let sc_synth = ScObsSeries::<_, ObserVec<_, 3>>::from_iterator((0..20).map(|i| {
//         ScObs::new(
//             i as f64 * 1.0 * 3600.0,
//             ScObsConf::Distance(1.0),
//             if output.row(i)[0].any_nan() {
//                 None
//             } else {
//                 Some(output.row(i)[0].clone())
//             },
//         )
//     }));

//     const ENSEMBLE_SIZE: usize = 4096;

//     let mut pfsettings = ParticleFilterSettingsBuilder::<_, 3>::default()
//         .exploration_factor(1.5)
//         .noise(FEVMNoise::Gaussian(NOISE_VAR.sqrt(), 1))
//         .quantiles([0.2, 0.5, 1.0])
//         .rseed(70)
//         .simulation_time_limit(5.0)
//         .build()
//         .unwrap();

//     let init_result = model
//         .pf_initialize_ensemble(
//             &sc_synth,
//             ENSEMBLE_SIZE,
//             2_usize.pow(18),
//             Some((&root_mean_square_filter, 9.5)),
//             &mut pfsettings,
//         )
//         .unwrap();

//     let base_dir_opt = if let Some(user_dirs) = UserDirs::new() {
//         let doc_dir = user_dirs.document_dir().unwrap();

//         let path = Path::new(doc_dir).join("Data").join("example_fisher_ech");

//         if path.exists() {
//             Some(path)
//         } else {
//             warn!(
//                 "path {} not found, results will not be saved",
//                 path.into_os_string().into_string().unwrap()
//             );

//             None
//         }
//     } else {
//         warn!("no document fold found, results will not be saved");

//         None
//     };

//     if let Some(base_dir) = &base_dir_opt {
//         let format_str = base_dir
//             .join("{}.particles")
//             .into_os_string()
//             .into_string()
//             .unwrap()
//             .format(&[0]);

//         init_result.write(format_str).unwrap();
//     }

//     let mut total_counter = 1;
//     let mut fevmd = init_result.get_fevmd().clone();
//     let mut epsilon = init_result.get_error_quantiles().unwrap()[0];

//     for _ in 1..10 {
//         let run = model.abcpf_run(
//             &sc_synth,
//             &fevmd,
//             ENSEMBLE_SIZE,
//             2_usize.pow(16),
//             ABCParticleFilterMode::Threshold((&root_mean_square_filter, epsilon)),
//             &mut pfsettings,
//         );

//         let run_abc_result = match run {
//             Ok(result) => result,
//             Err(err) => match err {
//                 FEVMError::ParticleFilter(pferr) => match pferr {
//                     ParticleFilterError::TimeLimitExceeded { .. } => break,
//                     _ => {
//                         pfsettings.rseed += 1;
//                         continue;
//                     }
//                 },
//                 _ => panic!(),
//             },
//         };

//         if let Some(base_dir) = &base_dir_opt {
//             let format_str = base_dir
//                 .join("{}.particles")
//                 .into_os_string()
//                 .into_string()
//                 .unwrap()
//                 .format(&[total_counter]);

//             run_abc_result.write(format_str).unwrap();
//         }

//         fevmd = run_abc_result.get_fevmd().clone();
//         total_counter += 1;
//         epsilon = run_abc_result.get_error_quantiles().unwrap()[0];
//     }

//     for _ in 0..1 {
//         let mut sir_settings = ParticleFilterSettingsBuilder::<_, 3>::default()
//             .exploration_factor(1.0)
//             .noise(FEVMNoise::Multivariate(
//                 CovMatrix::from_matrix(
//                     &DMatrix::from_diagonal_element(sc_synth.len(), sc_synth.len(), NOISE_VAR)
//                         .as_view(),
//                 )
//                 .unwrap(),
//                 0,
//             ))
//             .build()
//             .unwrap();

//         let run = model.sirpf_run(
//             &sc_synth,
//             &fevmd,
//             ENSEMBLE_SIZE,
//             4 * 4096,
//             &mut sir_settings,
//         );

//         let run_sir_result = run.unwrap();

//         if let Some(base_dir) = &base_dir_opt {
//             let format_str = base_dir
//                 .join("{}.particles")
//                 .into_os_string()
//                 .into_string()
//                 .unwrap()
//                 .format(&[total_counter]);

//             run_sir_result.write(format_str).unwrap();
//         }

//         fevmd = run_sir_result.get_fevmd().clone();
//         total_counter += 1;
//     }
// }

fn main() {}
