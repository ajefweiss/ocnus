use chrono::Local;
use dyn_fmt::AsStrFormatExt;
use env_logger::Builder;
use log::warn;
use nalgebra::{DVector, DVectorView};
use ocnus::{
    base::{ScObs, ScObsConf, ScObsSeries},
    methods::filters::{ParticleFilterBuilder, ParticleFilterSettingsBuilder},
    models::WSAHUXModel,
    obser::{MeasureInSituPlasmaBulkVelocity, ObserVec, ObserVecNoise, observec_rmsep},
    stats::{MultivariateDensity, UniformDensity},
};
use std::{fs::create_dir_all, io::prelude::*, path::Path};

fn main() {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S.%f"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Info)
        .init();

    #[allow(clippy::excessive_precision)]
    let refobs = [
        368.0, 366.0, 376.0, 373.0, 375.0, 364.0, 369.0, 365.0, 364.0, 361.0, 362.0, 361.0, 357.0,
        351.0, 353.0, 351.0, 351.0, 346.0, 347.0, 348.0, 344.0, 339.0, 340.0, 341.0, 342.0, 340.0,
        336.0, 334.0, 330.0, 331.0, 331.0, 330.0, 333.0, 332.0, 335.0, 335.0, 331.0, 330.0, 330.0,
        330.0, 331.0, 329.0, 323.0, 329.0, 329.0, 332.0, 328.0, 331.0, 331.0, 336.0, 342.0, 339.0,
        338.0, 335.0, 342.0, 340.0, 335.0, 336.0, 344.0, 374.0, 428.0, 444.0, 453.0, 465.0, 488.0,
        489.0, 516.0, 520.0, 523.0, 557.0, 611.0, 593.0, 602.0, 623.0, 640.0, 664.0, 652.0, 623.0,
        600.0, 599.0, 620.0, 624.0, 597.0, 606.0, 630.0, 600.0, 588.0, 578.0, 582.0, 576.0, 558.0,
        544.0, 556.0, 547.0, 552.0, 542.0, 529.0, 531.0, 563.0, 515.0, 506.0, 509.0, 524.0, 546.0,
        553.0, 549.0, 538.0, 536.0, 548.0, 550.0, 547.0, 538.0, 546.0, 558.0, 548.0, 556.0, 544.0,
        549.0, 548.0, 554.0, 542.0, 539.0, 541.0, 551.0, 542.0, 541.0, 532.0, 518.0, 529.0, 521.0,
        513.0, 506.0, 491.0, 486.0, 470.0, 463.0, 461.0, 455.0, 453.0, 453.0, 452.0, 442.0, 438.0,
        435.0, 432.0, 430.0, 427.0, 424.0, 420.0, 416.0, 410.0, 404.0, 397.0, 392.0, 388.0, 387.0,
        379.0, 378.0, 379.0, 372.0, 373.0, 380.0, 388.0, 391.0, 384.0, 376.0, 386.0, 387.0, 387.0,
        384.0, 376.0, 378.0, 381.0, 373.0, 378.0, 384.0, 398.0, 400.0, 400.0, 395.0, 388.0, 381.0,
        392.0, 416.0, 414.0, 388.0, 403.0, 386.0, 373.0, 358.0, 374.0, 383.0, 366.0, 393.0, 389.0,
        370.0, 373.0, 377.0, 374.0, 388.0, 391.0, 391.0, 392.0, 385.0, 387.0, 384.0, 389.0, 392.0,
        407.0, 420.0, 452.0, 456.0, 454.0, 479.0, 493.0, 497.0, 496.0, 462.0, 457.0, 447.0, 437.0,
        428.0, 431.0, 427.0, 414.0, 408.0, 403.0, 387.0, 393.0, 396.0, 400.0, 408.0, 419.0, 413.0,
        408.0, 400.0, 401.0, 399.0, 403.0, 436.0, 466.0, 459.0, 451.0, 443.0, 501.0, 496.0, 485.0,
        508.0, 524.0, 537.0, 523.0, 503.0, 499.0, 506.0, 544.0, 604.0, 595.0, 599.0, 617.0, 616.0,
        612.0, 621.0, 610.0, 604.0, 590.0, 608.0, 598.0, 610.0, 589.0, 567.0, 553.0, 544.0, 525.0,
        539.0, 521.0, 534.0, 527.0, 529.0, 526.0, 525.0, 529.0, 550.0, 540.0, 528.0, 528.0, 485.0,
        480.0, 477.0, 472.0, 486.0, 485.0, 482.0, 489.0, 485.0, 470.0, 492.0, 490.0, 510.0, 502.0,
        483.0, 474.0, 481.0, 458.0, 441.0, 445.0, 453.0, 460.0, 455.0, 457.0, 459.0, 457.0, 453.0,
        443.0, 440.0, 437.0, 434.0, 435.0, 443.0, 446.0, 431.0, 433.0, 423.0, 427.0, 414.0, 411.0,
        425.0, 419.0, 420.0, 413.0, 413.0, 407.0, 415.0, 429.0, 415.0, 417.0, 423.0, 431.0, 425.0,
        415.0, 412.0, 414.0, 398.0, 397.0, 393.0, 390.0, 383.0, 382.0, 377.0, 371.0, 367.0, 366.0,
        368.0, 362.0, 360.0, 354.0, 354.0, 353.0, 361.0, 364.0, 364.0, 360.0, 363.0, 365.0, 363.0,
        362.0, 363.0, 361.0, 355.0, 355.0, 355.0, 354.0, 349.0, 346.0, 344.0, 341.0, 342.0, 340.0,
        340.0, 344.0, 347.0, 346.0, 342.0, 338.0, 339.0, 344.0, 345.0, 341.0, 340.0, 337.0, 335.0,
        336.0, 337.0, 340.0, 339.0, 344.0, 343.0, 340.0, 341.0, 343.0, 351.0, 354.0, 352.0, 345.0,
        350.0, 351.0, 353.0, 353.0, 351.0, 358.0, 366.0, 379.0, 391.0, 403.0, 400.0, 405.0, 405.0,
        403.0, 397.0, 402.0, 418.0, 411.0, 395.0, 373.0, 369.0, 361.0, 352.0, 349.0, 346.0, 361.0,
        354.0, 347.0, 345.0, 337.0, 358.0, 357.0, 352.0, 347.0, 348.0, 356.0, 382.0, 392.0, 397.0,
        406.0, 417.0, 409.0, 403.0, 396.0, 392.0, 389.0, 382.0, 392.0, 388.0, 389.0, 379.0, 374.0,
        367.0, 363.0, 365.0, 377.0, 378.0, 381.0, 414.0, 416.0, 430.0, 463.0, 471.0, 483.0, 495.0,
        476.0, 479.0, 457.0, 426.0, 428.0, 423.0, 420.0, 420.0, 411.0, 395.0, 392.0, 388.0, 387.0,
        388.0, 395.0, 387.0, 383.0, 394.0, 392.0, 390.0, 393.0, 399.0, 400.0, 401.0, 394.0, 389.0,
        389.0, 385.0, 382.0, 395.0, 388.0, 368.0, 368.0, 365.0, 367.0, 364.0, 369.0, 377.0, 376.0,
        362.0, 355.0, 350.0, 351.0, 354.0, 359.0, 365.0, 358.0, 343.0, 349.0, 346.0, 345.0, 343.0,
        344.0, 339.0, 334.0, 337.0, 339.0, 335.0, 330.0, 326.0, 319.0, 317.0, 322.0, 318.0, 317.0,
        316.0, 322.0, 326.0, 329.0, 336.0, 345.0, 344.0, 345.0, 330.0, 330.0, 336.0, 322.0, 322.0,
        323.0, 330.0, 328.0, 339.0, 358.0, 357.0, 356.0, 353.0, 349.0, 348.0, 366.0, 369.0, 378.0,
        382.0, 401.0, 412.0, 411.0, 404.0, 401.0, 402.0, 399.0, 399.0, 410.0, 411.0, 408.0, 437.0,
        477.0, 485.0, 481.0, 497.0, 499.0, 498.0, 490.0, 501.0, 511.0, 511.0, 519.0, 536.0, 549.0,
        567.0, 565.0, 599.0, 580.0, 593.0, 603.0, 626.0, 607.0, 626.0, 639.0, 630.0, 626.0, 623.0,
        627.0, 638.0, 627.0, 631.0, 614.0, 605.0, 613.0, 598.0, 583.0, 586.0, 577.0, 570.0, 560.0,
        548.0, 534.0, 522.0, 510.0, 502.0, 497.0, 488.0, 487.0, 486.0, 485.0, 477.0, 466.0, 488.0,
        486.0, 477.0, 477.0, 472.0, 461.0, 456.0, 454.0, 445.0, 446.0, 439.0, 432.0, 426.0, 406.0,
        401.0, 397.0, 395.0, 379.0, 374.0, 374.0, 372.0, 368.0, 365.0, 364.0, 351.0, 350.0, 355.0,
        352.0, 353.0, 354.0, 354.0,
    ];

    const ENSEMBLE_SIZE: usize = 2_usize.pow(10);
    const STEP: usize = 4;

    let sc = ScObsSeries::<f32>::from_iterator((0..(refobs.len() / STEP)).map(|idx| {
        ScObs::new(
            (STEP * idx) as f32 * 3600.0,
            ScObsConf::Position([1.0, 0.0, 0.0]),
        )
    }));

    let sc_obser = DVector::from_iterator(
        sc.len(),
        (0..sc.len()).map(|idx| ObserVec::<_, 1>::from([refobs[STEP * idx]])),
    );

    let prior = MultivariateDensity::new(&[
        UniformDensity::new((250.0, 350.0)).unwrap(), // a1 = 285 # speeed
        UniformDensity::new((500.0, 620.0)).unwrap(), // a2 = 625 # speed
        UniformDensity::new((2.1 / 9.0, 2.5 / 9.0)).unwrap(), // a3 = 2/9 # expansion factor coefficient
        UniformDensity::new((0.75, 0.95)).unwrap(),           // a4 = 1 # exp offset
        UniformDensity::new((0.55, 0.7)).unwrap(),            // a5 = 0.8 # exp factor
        UniformDensity::new((1.8, 2.2)).unwrap(), // a6 = 2 # distance fraction (DO NOT CHANGE)
        UniformDensity::new((1.8, 2.2)).unwrap(), // a7 = 2 # distance factor coefficient
        UniformDensity::new((2.4, 3.0)).unwrap(), // a8 = 3 # coefficient everything
    ]);

    let path = Path::new("examples")
        .join("data")
        .join("wsapy_NSO-GONG_CR2047_0_NSteps90.json");

    // let range = (&prior).get_range();
    let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior, path, 2.0).unwrap();

    model.limit_latitude(1.0);

    let err_func = |o: &DVectorView<ObserVec<f32, 1>>| observec_rmsep(o, &sc_obser.as_view());

    let mut pf = ParticleFilterBuilder::default()
        .model(model)
        .rseed(42)
        .build()
        .unwrap();

    let mut pfs = ParticleFilterSettingsBuilder::default()
        .series(sc)
        .ensemble_size(ENSEMBLE_SIZE)
        .simulation_ensemble_size(8 * ENSEMBLE_SIZE)
        .expl_factor(2.0)
        .simulation_time_limit(10.0)
        .build()
        .unwrap();

    pf.pf_initialize_ensbl(&pfs, &WSAHUXModel::observe_pbv, (&err_func, 1.0))
        .unwrap();

    let path = Path::new("examples").join("output").join("fit_wsahux");

    let base_dir_opt = if path.exists() {
        Some(path)
    } else if path.parent().unwrap().parent().unwrap().exists() {
        create_dir_all(&path).expect("failed to create output directory");

        Some(path)
    } else {
        warn!(
            "path {} not found, results will not be saved",
            path.into_os_string().into_string().unwrap()
        );

        None
    };

    if let Some(base_dir) = &base_dir_opt {
        let format_str = base_dir
            .join("initial.particles")
            .into_os_string()
            .into_string()
            .unwrap()
            .format(&[0]);

        pf.save(format_str).unwrap();
    }

    let mut noise = ObserVecNoise::Gaussian(25.0, 0);

    pfs.max_iterations = 2;

    let _abc_loop_result = pf.pf_abc_loop(&pfs, &mut noise, &WSAHUXModel::observe_pbv, &err_func);

    if let Some(base_dir) = &base_dir_opt {
        let format_str = base_dir
            .join("1.particles")
            .into_os_string()
            .into_string()
            .unwrap()
            .format(&[0]);

        pf.save(format_str).unwrap();
    }

    let _abc_loop_result = pf.pf_abc_loop(&pfs, &mut noise, &WSAHUXModel::observe_pbv, &err_func);

    if let Some(base_dir) = &base_dir_opt {
        let format_str = base_dir
            .join("2.particles")
            .into_os_string()
            .into_string()
            .unwrap()
            .format(&[0]);

        pf.save(format_str).unwrap();
    }
}
