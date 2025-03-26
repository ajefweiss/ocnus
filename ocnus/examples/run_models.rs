use nalgebra::{Const, DMatrix, Dyn, Matrix, SVector, VecStorage};
use ocnus::{
    fevm::{CCLFFModel, CCUTModel, ECHModel, FEVM, FEVMData, FEVMNullState},
    geom::XCState,
    obser::{ObserVec, ScObs, ScObsConf, ScObsSeries},
    stats::{PDFConstant, PDFUniform, PDFUnivariates},
};

fn main() {
    {
        let prior = PDFUnivariates::new([
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.5, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.05, 0.1)).unwrap(),
            PDFUniform::new_uvpdf((0.1, 0.5)).unwrap(),
            PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
            PDFConstant::new_uvpdf(1125.0),
            PDFUniform::new_uvpdf((5.0, 100.0)).unwrap(),
            PDFUniform::new_uvpdf((-2.4, 2.4)).unwrap(),
        ]);

        let model = CCLFFModel::new(prior);

        let sc = ScObsSeries::<f32, ObserVec<f32, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f32 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut data = FEVMData {
            params: Matrix::<f32, Const<8>, Dyn, VecStorage<f32, Const<8>, Dyn>>::zeros(1),
            fevm_states: vec![FEVMNullState::default(); 1],
            geom_states: vec![XCState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);
        let mut output_ics = DMatrix::<ObserVec<f32, 12>>::zeros(sc.len(), 1);

        data.params.set_column(
            0,
            &SVector::<f32, 8>::from([
                5.0_f32.to_radians(),
                -3.0_f32.to_radians(),
                0.1,
                0.25,
                0.0,
                600.0,
                20.0,
                1.0 / 0.25,
            ]),
        );

        model
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");

        model
            .fevm_simulate(&sc, &mut data, &mut output, None)
            .expect("simulation failed");

        model
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");

        model
            .fevm_simulate_diagnostics(&sc, &mut data, &mut output_ics)
            .expect("simulation failed");

        println!("{:?}", &data);
        println!("{:?}", &output_ics[(0, 0)]);
        println!("{:?}", &output_ics[(2, 0)]);
        println!("{:?}", &output_ics[(4, 0)]);
        println!("{:?}", &output_ics[(7, 0)]);
    }

    {
        let prior = PDFUnivariates::new([
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.05, 0.1)).unwrap(),
            PDFUniform::new_uvpdf((0.1, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.1, 0.5)).unwrap(),
            PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
            PDFConstant::new_uvpdf(1125.0),
            PDFUniform::new_uvpdf((5.0, 100.0)).unwrap(),
            PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((-10.0, 10.0)).unwrap(),
            PDFUniform::new_uvpdf((-10.0, 10.0)).unwrap(),
        ]);

        let model = ECHModel::new(prior);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut data = FEVMData {
            params: Matrix::<f64, Const<12>, Dyn, VecStorage<f64, Const<12>, Dyn>>::zeros(1),
            fevm_states: vec![FEVMNullState::default(); 1],
            geom_states: vec![XCState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);
        let mut output_ics = DMatrix::<ObserVec<f64, 12>>::zeros(sc.len(), 1);

        data.params.set_column(
            0,
            &SVector::<f64, 12>::from([
                5.0_f64.to_radians(),
                -3.0_f64.to_radians(),
                0.0_f64.to_radians(),
                0.1,
                1.0,
                0.25,
                0.0,
                600.0,
                20.0,
                1.0,
                1.0 / 0.25,
                1.0 / 0.25,
            ]),
        );

        model
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");

        model
            .fevm_simulate(&sc, &mut data, &mut output, None)
            .expect("simulation failed");

        model
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");

        model
            .fevm_simulate_diagnostics(&sc, &mut data, &mut output_ics)
            .expect("simulation failed");

        println!("{:?}", &data);
        println!("{:?}", &output_ics[(0, 0)]);
        println!("{:?}", &output_ics[(2, 0)]);
        println!("{:?}", &output_ics[(4, 0)]);
        println!("{:?}", &output_ics[(7, 0)]);
    }
}
