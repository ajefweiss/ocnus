mod vector;
mod wsahux;

pub use vector::*;
pub use wsahux::*;

macro_rules! unroll_model_errors {
    ($match: expr) => {
        match $match {
            Ok(result) => Ok(result),
            Err(err) => match err {
                ModelError::Sampling => Err(PyRuntimeError::new_err(
                    ("failed to sample the parameter space"),
                )),
                _ => Err(PyRuntimeError::new_err(("unhandled model error"))),
            },
        }
    };
}

macro_rules! unroll_pf_errors {
    ($match: expr) => {
        match $match {
            Ok(result) => Ok(result),
            Err(err) => match err {
                ParticleFilterError::TimeLimitExceeded { limit, .. } => Err(
                    PyRuntimeError::new_err(("the time limit of", limit, "seconds was exceeded")),
                ),
                ParticleFilterError::Model(model_err) => unroll_model_errors!(Err(model_err)),
                ParticleFilterError::InsufficientParticles(count) => Err(PyRuntimeError::new_err(
                    ("insufficient effective particles (", count, ")"),
                )),
                _ => Err(PyRuntimeError::new_err(("unhandled particle filter error"))),
            },
        }
    };
}

pub(crate) use unroll_model_errors;
pub(crate) use unroll_pf_errors;
