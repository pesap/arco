//! Arco core model builder with lazy lifecycle.

pub mod model;
pub mod slack;
pub mod solver;
pub mod types;

pub use model::{
    CoefficientView, ConstraintView, CscInput, InspectOptions, Model, ModelError, ModelSnapshot,
    ObjectiveView, SlackView, SnapshotMetadata, VariableView,
};

pub use slack::{ElasticHandle, SlackBound, SlackHandle, SlackVariables};
pub use solver::{Solution, Solver, SolverError, SolverStatus};
pub use types::{Bounds, Constraint, Objective, Sense, SimplifyLevel, Variable};
