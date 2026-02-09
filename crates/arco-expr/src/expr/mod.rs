//! Expression types for optimization modeling.
//!
//! - `expr`       — Expr: terms by degree + constant
//! - `constraint` — ConstraintExpr: expression with comparison sense and RHS
//! - `error`      — Expression construction errors

pub mod constraint;
pub mod core;
pub mod error;

pub use constraint::{ComparisonSense, ConstraintExpr};
pub use core::Expr;
pub use error::LinearExprError;
