//! Builder functions for constructing linear expressions.

use crate::expr::core::Expr;

pub fn linear_sum(exprs: Vec<Expr>) -> Expr {
    let mut terms = Vec::new();
    for expr in exprs {
        terms.extend(expr.into_linear_terms());
    }
    Expr::from_linear(terms)
}
