//! Model builder methods for adding variables, constraints, and objectives.

use crate::types::{Bounds, Constraint, Objective, Sense, Variable};
use arco_expr::expr::{ComparisonSense, ConstraintExpr, Expr};
use arco_expr::ids::{ConstraintId, VariableId};

use crate::model::Model;
use crate::model::error::ModelError;

impl Model {
    /// Add a variable to the model.
    pub fn add_variable(&mut self, variable: Variable) -> Result<VariableId, ModelError> {
        if variable.bounds.lower > variable.bounds.upper {
            return Err(ModelError::InvalidVariableBounds {
                lower: variable.bounds.lower,
                upper: variable.bounds.upper,
            });
        }

        let id = VariableId::new(self.next_variable_id);
        self.next_variable_id += 1;

        self.variables.insert(id, variable);
        self.columns.insert(id, Vec::new());

        tracing::debug!(
            component = "model",
            operation = "add_variable",
            status = "success",
            var_id = id.inner(),
            lower = variable.bounds.lower,
            upper = variable.bounds.upper,
            is_integer = variable.is_integer,
            "Added variable to model"
        );
        Ok(id)
    }

    /// Add a constraint to the model.
    pub fn add_constraint(&mut self, constraint: Constraint) -> Result<ConstraintId, ModelError> {
        if constraint.bounds.lower > constraint.bounds.upper {
            return Err(ModelError::InvalidConstraintBounds {
                lower: constraint.bounds.lower,
                upper: constraint.bounds.upper,
            });
        }

        let id = ConstraintId::new(self.next_constraint_id);
        self.next_constraint_id += 1;

        self.constraints.insert(id, constraint);

        tracing::debug!(
            component = "model",
            operation = "add_constraint",
            status = "success",
            constraint_id = id.inner(),
            lower = constraint.bounds.lower,
            upper = constraint.bounds.upper,
            "Added constraint to model"
        );
        Ok(id)
    }

    /// Set the objective function.
    pub fn set_objective(&mut self, objective: Objective) -> Result<(), ModelError> {
        let sense = objective.sense.ok_or(ModelError::NoObjective)?;
        for (var_id, _) in &objective.terms {
            self.ensure_variable_exists(*var_id)?;
        }

        let normalized = self.normalize_terms(objective.terms);
        self.objective = Objective {
            sense: Some(sense),
            terms: normalized,
        };
        self.objective_name = None;
        tracing::debug!(
            component = "model",
            operation = "set_objective",
            status = "success",
            sense = ?sense,
            terms = self.objective.terms.len(),
            "Set objective function"
        );
        Ok(())
    }

    /// Minimize a linear expression.
    ///
    /// Returns an error if the model already has an objective.
    pub fn minimize(&mut self, expr: Expr) -> Result<(), ModelError> {
        if self.objective.sense.is_some() {
            return Err(ModelError::MultipleObjectives);
        }
        self.set_objective(Objective {
            sense: Some(Sense::Minimize),
            terms: expr.into_linear_terms(),
        })
    }

    /// Maximize a linear expression.
    ///
    /// Returns an error if the model already has an objective.
    pub fn maximize(&mut self, expr: Expr) -> Result<(), ModelError> {
        if self.objective.sense.is_some() {
            return Err(ModelError::MultipleObjectives);
        }
        self.set_objective(Objective {
            sense: Some(Sense::Maximize),
            terms: expr.into_linear_terms(),
        })
    }

    /// Add a constraint from an expression and explicit bounds.
    pub fn add_expr_constraint(
        &mut self,
        expr: Expr,
        bounds: Bounds,
    ) -> Result<ConstraintId, ModelError> {
        let constraint_id = self.add_constraint(Constraint { bounds })?;
        for (var_id, coeff) in self.normalize_terms(expr.into_linear_terms()) {
            self.set_coefficient(var_id, constraint_id, coeff)?;
        }
        Ok(constraint_id)
    }

    /// Add a constraint from a comparison expression (e.g., `x + y <= 10`).
    pub fn add_constraint_expr(
        &mut self,
        constraint: ConstraintExpr,
    ) -> Result<ConstraintId, ModelError> {
        let (expr, sense, rhs) = constraint.into_parts();
        let bounds = match sense {
            ComparisonSense::LessEqual => Bounds::new(f64::NEG_INFINITY, rhs),
            ComparisonSense::GreaterEqual => Bounds::new(rhs, f64::INFINITY),
            ComparisonSense::Equal => Bounds::new(rhs, rhs),
        };
        self.add_expr_constraint(expr, bounds)
    }

    /// Add a coefficient to the constraint matrix.
    ///
    /// This adds a coefficient at the intersection of a variable column and constraint row.
    /// Returns an error if the variable or constraint IDs are invalid.
    pub fn set_coefficient(
        &mut self,
        var_id: VariableId,
        constraint_id: ConstraintId,
        coefficient: f64,
    ) -> Result<(), ModelError> {
        self.ensure_variable_exists(var_id)?;
        self.ensure_constraint_exists(constraint_id)?;

        // Update or insert in column-first storage
        if let Some(column) = self.columns.get_mut(&var_id) {
            if let Some(entry) = column.iter_mut().find(|(cid, _)| *cid == constraint_id) {
                entry.1 = coefficient;
            } else {
                column.push((constraint_id, coefficient));
            }
        }

        tracing::trace!(
            component = "model",
            operation = "set_coefficient",
            status = "success",
            var_id = var_id.inner(),
            constraint_id = constraint_id.inner(),
            coefficient,
            "Set coefficient"
        );
        Ok(())
    }

    /// Check if a variable is active.
    pub fn is_variable_active(&self, id: VariableId) -> Result<bool, ModelError> {
        Ok(self.get_variable(id)?.is_active)
    }

    /// Deactivate a variable without removing its column.
    pub fn deactivate_variable(&mut self, id: VariableId) -> Result<(), ModelError> {
        let variable = self
            .variables
            .get_mut(&id)
            .ok_or(ModelError::InvalidVariableId(id))?;
        variable.is_active = false;
        Ok(())
    }

    /// Activate a previously deactivated variable.
    pub fn activate_variable(&mut self, id: VariableId) -> Result<(), ModelError> {
        let variable = self
            .variables
            .get_mut(&id)
            .ok_or(ModelError::InvalidVariableId(id))?;
        variable.is_active = true;
        Ok(())
    }
}
