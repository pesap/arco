//! Storage access methods for the model.

use crate::types::{Constraint, Variable};
use arco_expr::ids::{ConstraintId, VariableId};

use super::Model;
use super::error::ModelError;

impl Model {
    /// Get the number of variables
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Get the number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Get the number of coefficients in the model.
    pub fn num_coefficients(&self) -> usize {
        self.columns.values().map(|coeffs| coeffs.len()).sum()
    }

    /// Get a variable by ID.
    pub fn get_variable(&self, id: VariableId) -> Result<&Variable, ModelError> {
        self.variables
            .get(&id)
            .ok_or(ModelError::InvalidVariableId(id))
    }

    /// Get a constraint by ID.
    pub fn get_constraint(&self, id: ConstraintId) -> Result<&Constraint, ModelError> {
        self.constraints
            .get(&id)
            .ok_or(ModelError::InvalidConstraintId(id))
    }

    /// Get the coefficient matrix in CSC (column-sparse-compressed) format
    ///
    /// Returns an iterator over columns, where each column contains (constraint_id, coefficient) pairs.
    /// This enables zero-copy access to the sparse matrix structure.
    pub fn columns(&self) -> impl Iterator<Item = (VariableId, &Vec<(ConstraintId, f64)>)> {
        self.columns.iter().map(|(&vid, coeffs)| (vid, coeffs))
    }

    /// Get the coefficient matrix in CRS (row-sparse-compressed) form.
    ///
    /// Returns a vector of rows, each containing (variable_id, coefficient) pairs.
    pub fn rows(&self) -> Vec<Vec<(VariableId, f64)>> {
        let mut rows = vec![Vec::new(); self.num_constraints()];
        for (var_id, coeffs) in &self.columns {
            for (constraint_id, coeff) in coeffs {
                let idx = constraint_id.inner() as usize;
                if let Some(row) = rows.get_mut(idx) {
                    row.push((*var_id, *coeff));
                }
            }
        }
        rows
    }

    /// Get the coefficients for a specific variable (column)
    pub fn get_column(&self, var_id: VariableId) -> Option<&Vec<(ConstraintId, f64)>> {
        self.columns.get(&var_id)
    }
}
