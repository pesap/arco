//! Python bindings for the Arco optimization library.

use pyo3::prelude::*;

/// Arco Python module.
#[pymodule]
fn arco(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO: Add Python class/function bindings here
    // Example:
    // m.add_class::<PyModel>()?;
    let _ = m;
    Ok(())
}
