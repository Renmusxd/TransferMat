mod full_tmat;
mod make_submat;

use full_tmat::*;
use make_submat::*;
use pyo3::prelude::*;

#[pymodule]
fn py_tiamat(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CircuitSamples>()?;
    m.add_function(wrap_pyfunction!(make_uniform_perm_overlap_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(make_overlap_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(make_ortho_overlap_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(generate_d2s, m)?)?;
    m.add_function(wrap_pyfunction!(get_n_states, m)?)?;
    m.add_function(wrap_pyfunction!(get_n_uniform_states, m)?)?;
    m.add_function(wrap_pyfunction!(generate_uniform_states, m)?)?;
    m.add_function(wrap_pyfunction!(generate_states, m)?)?;
    m.add_function(wrap_pyfunction!(gen_cycles_for_perm, m)?)?;
    m.add_function(wrap_pyfunction!(make_cycles_mat, m)?)?;
    m.add_function(wrap_pyfunction!(gen_self_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(gen_self_overlap_matrix, m)?)?;
    Ok(())
}
