use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod tokenizer;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_bpe_rs")]
fn bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenizer::get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::encode_train, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::encode, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::decode_string, m)?)?;
    // m.add_function(wrap_pyfunction!(encoder::save_vocab_list, m)?)?;
    Ok(())
}
