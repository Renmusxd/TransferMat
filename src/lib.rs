use itertools::Itertools;
use numpy::ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn make_d2s(d1s: &[usize]) -> Vec<usize> {
    let d1 = |n: usize| {
        if n >= d1s.len() {
            0
        } else {
            d1s[n]
        }
    };
    let max_d2 = 2 * (d1s.len() - 1);
    (0..=max_d2)
        .map(|d2| {
            (0..=d2)
                .map(|na| {
                    let nb = d2 - na;
                    d1(na) * d1(nb)
                })
                .sum::<usize>()
        })
        .collect()
}

#[pyfunction]
fn generate_d2s(d1s: Vec<usize>) -> Vec<usize> {
    make_d2s(&d1s)
}

#[pyfunction]
fn get_n_states(l: usize, k: u8, d1s: Vec<usize>) -> usize {
    let d2s = make_d2s(&d1s);
    let max_gate_n = d2s.len() - 1;
    let n_product_states = l / 2;
    let n_perms = fact(k);
    let n_ng_vecs = max_gate_n.pow(k as u32);
    (n_ng_vecs * n_perms).pow(n_product_states as u32)
}

#[pyfunction]
fn get_n_uniform_states(l: usize, k: u8, d1s: Vec<usize>) -> usize {
    let d2s = make_d2s(&d1s);
    let num_gate_ns = d2s.len();
    let n_product_states = l / 2;
    let n_perms = fact(k);
    let n_ng_vecs = num_gate_ns.pow(k as u32);
    n_ng_vecs.pow(n_product_states as u32) * n_perms
}

fn make_states(
    l: usize,
    k: u8,
    num_gate_ns: usize,
    n_sector: Option<Vec<usize>>,
) -> Vec<Vec<(Vec<usize>, usize)>> {
    let n_product_states = l / 2;
    let n_perms = fact(k);
    let n_ng_vecs = num_gate_ns.pow(k as u32);
    let n_states = (n_perms * n_ng_vecs).pow(n_product_states as u32);

    (0..n_states)
        .filter_map(|i| {
            // Each pair of sites gets ng vec and permutation.
            let (_, ngs) = (0..n_product_states).fold((i, vec![]), |(i, mut acc), _| {
                let perm = i % n_perms;
                let i = i / n_perms;
                let ngindex = i % n_ng_vecs;
                let i = i / n_ng_vecs;
                // Now turn ngindex into [ng1, ng2, ng3..., ngk]
                let (_, ngvec) = (0..k).fold((ngindex, vec![]), |(ngindex, mut acc), _| {
                    let ng = ngindex % num_gate_ns;
                    let ngindex = ngindex / num_gate_ns;
                    acc.push(ng);
                    (ngindex, acc)
                });
                acc.push((ngvec, perm));
                (i, acc)
            });
            if let Some(n_sector) = n_sector.as_ref() {
                let own_n_sector = get_n_sector(ngs.iter().map(|(v, _)| v), k);
                if own_n_sector.eq(n_sector) {
                    Some(ngs)
                } else {
                    None
                }
            } else {
                Some(ngs)
            }
        })
        .collect()
}

fn make_uniform_states(
    l: usize,
    k: u8,
    num_gate_ns: usize,
    n_sector: Option<Vec<usize>>,
) -> Vec<(Vec<Vec<usize>>, usize)> {
    let n_product_states = l / 2;
    let n_perms = fact(k);
    let n_ng_vecs = num_gate_ns.pow(k as u32);
    let n_states = n_ng_vecs.pow(n_product_states as u32) * n_perms;

    (0..n_states)
        .filter_map(|i| {
            // get permutation
            let perm = i % n_perms;
            let i = i / n_perms;
            // get each `n_product_states` gate ng vector across `k` replicas.
            let (_, ngs) = (0..n_product_states).fold((i, vec![]), |(i, mut acc), _| {
                let ngindex = i % n_ng_vecs;
                let i = i / n_ng_vecs;
                // Now turn ngindex into [ng1, ng2, ng3..., ngk]
                let (_, ngvec) = (0..k).fold((ngindex, vec![]), |(ngindex, mut acc), _| {
                    let ng = ngindex % num_gate_ns;
                    let ngindex = ngindex / num_gate_ns;
                    acc.push(ng);
                    (ngindex, acc)
                });
                acc.push(ngvec);
                (i, acc)
            });
            if let Some(n_sector) = n_sector.as_ref() {
                let own_n_sector = get_n_sector(&ngs, k);
                if own_n_sector.eq(n_sector) {
                    Some((ngs, perm))
                } else {
                    None
                }
            } else {
                Some((ngs, perm))
            }
        })
        .collect()
}

fn get_norm_squared<It, Itt>(ngs: It, d2s: &[usize]) -> usize
where
    It: IntoIterator<Item = Itt>,
    Itt: IntoIterator<Item = usize>,
{
    ngs.into_iter()
        .flat_map(|v| v.into_iter())
        .map(|ng| d2s[ng])
        .product()
}

#[pyfunction]
fn generate_states(
    py: Python,
    l: usize,
    k: u8,
    d1s: Vec<usize>,
    n_sector: Option<Vec<usize>>,
) -> PyResult<(Py<PyArray3<usize>>, Py<PyArray2<usize>>, Py<PyArray1<f64>>)> {
    let d2s = make_d2s(&d1s);
    let num_gate_ns = d2s.len();
    let states = make_states(l, k, num_gate_ns, n_sector);
    let (ngs, perms): (Vec<_>, Vec<_>) = states
        .into_iter()
        .map(|v| -> (Vec<_>, Vec<_>) { v.into_iter().unzip() })
        .unzip();
    let n_states = ngs.len();

    // Normalizations
    let norms = ngs
        .iter()
        .map(|ngs| get_norm_squared(ngs.iter().map(|v| v.iter().copied()), &d2s))
        .map(|norm2| (norm2 as f64).sqrt())
        .collect::<Vec<_>>();

    let ngs = ngs
        .into_iter()
        .flat_map(|v| v.into_iter().flat_map(|v| v.into_iter()))
        .collect::<Vec<_>>();
    let perms = perms
        .into_iter()
        .flat_map(|v| v.into_iter())
        .collect::<Vec<_>>();
    let ngs = Array3::from_shape_vec((n_states, l / 2, k as usize), ngs)
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
    let perms = Array2::from_shape_vec((n_states, l / 2), perms)
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
    let norms = Array1::from_vec(norms);

    let ngs = ngs.to_pyarray(py).to_owned();
    let perms = perms.to_pyarray(py).to_owned();
    let norms = norms.to_pyarray(py).to_owned();
    Ok((ngs, perms, norms))
}

#[pyfunction]
fn generate_uniform_states(
    py: Python,
    l: usize,
    k: u8,
    d1s: Vec<usize>,
    n_sector: Option<Vec<usize>>,
) -> PyResult<(Py<PyArray3<usize>>, Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
    let d2s = make_d2s(&d1s);
    let num_gate_ns = d2s.len();
    let states = make_uniform_states(l, k, num_gate_ns, n_sector);
    let (ngs, perms): (Vec<_>, Vec<_>) = states.into_iter().unzip();
    let n_states = ngs.len();

    // Normalizations
    let norms = ngs
        .iter()
        .map(|ngs| get_norm_squared(ngs.iter().map(|v| v.iter().copied()), &d2s))
        .map(|norm2| (norm2 as f64).sqrt())
        .collect::<Vec<_>>();

    let ngs = ngs
        .into_iter()
        .flat_map(|v| v.into_iter().flat_map(|v| v.into_iter()))
        .collect::<Vec<_>>();
    let ngs = Array3::from_shape_vec((n_states, l / 2, k as usize), ngs)
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
    let perms = Array1::from_vec(perms);
    let norms = Array1::from_vec(norms);

    let ngs = ngs.to_pyarray(py).to_owned();
    let perms = perms.to_pyarray(py).to_owned();
    let norms = norms.to_pyarray(py).to_owned();
    Ok((ngs, perms, norms))
}

fn get_n_sector<It, Arr>(ngs: It, k: u8) -> Vec<usize>
where
    It: IntoIterator<Item = Arr>,
    Arr: AsRef<[usize]>,
{
    ngs.into_iter().fold(vec![0; k as usize], |mut acc, a| {
        acc.iter_mut()
            .zip(a.as_ref().iter().copied())
            .for_each(|(x, a)| {
                *x += a;
            });
        acc
    })
}

fn overlap<Arra, Arrb>(
    even_ngs: &[Arra],
    odd_ngs: &[Arrb],
    even_perms: &[usize],
    odd_perms: &[usize],
    d1s: &[usize],
    k: u8,
    cycle_mat: &[Vec<Vec<Vec<usize>>>],
) -> usize
where
    Arra: AsRef<[usize]>,
    Arrb: AsRef<[usize]>,
{
    let d1 = |n: usize| -> usize {
        if n >= d1s.len() {
            0
        } else {
            d1s[n]
        }
    };
    let even_perm = |x: usize| -> usize { even_perms[x / 2] };
    let odd_perm = |x: usize| -> usize {
        let x = (x + 2*odd_perms.len() - 1) % (2*odd_perms.len());
        odd_perms[x / 2]
    };

    if get_n_sector(even_ngs, k) == get_n_sector(odd_ngs, k) {
        // Make a set of vectors which satisfy both even_ngs and odd_ngs.
        let mut base_vecs = (even_ngs.iter().zip(odd_ngs.iter())).fold(
            vec![vec![0; k as usize]],
            |mut acc, (even_ng, odd_ng)| {
                let last = acc.last().unwrap();
                let next = last
                    .iter()
                    .zip(even_ng.as_ref().iter())
                    .map(|(last, gate)| gate - last)
                    .collect::<Vec<_>>();
                acc.push(next);
                let last = acc.last().unwrap();
                let next = last
                    .iter()
                    .zip(odd_ng.as_ref().iter())
                    .map(|(last, gate)| gate - last)
                    .collect::<Vec<_>>();
                acc.push(next);
                acc
            },
        );
        // Should have added one extra.
        assert_eq!(
            base_vecs.len(),
            2 * even_ngs.len() + 1,
            "Size of base vector not expected."
        );
        // Should come full cycle, dont need last entry.
        assert_eq!(
            base_vecs.pop(),
            Some(vec![0; k as usize]),
            "Did not return to 0."
        );

        // Now each gets iterated from 0 to max_d1 to find all overlaps.
        let n1 = d1s.len();
        let num_iters = n1.pow(k as u32);
        (0..num_iters)
            .map(|ci| {
                let (_, cvec) = (0..k).fold((ci, vec![]), |(ci, mut acc), _| {
                    let cx = ci % n1;
                    let ci = ci / n1;
                    acc.push(cx);
                    (ci, acc)
                });
                // Now go across sites
                base_vecs
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(x, mut site_ns)| {
                        let res =
                            site_ns
                                .iter_mut()
                                .zip(cvec.iter())
                                .try_for_each(|(nalpha, calpha)| {
                                    if x % 2 == 0 {
                                        *nalpha += calpha
                                    } else if *nalpha >= *calpha {
                                        *nalpha -= calpha
                                    } else {
                                        return Err(());
                                    }
                                    Ok(())
                                });
                        // If one went below 0, then no dof.
                        if res.is_err() {
                            return 0;
                        }
                        let cycles = &cycle_mat[even_perm(x)][odd_perm(x)];

                        let all_cycles_same_n = cycles
                            .iter()
                            .map(|cycle| {
                                let alpha = *cycle.first().unwrap();
                                let n = site_ns[alpha];
                                cycle.iter().copied().all(|alpha| site_ns[alpha] == n)
                            })
                            .all(|x| x);
                        if all_cycles_same_n {
                            cycles
                                .iter()
                                .map(|cycle| {
                                    let alpha = *cycle.first().unwrap();
                                    let n = site_ns[alpha];
                                    d1(n)
                                })
                                .product::<usize>()
                        } else {
                            0
                        }
                    })
                    .product::<usize>()
            })
            .sum::<usize>()
    } else {
        0
    }
}

fn overlap_uniform(
    even_ngs: &[Vec<usize>],
    odd_ngs: &[Vec<usize>],
    even_perm: usize,
    odd_perm: usize,
    d1s: &[usize],
    k: u8,
    cycle_mat: &[Vec<Vec<Vec<usize>>>],
) -> usize {
    let d1 = |n: usize| -> usize {
        if n >= d1s.len() {
            0
        } else {
            d1s[n]
        }
    };

    if get_n_sector(even_ngs, k) == get_n_sector(odd_ngs, k) {
        let cycles = &cycle_mat[even_perm][odd_perm];
        // let even_perm = &cycle_vec[even_perm];
        // let odd_perm = &cycle_vec[odd_perm];
        // let mut inverse_odd = vec![0usize; k as usize];
        // odd_perm.iter().copied().enumerate().for_each(|(i, j)| inverse_odd[j] = i);

        // Make a set of vectors which satisfy both even_ngs and odd_ngs.
        let mut base_vecs = (even_ngs.iter().zip(odd_ngs.iter())).fold(
            vec![vec![0; k as usize]],
            |mut acc, (even_ng, odd_ng)| {
                let last = acc.last().unwrap();
                let next = last
                    .iter()
                    .zip(even_ng.iter())
                    .map(|(last, gate)| gate - last)
                    .collect::<Vec<_>>();
                acc.push(next);
                let last = acc.last().unwrap();
                let next = last
                    .iter()
                    .zip(odd_ng.iter())
                    .map(|(last, gate)| gate - last)
                    .collect::<Vec<_>>();
                acc.push(next);
                acc
            },
        );
        // Should have added one extra.
        assert_eq!(base_vecs.len(), 2 * even_ngs.len() + 1);
        // Should come full cycle, dont need last entry.
        assert_eq!(base_vecs.pop(), Some(vec![0; k as usize]));

        // Now each gets iterated from 0 to max_d1 to find all overlaps.
        let n1 = d1s.len();
        let num_iters = n1.pow(k as u32);
        (0..num_iters)
            .map(|ci| {
                let (_, cvec) = (0..k).fold((ci, vec![]), |(ci, mut acc), _| {
                    let cx = ci % n1;
                    let ci = ci / n1;
                    acc.push(cx);
                    (ci, acc)
                });
                // Now go across sites
                base_vecs
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(x, mut site_ns)| {
                        let res =
                            site_ns
                                .iter_mut()
                                .zip(cvec.iter())
                                .try_for_each(|(nalpha, calpha)| {
                                    if x % 2 == 0 {
                                        *nalpha += calpha
                                    } else if *nalpha >= *calpha {
                                        *nalpha -= calpha
                                    } else {
                                        return Err(());
                                    }
                                    Ok(())
                                });
                        // If one went below 0, then no dof.
                        if res.is_err() {
                            return 0;
                        }

                        let all_cycles_same_n = cycles
                            .iter()
                            .map(|cycle| {
                                let alpha = *cycle.first().unwrap();
                                let n = site_ns[alpha];
                                cycle.iter().copied().all(|alpha| site_ns[alpha] == n)
                            })
                            .all(|x| x);
                        if all_cycles_same_n {
                            cycles
                                .iter()
                                .map(|cycle| {
                                    let alpha = *cycle.first().unwrap();
                                    let n = site_ns[alpha];
                                    d1(n)
                                })
                                .product::<usize>()
                        } else {
                            0
                        }
                    })
                    .product::<usize>()
            })
            .sum::<usize>()
    } else {
        0
    }
}

fn cycles_for_perm(perm: &[usize]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; perm.len()];
    let mut all_cycles = vec![];
    for mut i in 0..perm.len() {
        let mut cycle = vec![];
        while !seen[i] {
            seen[i] = true;
            cycle.push(i);
            i = perm[i];
        }
        if !cycle.is_empty() {
            all_cycles.push(cycle);
        }
    }
    all_cycles
}

#[pyfunction]
fn gen_cycles_for_perm(perm: Vec<usize>) -> Vec<Vec<usize>> {
    cycles_for_perm(&perm)
}

#[pyfunction]
fn make_cycles_mat(k: u8) -> Vec<Vec<Vec<Vec<usize>>>> {
    let alphas = (0..k as usize).collect::<Vec<_>>();
    let mut all_cycles = vec![];
    for sigma in alphas.iter().permutations(k as usize) {
        let mut sigma_cycles = vec![];
        for tau in alphas.iter().permutations(k as usize) {
            let mut inv_tau = vec![0usize; k as usize];
            for (i, alpha) in tau.into_iter().copied().enumerate() {
                inv_tau[alpha] = i;
            }
            let invtau_sigma = (0..k as usize)
                .map(|alpha| inv_tau[*sigma[alpha]])
                .collect::<Vec<_>>();
            let cycles = cycles_for_perm(&invtau_sigma);
            sigma_cycles.push(cycles);
        }
        all_cycles.push(sigma_cycles);
    }
    all_cycles
}

fn make_overlap_matrix_raw(
    l: usize,
    k: u8,
    d1s: Vec<usize>,
    n_sector: Option<Vec<usize>>,
) -> Result<Array2<f64>, String> {
    if l % 2 == 1 {
        return Err("L must be even".to_string());
    }
    let d2s = make_d2s(&d1s);
    let num_gate_ns = d2s.len();
    // Valid for even and odd, just interpreted slightly differently.
    let states = make_states(l, k, num_gate_ns, n_sector);

    let norms = states
        .iter()
        .map(|v| get_norm_squared(v.iter().map(|(v, _)| v.iter().copied()), &d2s))
        .map(|norm2| (norm2 as f64).sqrt())
        .collect::<Vec<_>>();

    let cycles_mat = make_cycles_mat(k);

    let n_states = states.len();
    let mut mat = Array2::<f64>::zeros((n_states, n_states));
    ndarray::Zip::indexed(&mut mat)
        .into_par_iter()
        .for_each(|((i, j), x)| {
            let (even_ngs, even_perms): (Vec<_>, Vec<_>) =
                states[i].iter().map(|(a, b)| (a, b)).unzip();
            let (odd_ngs, odd_perms): (Vec<_>, Vec<_>) =
                states[j].iter().map(|(a, b)| (a, b)).unzip();
            *x = overlap(
                &even_ngs,
                &odd_ngs,
                &even_perms,
                &odd_perms,
                &d1s,
                k,
                &cycles_mat,
            ) as f64
                / (norms[i] * norms[j]);
        });
    Ok(mat)
}

fn make_uniform_perm_overlap_matrix_raw(
    l: usize,
    k: u8,
    d1s: Vec<usize>,
    n_sector: Option<Vec<usize>>,
) -> Result<Array2<f64>, String> {
    if l % 2 == 1 {
        return Err("L must be even".to_string());
    }
    let d2s = make_d2s(&d1s);
    let num_gate_ns = d2s.len();
    // Valid for even and odd, just interpreted slightly differently.
    let states = make_uniform_states(l, k, num_gate_ns, n_sector);

    let norms = states
        .iter()
        .map(|(ngs, _)| get_norm_squared(ngs.iter().map(|v| v.iter().copied()), &d2s))
        .map(|norm2| (norm2 as f64).sqrt())
        .collect::<Vec<_>>();

    let cycles_mat = make_cycles_mat(k);

    let n_states = states.len();
    let mut mat = Array2::<f64>::zeros((n_states, n_states));
    ndarray::Zip::indexed(&mut mat)
        .into_par_iter()
        .for_each(|((i, j), x)| {
            let (even_ngs, even_perm) = &states[i];
            let (odd_ngs, odd_perm) = &states[j];
            *x = overlap_uniform(
                even_ngs,
                odd_ngs,
                *even_perm,
                *odd_perm,
                &d1s,
                k,
                &cycles_mat,
            ) as f64
                / (norms[i] * norms[j]);
        });
    Ok(mat)
}

#[pyfunction]
fn make_overlap_matrix(
    py: Python,
    l: usize,
    k: u8,
    d1s: Vec<usize>,
    n_sector: Option<Vec<usize>>,
) -> PyResult<Py<PyArray2<f64>>> {
    make_overlap_matrix_raw(l, k, d1s, n_sector)
        .map_err(PyValueError::new_err)
        .map(|mat| mat.to_pyarray(py).to_owned())
}

#[pyfunction]
fn make_uniform_perm_overlap_matrix(
    py: Python,
    l: usize,
    k: u8,
    d1s: Vec<usize>,
    n_sector: Option<Vec<usize>>,
) -> PyResult<Py<PyArray2<f64>>> {
    make_uniform_perm_overlap_matrix_raw(l, k, d1s, n_sector)
        .map_err(PyValueError::new_err)
        .map(|mat| mat.to_pyarray(py).to_owned())
}

fn fact(k: u8) -> usize {
    match k {
        k if k <= 1 => 1,
        k => k as usize * fact(k - 1),
    }
}

#[pymodule]
fn py_tiamat(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_uniform_perm_overlap_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(make_overlap_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(generate_d2s, m)?)?;
    m.add_function(wrap_pyfunction!(get_n_states, m)?)?;
    m.add_function(wrap_pyfunction!(get_n_uniform_states, m)?)?;
    m.add_function(wrap_pyfunction!(generate_uniform_states, m)?)?;
    m.add_function(wrap_pyfunction!(generate_states, m)?)?;
    m.add_function(wrap_pyfunction!(gen_cycles_for_perm, m)?)?;
    m.add_function(wrap_pyfunction!(make_cycles_mat, m)?)?;
    Ok(())
}

#[cfg(test)]
mod libtests {
    use super::*;

    #[test]
    fn test_overlap() {
        let res = make_overlap_matrix_raw(4, 3, vec![4, 4], Some(vec![0, 0, 0]));
    }
}