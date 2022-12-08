use crate::utils::{enumerate_states, make_unitary};
use ndarray::linalg::kron;
use ndarray::{s, Array2, Array3, ArrayView3, Axis};
use num_complex::Complex;
use num_traits::One;
use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct CircuitSamples {
    l: usize,
    samples: Array3<Complex<f64>>,
}

#[pymethods]
impl CircuitSamples {
    #[new]
    pub fn new(samples: usize, l: usize) -> Self {
        let mut sample_array = Array3::<Complex<f64>>::zeros((samples, 1 << l, 1 << l));
        sample_array
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|sample| {
                let mut rng = thread_rng();
                let res = make_even_layer(l, &mut rng);
                let res = res.dot(&make_odd_layer(l, &mut rng));
                let res = res.dot(&make_even_layer(l, &mut rng));
                sample
                    .into_iter()
                    .zip(res.into_iter())
                    .for_each(|(x, v)| *x = v);
            });
        Self {
            l,
            samples: sample_array,
        }
    }

    pub fn get_num_sector_states(&self, nsector: Vec<usize>, nbarsector: Vec<usize>) -> usize {
        let nstates = nsector
            .into_par_iter()
            .map(|n| enumerate_states::<8>(self.l, n))
            .map(|x| x.len())
            .product::<usize>();
        let nbstates = nbarsector
            .into_par_iter()
            .map(|nb| enumerate_states::<8>(self.l, nb))
            .map(|x| x.len())
            .product::<usize>();

        nstates * nbstates
    }

    pub fn get_nsector_states(&self, nsector: PyReadonlyArray1<usize>) -> Vec<Vec<Vec<usize>>> {
        nsector
            .as_array()
            .into_par_iter()
            .map(|n| {
                enumerate_states::<8>(self.l, *n)
                    .into_iter()
                    .map(|x| x.into_vec())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    pub fn get_nsector_submatrix(
        &self,
        py: Python,
        nsector: Vec<usize>,
        nbarsector: Vec<usize>,
    ) -> PyResult<Py<PyArray2<Complex<f64>>>> {
        let k = nsector.len();
        if k != nbarsector.len() {
            return Err(PyValueError::new_err(format!(
                "k values must agree ({} vs {})",
                k,
                nbarsector.len()
            )));
        }

        let res = get_submatrix_raw(k, self.l, &nsector, &nbarsector, self.samples.view());
        Ok(res.to_pyarray(py).to_owned())
    }
}

fn get_submatrix_raw(
    k: usize,
    l: usize,
    nsector: &[usize],
    nbarsector: &[usize],
    samples: ArrayView3<Complex<f64>>,
) -> Array2<Complex<f64>> {
    let nstates = nsector
        .into_par_iter()
        .copied()
        .map(|n| {
            enumerate_states::<8>(l, n)
                .into_iter()
                .map(|x| multidefect_to_binary(l, &x))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let nbstates = nbarsector
        .into_par_iter()
        .copied()
        .map(|nb| {
            enumerate_states::<8>(l, nb)
                .into_iter()
                .map(|x| multidefect_to_binary(l, &x))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    make_kron_matrix(l, k, &nstates, &nbstates, samples)
}

fn make_kron_matrix(
    l: usize,
    k: usize,
    nsector_subindices: &[Vec<usize>],
    nbsector_subindices: &[Vec<usize>],
    samples: ArrayView3<Complex<f64>>,
) -> Array2<Complex<f64>> {
    let nstates = nsector_subindices
        .iter()
        .map(|x| x.len())
        .product::<usize>();
    let nbstates = nbsector_subindices
        .iter()
        .map(|x| x.len())
        .product::<usize>();
    let n_states = nstates * nbstates;

    let mut res = Array2::<Complex<f64>>::zeros((n_states, n_states));
    let mask = make_mask(l);
    ndarray::Zip::indexed(&mut res).par_for_each(|(i, j), x| {
        let i_nb = i % nbstates;
        let i_n = i / nbstates;
        let j_nb = j % nbstates;
        let j_n = j / nbstates;

        let bin_in = construct_index(i_n, l, nsector_subindices);
        let bin_inb = construct_index(i_nb, l, nbsector_subindices);
        let bin_jn = construct_index(j_n, l, nsector_subindices);
        let bin_jnb = construct_index(j_nb, l, nbsector_subindices);

        let avg = samples
            .axis_iter(Axis(0))
            .map(|arr| {
                let mut prod = Complex::one();
                for i in 0..k {
                    let ui = (bin_in >> (l * i)) & mask;
                    let uj = (bin_jn >> (l * i)) & mask;
                    prod *= arr[(ui, uj)];

                    let usi = (bin_inb >> (l * i)) & mask;
                    let usj = (bin_jnb >> (l * i)) & mask;
                    prod *= arr[(usi, usj)].conj();
                }
                prod
            })
            .sum::<Complex<f64>>()
            / samples.shape()[0] as f64;
        *x = avg;
    });
    res
}

#[inline]
fn make_mask(l: usize) -> usize {
    (1 << l) - 1
}

fn construct_index(
    mut global_index: usize,
    chunk_size: usize,
    sub_indices: &[Vec<usize>],
) -> usize {
    debug_assert!({
        let max_index = sub_indices.iter().map(|x| x.len()).product::<usize>();
        global_index < max_index
    });

    let mut index = 0;
    for sub in sub_indices {
        let binary_subindex = sub[global_index % sub.len()];
        global_index /= sub.len();
        index <<= chunk_size;
        index |= binary_subindex;
    }
    index
}

fn multidefect_to_binary(l: usize, x: &[usize]) -> usize {
    let mut r = 0usize;
    x.iter().for_each(|i| {
        r |= 1 << (l - *i - 1);
    });
    r
}

fn make_two_body<R: Rng>(rng: &mut R) -> Array2<Complex<f64>> {
    let mut res = Array2::zeros((4, 4));
    let aa = Complex::from_polar(1.0, rng.gen::<f64>() * std::f64::consts::FRAC_2_PI);
    let u = make_unitary(rng);
    let bb = Complex::from_polar(1.0, rng.gen::<f64>() * std::f64::consts::FRAC_2_PI);

    res[(0, 0)] = aa;
    res.slice_mut(s![1..=2, 1..=2])
        .iter_mut()
        .zip(u.into_iter())
        .for_each(|(x, c)| *x = c);
    res[(3, 3)] = bb;
    res
}

fn make_even_layer<R: Rng>(l: usize, rng: &mut R) -> Array2<Complex<f64>> {
    let arr = make_two_body(rng);
    let evens = (2..l - 1).step_by(2).fold(arr, |acc, _| {
        let arr = make_two_body(rng);
        kron(&acc, &arr)
    });
    if l % 2 == 1 {
        kron(&evens, &Array2::eye(2))
    } else {
        evens
    }
}

fn make_odd_layer<R: Rng>(l: usize, rng: &mut R) -> Array2<Complex<f64>> {
    let arr = Array2::eye(2);
    let odds = (1..l - 1).step_by(2).fold(arr, |acc, _| {
        let arr = make_two_body(rng);
        kron(&acc, &arr)
    });
    if l % 2 == 0 {
        kron(&odds, &Array2::eye(2))
    } else {
        odds
    }
}

#[cfg(test)]
mod test_stuff {
    use super::*;
    use ndarray::{Array, ShapeError};

    #[test]
    fn check_sizes() {
        let mut rng = SmallRng::from_entropy();
        let even = make_even_layer(4, &mut rng);
        assert_eq!(even.shape(), &[2usize.pow(4), 2usize.pow(4)]);
        let odd = make_odd_layer(4, &mut rng);
        assert_eq!(odd.shape(), &[2usize.pow(4), 2usize.pow(4)]);
    }

    #[test]
    fn check_sizes_odd() {
        let mut rng = SmallRng::from_entropy();
        let even = make_even_layer(3, &mut rng);
        assert_eq!(even.shape(), &[2usize.pow(3), 2usize.pow(3)]);
        let odd = make_odd_layer(3, &mut rng);
        assert_eq!(odd.shape(), &[2usize.pow(3), 2usize.pow(3)]);
    }

    #[test]
    fn check_convert_high() {
        let x = [0, 1, 3];
        let b = multidefect_to_binary(4, &x);
        assert_eq!(b, 0b1101);
    }

    #[test]
    fn check_convert_low() {
        let x = [0, 2, 3];
        let b = multidefect_to_binary(4, &x);
        assert_eq!(b, 0b1011);
    }

    #[test]
    fn check_convert_empty() {
        let x = [];
        let b = multidefect_to_binary(4, &x);
        assert_eq!(b, 0b0000);
    }

    #[test]
    fn check_constructed_index() {
        let binary_sub = vec![vec![0b00, 0b01, 0b10, 0b11], vec![0b00, 0b01, 0b10]];
        for i in 0..(4 * 3) {
            let r = construct_index(i, 2, &binary_sub);
            println!("{r:4b}");
        }
    }

    #[test]
    fn check_mask() {
        assert_eq!(make_mask(4), 0b1111);
    }

    #[test]
    fn test_kron() -> Result<(), ShapeError> {
        let mat = (0..16)
            .map(|x| Complex::from_polar(x as f64, 0.0))
            .collect::<Vec<_>>();
        let mat = Array3::from_shape_vec((1, 4, 4), mat)?;
        let res = make_kron_matrix(
            2,
            1,
            &[vec![0b00, 0b01, 0b10, 0b11]],
            &[vec![0b00, 0b01, 0b10, 0b11]],
            mat.view(),
        );
        println!("{:?}", res);
        let m = mat.index_axis(Axis(0), 0);
        assert_eq!(res, kron(&m, &m));
        Ok(())
    }

    #[test]
    fn make_samples() {
        let s = CircuitSamples::new(1024, 3);
        let sub = get_submatrix_raw(1, 3, &[0], &[1], s.samples.view());
    }
}
