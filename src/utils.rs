use ndarray::{Array2, ArrayView2, ArrayView3, LinalgScalar};
use num_complex::Complex;
use rand::prelude::*;
use smallvec::*;
use std::ops::Mul;

// From github user notmgsk
pub fn kron<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: LinalgScalar,
{
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;
    let mut out = Array2::zeros((dimout, dimout));
    for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        let v: Array2<T> = Array2::from_elem((dimb, dimb), *(elem)) * b;
        chunk.assign(&v);
    }
    out
}

// From section 2.3 of http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
/// Make a random 2x2 unitary matrix.
pub fn make_unitary<R: Rng>(rng: &mut R) -> [Complex<f64>; 4] {
    let two_pi = std::f64::consts::PI * 2.0;
    let alpha: f64 = rng.gen::<f64>() * two_pi;
    let psi: f64 = rng.gen::<f64>() * two_pi;
    let chi: f64 = rng.gen::<f64>() * two_pi;
    let xi: f64 = rng.gen::<f64>();
    let phi = xi.sqrt().asin();

    let ei_alpha = Complex::from_polar(1.0, alpha);
    let ei_psi = Complex::from_polar(1.0, psi);
    let ei_chi = Complex::from_polar(1.0, chi);
    let (phi_s, phi_c) = phi.sin_cos();
    [
        ei_alpha * ei_psi * phi_c,
        ei_alpha * ei_chi * phi_s,
        -ei_alpha * ei_chi.conj() * phi_s,
        ei_alpha * ei_psi.conj() * phi_c,
    ]
}

pub fn enumerate_states<const N: usize>(sites: usize, defects: usize) -> Vec<SmallVec<[usize; N]>> {
    if defects > 0 {
        let mut states = vec![];
        enumerate_rec(&mut states, smallvec![], defects - 1, 0, sites);
        states
    } else {
        vec![smallvec![]]
    }
}

pub fn enumerate_rec<const N: usize>(
    acc: &mut Vec<SmallVec<[usize; N]>>,
    prefix: SmallVec<[usize; N]>,
    loop_num: usize,
    min_val: usize,
    max_val: usize,
) {
    if loop_num == 0 {
        for i in min_val..max_val {
            let mut state = prefix.clone();
            state.push(i);
            acc.push(state);
        }
    } else {
        for i in min_val..max_val - loop_num {
            let mut state = prefix.clone();
            state.push(i);
            enumerate_rec(acc, state, loop_num - 1, i + 1, max_val);
        }
    }
}
