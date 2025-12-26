mod frc;

use nalgebra::{DMatrix, dmatrix};
use std::time::Instant;

fn discretize_ab(
    cont_A: &DMatrix<f64>,
    cont_B: &DMatrix<f64>,
    dt: f64,
    disc_A: &mut DMatrix<f64>,
    disc_B: &mut DMatrix<f64>,
) {
    let States = cont_A.nrows();
    let Inputs = cont_B.ncols();

    // M = [A  B]
    //     [0  0]
    let mut M = DMatrix::<f64>::zeros(States + Inputs, States + Inputs);
    M.view_mut((0, 0), (States, States)).copy_from(cont_A);
    M.view_mut((0, States), (States, Inputs)).copy_from(cont_B);

    // ϕ = eᴹᵀ = [A_d  B_d]
    //           [ 0    I ]
    let phi = (M * dt).exp();

    *disc_A = phi.view((0, 0), (States, States)).into();
    *disc_B = phi.view((0, States), (States, Inputs)).into();
}

#[rustfmt::skip]
fn init_args(
    A: &mut DMatrix<f64>,
    B: &mut DMatrix<f64>,
    Q: &mut DMatrix<f64>,
    R: &mut DMatrix<f64>) {
    let mut cont_A = dmatrix![
        0.0, 0.0, 0.0, 0.5, 0.5;
        0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, -1.1111111111111112, 1.1111111111111112;
        0.0, 0.0, 0.0, -10.486221508345572, 5.782171664108812;
        0.0, 0.0, 0.0, 5.782171664108812, -10.486221508345572];
    let cont_B = dmatrix![
        0.0, 0.0;
        0.0, 0.0;
        0.0, 0.0;
        6.664631384780125, -5.106998986026231;
        -5.106998986026231, 6.664631384780125];
    *Q = dmatrix![
        256.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 64.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.16, 0.0, 0.0;
        0.0, 0.0, 0.0, 1.10803324099723, 0.0;
        0.0, 0.0, 0.0, 0.0, 1.10803324099723];
    *R = dmatrix![
        0.006944444444444444, 0.0;
        0.0, 0.006944444444444444];

    const VELOCITY: f64 = 2.0;
    cont_A[(1, 2)] = VELOCITY;

    discretize_ab(&cont_A, &cont_B, 0.005, A, B);
}

fn main() {
    let mut A = DMatrix::<f64>::zeros(5, 5);
    let mut B = DMatrix::<f64>::zeros(5, 2);
    let mut Q = DMatrix::<f64>::zeros(5, 5);
    let mut R = DMatrix::<f64>::zeros(2, 2);
    init_args(&mut A, &mut B, &mut Q, &mut R);

    let start = Instant::now();
    let S = frc::dare(&A, &B, &Q, &R);
    let end = Instant::now();

    println!("S = {}", S);

    println!("elapsed = {} us", (end - start).as_secs_f64() * 1e6);
}
