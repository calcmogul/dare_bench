mod frc;

use nalgebra::{DMatrix, dmatrix};
use std::time::Instant;

fn discretize_ab(
    contA: &DMatrix<f64>,
    contB: &DMatrix<f64>,
    dt: f64,
    discA: &mut DMatrix<f64>,
    discB: &mut DMatrix<f64>,
) {
    let States = contA.nrows();
    let Inputs = contB.ncols();

    // M = [A  B]
    //     [0  0]
    let mut M = DMatrix::<f64>::zeros(States + Inputs, States + Inputs);
    M.view_mut((0, 0), (States, States)).copy_from(contA);
    M.view_mut((0, States), (States, Inputs)).copy_from(contB);

    // ϕ = eᴹᵀ = [A_d  B_d]
    //           [ 0    I ]
    let phi = (M * dt).exp();

    *discA = phi.view((0, 0), (States, States)).into();
    *discB = phi.view((0, States), (States, Inputs)).into();
}

#[rustfmt::skip]
fn init_args(
    A: &mut DMatrix<f64>,
    B: &mut DMatrix<f64>,
    Q: &mut DMatrix<f64>,
    R: &mut DMatrix<f64>) {
    let mut contA = dmatrix![
        0.0, 0.0, 0.0, 0.5, 0.5;
        0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, -1.1111111111111112, 1.1111111111111112;
        0.0, 0.0, 0.0, -10.486221508345572, 5.782171664108812;
        0.0, 0.0, 0.0, 5.782171664108812, -10.486221508345572];
    let contB = dmatrix![
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
    contA[(1, 2)] = VELOCITY;

    discretize_ab(&contA, &contB, 0.005, A, B);
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
