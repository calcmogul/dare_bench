#![feature(generic_const_exprs)]

mod frc;

use nalgebra::SMatrix;

fn discretize_ab<const States: usize, const Inputs: usize>(
    contA: &SMatrix<f64, States, States>,
    contB: &SMatrix<f64, States, Inputs>,
    dt: f64,
    discA: &mut SMatrix<f64, States, States>,
    discB: &mut SMatrix<f64, States, Inputs>,
) where
    [(); States + Inputs]:,
{
    // M = [A  B]
    //     [0  0]
    let mut M = SMatrix::<f64, { States + Inputs }, { States + Inputs }>::zeros();
    // let mut M = SMatrix::<f64, 7, 7>::zeros();
    M.fixed_view_mut::<States, States>(0, 0).copy_from(contA);
    M.fixed_view_mut::<States, Inputs>(0, States)
        .copy_from(contB);

    // ϕ = eᴹᵀ = [A_d  B_d]
    //           [ 0    I ]
    let phi = (M * dt).exp();

    *discA = phi.fixed_view::<States, States>(0, 0).into();
    *discB = phi.fixed_view::<States, Inputs>(0, States).into();
}

#[rustfmt::skip]
fn init_args(
    A: &mut SMatrix<f64, 5, 5>,
    B: &mut SMatrix<f64, 5, 2>,
    Q: &mut SMatrix<f64, 5, 5>,
    R: &mut SMatrix<f64, 2, 2>,
) {
    let contA = SMatrix::<f64, 5, 5>::new(
        0.0, 0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, -1.1111111111111112, 1.1111111111111112,
        0.0, 0.0, 0.0, -10.486221508345572, 5.782171664108812,
        0.0, 0.0, 0.0, 5.782171664108812, -10.486221508345572,
    );
    let contB = SMatrix::<f64, 5, 2>::new(
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        6.664631384780125, -5.106998986026231,
        -5.106998986026231, 6.664631384780125,
    );
    let Q = SMatrix::<f64, 5, 5>::new(
        256.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 64.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.16, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.10803324099723, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.10803324099723,
    );
    let R = SMatrix::<f64, 2, 2>::new(
        0.006944444444444444, 0.0,
        0.0, 0.006944444444444444,
    );

    const velocity: f64 = 2.0;
    contA[(1, 2)] = velocity;

    discretize_ab::<5, 2>(&contA, &contB, 0.005, &mut A, &mut B);
}

fn main() {
    let A: SMatrix<f64, 5, 5>;
    let B: SMatrix<f64, 5, 2>;
    let Q: SMatrix<f64, 5, 5>;
    let R: SMatrix<f64, 2, 2>;
    init_args(&mut A, &mut B, &mut Q, &mut R);

    let S = frc::dare::<5, 2>(&A, &B, &Q, &R);
    println!("{}\n", S);
}
