#![feature(generic_const_exprs)]

mod frc;

use nalgebra::{
    allocator::Allocator, ArrayStorage, Const, DefaultAllocator, DimMin, SMatrix, ToTypenum,
};
use std::time::Instant;

fn discretize_ab<const States: usize, const Inputs: usize>(
    contA: &SMatrix<f64, States, States>,
    contB: &SMatrix<f64, States, Inputs>,
    dt: f64,
    discA: &mut SMatrix<f64, States, States>,
    discB: &mut SMatrix<f64, States, Inputs>,
) where
    Const<{ States + Inputs }>:
        ToTypenum + DimMin<Const<{ States + Inputs }>, Output = Const<{ States + Inputs }>>,
    DefaultAllocator: Allocator<
        f64,
        Const<{ States + Inputs }>,
        Const<{ States + Inputs }>,
        Buffer = ArrayStorage<f64, { States + Inputs }, { States + Inputs }>,
    >,
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
    let mut contA = SMatrix::<f64, 5, 5>::new(
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
    *Q = SMatrix::<f64, 5, 5>::new(
        256.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 64.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.16, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.10803324099723, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.10803324099723,
    );
    *R = SMatrix::<f64, 2, 2>::new(
        0.006944444444444444, 0.0,
        0.0, 0.006944444444444444,
    );

    const VELOCITY: f64 = 2.0;
    contA[(1, 2)] = VELOCITY;

    discretize_ab::<5, 2>(&contA, &contB, 0.005, A, B);
}

fn main() {
    let mut A = SMatrix::<f64, 5, 5>::zeros();
    let mut B = SMatrix::<f64, 5, 2>::zeros();
    let mut Q = SMatrix::<f64, 5, 5>::zeros();
    let mut R = SMatrix::<f64, 2, 2>::zeros();
    init_args(&mut A, &mut B, &mut Q, &mut R);

    let start = Instant::now();
    let S = frc::dare::<5, 2>(&A, &B, &Q, &R);
    let end = Instant::now();

    println!("S = {}", S);

    println!("elapsed = {} us", (end - start).as_secs_f64() * 1e6);
}
