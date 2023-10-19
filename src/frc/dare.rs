use nalgebra::{
    allocator::{Allocator, SameShapeAllocator},
    constraint::{AreMultipliable, SameNumberOfRows, ShapeConstraint},
    ArrayStorage, Const, DefaultAllocator, DimMin, DimMinimum, SMatrix,
};

// Works cited:
//
// [1] E. K.-W. Chu, H.-Y. Fan, W.-W. Lin & C.-S. Wang "Structure-Preserving
//     Algorithms for Periodic Discrete-Time Algebraic Riccati Equations",
//     International Journal of Control, 77:8, 767-788, 2004.
//     DOI: 10.1080/00207170410001714988

pub fn dare<const States: usize, const Inputs: usize>(
    A: &SMatrix<f64, States, States>,
    B: &SMatrix<f64, States, Inputs>,
    Q: &SMatrix<f64, States, States>,
    R: &SMatrix<f64, Inputs, Inputs>,
) -> SMatrix<f64, States, States>
where
    DefaultAllocator:
        Allocator<f64, Const<States>, Const<States>, Buffer = ArrayStorage<f64, States, States>>,
    DefaultAllocator:
        Allocator<f64, Const<States>, Const<Inputs>, Buffer = ArrayStorage<f64, States, Inputs>>,
    DefaultAllocator:
        SameShapeAllocator<f64, Const<States>, Const<States>, Const<States>, Const<States>>,
    DefaultAllocator: Allocator<(usize, usize), DimMinimum<Const<States>, Const<States>>>,

    ShapeConstraint: AreMultipliable<Const<States>, Const<Inputs>, Const<Inputs>, Const<States>>,
    ShapeConstraint: SameNumberOfRows<Const<Inputs>, Const<Inputs>>,
    ShapeConstraint: SameNumberOfRows<Const<States>, Const<States>, Representative = Const<States>>,

    Const<States>: DimMin<Const<States>, Output = Const<States>>,
{
    // Implements the SDA algorithm on page 5 of [1].

    // A₀ = A
    let mut A_k = A.clone();

    // G₀ = BR⁻¹Bᵀ
    //
    // See equation (4) of [1].
    let mut G_k = B * &R.cholesky().unwrap().solve(&B.transpose());

    // H₀ = Q
    //
    // See equation (4) of [1].
    let mut H_k;
    let mut H_k1 = Q.clone();

    loop {
        H_k = H_k1;

        // W = I + GₖHₖ
        let W = SMatrix::<f64, States, States>::identity() + G_k * H_k;

        let W_solver = W.lu();

        // Solve WV₁ = Aₖ for V₁
        let V_1 = W_solver.solve(&A_k).unwrap();

        // Solve V₂Wᵀ = Gₖ for V₂
        //
        // We want to put V₂Wᵀ = Gₖ into Ax = b form so we can solve it more
        // efficiently.
        //
        // V₂Wᵀ = Gₖ
        // (V₂Wᵀ)ᵀ = Gₖᵀ
        // WV₂ᵀ = Gₖᵀ
        //
        // The solution of Ax = b can be found via x = A.solve(b).
        //
        // V₂ᵀ = W.solve(Gₖᵀ)
        // V₂ = W.solve(Gₖᵀ)ᵀ
        let V_2 = W_solver.solve(&G_k.transpose()).unwrap().transpose();

        // Gₖ₊₁ = Gₖ + AₖV₂Aₖᵀ
        G_k += A_k * V_2 * A_k.transpose();

        // Hₖ₊₁ = Hₖ + V₁ᵀHₖAₖ
        H_k1 = H_k + V_1.transpose() * H_k * A_k;

        // Aₖ₊₁ = AₖV₁
        A_k *= V_1;

        // while |Hₖ₊₁ − Hₖ| > ε |Hₖ₊₁|
        if (H_k1 - H_k).norm() <= 1e-10f64 * H_k1.norm() {
            break;
        }
    }

    H_k1
}
