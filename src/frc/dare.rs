use nalgebra::DMatrix;

// Works cited:
//
// [1] E. K.-W. Chu, H.-Y. Fan, W.-W. Lin & C.-S. Wang "Structure-Preserving
//     Algorithms for Periodic Discrete-Time Algebraic Riccati Equations",
//     International Journal of Control, 77:8, 767-788, 2004.
//     DOI: 10.1080/00207170410001714988

pub fn dare(
    A: &DMatrix<f64>,
    B: &DMatrix<f64>,
    Q: &DMatrix<f64>,
    R: &DMatrix<f64>,
) -> DMatrix<f64> {
    // Implements the SDA algorithm on page 5 of [1].

    // A₀ = A
    let mut A_k = A.clone();

    // G₀ = BR⁻¹Bᵀ
    //
    // See equation (4) of [1].
    let mut G_k = B * &R.clone().cholesky().unwrap().solve(&B.transpose());

    // H₀ = Q
    //
    // See equation (4) of [1].
    let mut H_k;
    let mut H_k1 = Q.clone();

    loop {
        H_k = H_k1;

        // W = I + GₖHₖ
        let W = DMatrix::<f64>::identity(A.nrows(), A.ncols()) + G_k.clone() * H_k.clone();

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
        G_k += A_k.clone() * V_2 * A_k.transpose();

        // Hₖ₊₁ = Hₖ + V₁ᵀHₖAₖ
        H_k1 = H_k.clone() + V_1.transpose() * H_k.clone() * A_k.clone();

        // Aₖ₊₁ = AₖV₁
        A_k *= V_1;

        // while |Hₖ₊₁ − Hₖ| > ε |Hₖ₊₁|
        if (H_k1.clone() - H_k).norm() <= 1e-10f64 * H_k1.norm() {
            break;
        }
    }

    H_k1
}
