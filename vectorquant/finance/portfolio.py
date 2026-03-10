"""
Portfolio Theory
"""
import math
from vectorquant.core.linear_algebra import dot, matrix_multiply, transpose

def portfolio_return(weights, expected_returns):
    """
    R_p = w^T * E[R]
    """
    return dot(weights, expected_returns)

def portfolio_variance(weights, cov_matrix):
    """
    Var_p = w^T * Cov * w
    """
    w_matrix = [[w] for w in weights]
    w_t = transpose(w_matrix)
    temp = matrix_multiply(w_t, cov_matrix)
    var = matrix_multiply(temp, w_matrix)[0][0]
    return max(var, 0.0)

def portfolio_volatility(weights, cov_matrix):
    return math.sqrt(portfolio_variance(weights, cov_matrix))

def optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate=0.0, max_iter=1000, lr=0.01):
    """
    Uses gradient descent to find Max Sharpe portfolio weights.
    Constraint: sum(weights) = 1, weights >= 0
    Sharpe = (w^T * mu - rf) / sqrt(w^T * Cov * w)
    We minimize negative Sharpe ratio.
    """
    n = len(expected_returns)
    w = [1.0 / n] * n # Equal weight start
    
    for _ in range(max_iter):
        # Calculate current obj
        p_ret = portfolio_return(w, expected_returns)
        p_var = portfolio_variance(w, cov_matrix)
        p_vol = math.sqrt(p_var)
        
        if p_vol < 1e-8:
            break
            
        sharpe = (p_ret - risk_free_rate) / p_vol
        
        # Gradients (approximate via small finite difference to avoid complex analytical grad here)
        grad = []
        h = 1e-5
        for i in range(n):
            w_h = list(w)
            w_h[i] += h
            p_ret_h = portfolio_return(w_h, expected_returns)
            p_vol_h = math.sqrt(portfolio_variance(w_h, cov_matrix))
            sharpe_h = (p_ret_h - risk_free_rate) / p_vol_h if p_vol_h > 1e-8 else 0.0
            grad.append((sharpe_h - sharpe) / h)
            
        # Update weights (maximize Sharpe -> add gradient * lr)
        w = [w[i] + lr * grad[i] for i in range(n)]
        
        # Enforce non-negativity
        w = [max(wi, 0.0) for wi in w]
        
        # Enforce sum = 1
        s = sum(w)
        if s > 0:
            w = [wi / s for wi in w]
        else:
            w = [1.0 / n] * n # Reset if weights collapse
            
    return w

def black_litterman_returns(pi, cov_matrix, P, Q, tau=0.05, omega=None):
    """
    pi: Implied equilibrium returns
    P: Pick matrix (views to assets)
    Q: Vector of expected view returns
    tau: scalar
    omega: Covariance matrix of views (if None, inferred proportional to P * cov * P^T)
    mu = [(tau * Cov)^-1 + P^T * Omega^-1 * P]^-1 * [(tau * Cov)^-1 * pi + P^T * Omega^-1 * Q]
    """
    from vectorquant.core.linear_algebra import matrix_inverse, matrix_add, matrix_scale, pseudoinverse
    n = len(pi)
    tau_cov = matrix_scale(cov_matrix, tau)
    
    try:
        inv_tau_cov = matrix_inverse(tau_cov)
    except:
        inv_tau_cov = pseudoinverse(tau_cov)
        
    P_t = transpose(P)
    
    if omega is None:
        # Heuristic: Omega = diag(P * tau * Cov * P^T)
        _temp = matrix_multiply(P, tau_cov)
        _temp2 = matrix_multiply(_temp, P_t)
        omega = [[_temp2[i][j] if i == j else 0.0 for j in range(len(_temp2[0]))] for i in range(len(_temp2))]
        
    try:
        inv_omega = matrix_inverse(omega)
    except:
        inv_omega = pseudoinverse(omega)
        
    # Part 1: [(tau * Cov)^-1 + P^T * Omega^-1 * P]^-1
    pt_invO_p = matrix_multiply(matrix_multiply(P_t, inv_omega), P)
    part1_inv = matrix_add(inv_tau_cov, pt_invO_p)
    try:
        part1 = matrix_inverse(part1_inv)
    except:
        part1 = pseudoinverse(part1_inv)
        
    # Part 2: [(tau * Cov)^-1 * pi + P^T * Omega^-1 * Q]
    t1 = matrix_multiply(inv_tau_cov, [[x] for x in pi])
    t2 = matrix_multiply(matrix_multiply(P_t, inv_omega), [[x] for x in Q])
    part2 = matrix_add(t1, t2)
    
    # Final mu
    mu_mat = matrix_multiply(part1, part2)
    return [x[0] for x in mu_mat]
