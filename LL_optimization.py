import networkx as nx
import numpy as np
import scipy
from tqdm import tqdm


def f_a(x, A, alpha):
    xAx = x.T @ A @ x
    norm_sq = np.dot(x, x)               # ||x||^2
    sum_sq = np.sum(x) ** 2              # (1^T * x)^2
    return (xAx + alpha * norm_sq - alpha * sum_sq)/2

def proximal_hull_obj_fun(x, A, alpha, lam):
    return -f_a(x, A, alpha) + (1 / (2 * lam)) * np.sum(x * (1 - x))

def opt_proximal_hull(A, alpha, lam, x0, method, bounds):
    result = scipy.optimize.minimize(proximal_hull_obj_fun, x0, args=(A, alpha, lam), method=method, bounds=bounds)
    return result

def LL_double_envelope(x, lam, mu):
    """
    Compute the value of the function:
        h̄^{(λ, μ)}(x) as defined in equation (1)

    Parameters:
    - x: float or NumPy array
    - lam > mu > 0

    Returns:
    - float or NumPy array with the same shape as x
    """

    x = np.asarray(x)  # Ensure x can be vectorized

    thresh1 = (lam - mu) / (2 * lam)
    thresh2 = mu / lam + (lam - mu) / (2 * lam)

    out = np.where(
        x < thresh1,
        x**2 / (2 * (lam - mu)),
        np.where(
            x > thresh2,
            (x - 1)**2 / (2 * (lam - mu)),
            1 / (8 * lam) - (1 / (2 * mu)) * (x - 0.5)**2
        )
    )
    return out

def LL_obj_fun(x, A, alpha, lam, mu):
    return -f_a(x, A, alpha) + np.sum(LL_double_envelope(x, lam, mu))

def opt_LL(A, alpha, lam, mu, x0, method, bounds):
    result = scipy.optimize.minimize(LL_obj_fun, x0, args=(A, alpha, lam, mu), method=method, bounds=bounds)
    return result

def homotopy_LL(A, alpha, lam0, mu0, x0, method, bounds, tolerance=None, max_iters=100, dec_rate=0.9, dec_rate_tol=0.9):

    f_a_ = []
    is_binary_ = []

    lam, mu = lam0, mu0
    for _ in tqdm(range(max_iters)):
        # result = scipy.optimize.minimize(LL_obj_fun, x0, args=(A, alpha, lam, mu), method=method, bounds=bounds, tol=tolerance)
        result = scipy.optimize.minimize(LL_obj_fun, x0, args=(A, alpha, lam, mu), method=method, jac=g_jacobian, bounds=bounds, tol=tolerance)
        x0 = result.x
        tolerance = dec_rate_tol*tolerance
        lam, mu = dec_rate*lam, dec_rate*mu

        f_a_.append(f_a(result.x,A,alpha))
        # fun_values_.append(result.fun)
        is_binary_.append(np.array_equal(x0, x0.astype(bool)))

    return result, f_a_, is_binary_


########################################################################

def h_bar_double_prime(x, lam, mu):
    x = np.asarray(x)
    t1 = (lam - mu) / (2 * lam)
    t2 = mu / lam + (lam - mu) / (2 * lam)
    return np.where(
        (x < t1) | (x > t2),
        1 / (lam - mu),
        -1 / mu
    )

def h_bar_prime(x, lam, mu):
    x = np.asarray(x)
    t1 = (lam - mu) / (2 * lam)
    t2 = mu / lam + (lam - mu) / (2 * lam)
    return np.where(
        x < t1,
        x / (lam - mu),
        np.where(
            x > t2,
            (x - 1) / (lam - mu),
            -(x - 0.5) / mu
        )
    )

def h_bar_prime(x, lam, mu):
    """
    Compute h_bar_prime efficiently.
    """
    x = np.asarray(x)
    t1 = (lam - mu) / (2 * lam)
    t2 = mu / lam + (lam - mu) / (2 * lam)
    
    # Vectorized computation
    result = np.empty_like(x)
    
    # Region 1: x < t1
    mask1 = x < t1
    result[mask1] = x[mask1] / (lam - mu)
    
    # Region 3: x > t2  
    mask3 = x > t2
    result[mask3] = (x[mask3] - 1) / (lam - mu)
    
    # Region 2: t1 <= x <= t2
    mask2 = ~(mask1 | mask3)
    result[mask2] = -(x[mask2] - 0.5) / mu
    
    return result

# def g_jacobian(x, A, alpha, lam, mu):
#     """
#     Compute the Jacobian (gradient) of g(x).
#     """
#     n = len(x)
#     e = np.ones(n)
#     M = A + alpha * np.eye(n) - alpha * np.outer(e, e)
#     return -2 * M @ x + h_bar_prime(x, lam, mu)

def g_jacobian(x, A, alpha, lam, mu):
    """
    Compute the Jacobian (gradient) of g(x) without building dense matrices.
    """
    n = len(x)
    
    # Compute M*x without building M
    # M = A + alpha*I - alpha*e*e^T
    # So M*x = A*x + alpha*x - alpha*(e^T*x)*e
    
    Ax = A @ x  # Sparse matrix multiplication
    sum_x = np.sum(x)  # e^T * x
    Mx = Ax + alpha * x - alpha * sum_x
    
    # Compute h_bar_prime
    h_prime = h_bar_prime(x, lam, mu)
    
    return -2 * Mx + h_prime


def g_hessian(x, A, alpha, lam, mu):
    """
    Compute the Hessian of g(x).
    """
    n = len(x)
    e = np.ones(n)
    M = A + alpha * np.eye(n) - alpha * np.outer(e, e)
    H_diag = h_bar_double_prime(x, lam, mu)
    return -2 * M + np.diag(H_diag)
