from msilib import datasizemask
from typing import Callable, Tuple
import numpy as np
from scipy.optimize import fsolve, root, approx_fprime
from scipy.linalg import norm, solve, lstsq, null_space
import const
from dynamical_system import full_system, dDelta_dt
import matplotlib.pyplot as plt

def real_imag_split_z(zlambd, N):
    #only need to split \eta
    eta = zlambd[N:2*N]
    eta_real = eta.real
    eta_imag = eta.imag
    params = zlambd[2*N:] if 2*N < len(zlambd) else []
    return np.concatenate([zlambd[:N].real, eta_real, eta_imag, params])

def recombine_z(zlambd, N):
    c = np.empty(N, dtype=np.complex128)
    c.real = zlambd[N:2*N]
    c.imag = zlambd[2*N:3*N]
    params = zlambd[3*N:] if 3*N < len(zlambd) else []
    return np.concatenate([zlambd[:N], c, params]) 

def construct_params(zlambd, other_parameters, N):
    independent_parms = zlambd[2*N:]
    constructed_parms = []
    i = 0
    for parm in other_parameters:
        if parm is None:
            constructed_parms.append(independent_parms[i])
            i += 1
        else:
            constructed_parms.append(parm)
    return tuple(constructed_parms)

def parameterized_system(zlambd, other_parameters, function_params, N, dz):
    z = zlambd[:2*N]
    constant_params = construct_params(zlambd, other_parameters, N)

    return full_system(z, constant_params, function_params, dz)

def parameterized_system_split(zlambd_split, other_parameters, function_params, N, dz):
    return real_imag_split_z(parameterized_system(recombine_z(zlambd_split, N), other_parameters, function_params, N, dz), N)

def compute_jacobian(zlambd, other_parameters, function_parameters, N, dz, epsilon=np.sqrt(np.finfo(float).eps)):
    """
    Compute the Jacobian matrix of F numerically via finite differences.
    This version handles complex variables by splitting into real/imaginary parts.
    """
    return approx_fprime(zlambd, parameterized_system_split, epsilon, other_parameters, function_parameters, N, dz)

def compute_tangent(zlambd, other_parameters, function_parameters, N, dz,epsilon=1e-8):
    jac = compute_jacobian(zlambd, other_parameters, function_parameters, N, dz, epsilon=epsilon)

    # comput the tangent on the solution curve
    nullspace = null_space(jac).T

    '''
    nonzero_mask = np.any(nullspace != 0, axis=1)  # Create a mask for arrays with nonzero elements
    nonzero_null_space = nullspace[nonzero_mask]
    positive_mask = nonzero_mask[nonzero_null_space[:, -3] >= 0 and nonzero_null_space[:, -2] >= 0 and nonzero_null_space[:, -1] >= 0]

    if np.any(nonzero_mask):
        if np.any(positive_mask):
            dzlambd = nullspace[positive_mask][0]
        else:
            dzlambd = nullspace[nonzero_mask][0]  # Return the first array with nonzero elements
            # Normalize the tangent vector (dz, dparams)
        norms = np.linalg.norm(dzlambd)

        dzlambd = dzlambd / norms #/ norms
    else:
        dzlambd = nullspace[0]
    '''
    dzlambd = np.sum(nullspace, axis=0) 
    dzlambd = dzlambd / np.linalg.norm(dzlambd)
    return dzlambd

def predictor_step(zlambd, dzlambd, ds):
    """
    Compute the next predicted point along the curve.
    z_real_imag: current point in real/imaginary form.
    params: current parameters.
    ds: step size.
    """
    zlambd_next = zlambd + (ds * dzlambd)
    return zlambd_next

def arclength_constraint(zlambd, zlambd_prev, dzlambd, ds, N):
    z = zlambd[:3*N]
    z_prev = zlambd_prev[:3*N]
    lambd = zlambd[3*N:]
    lambd_prev = zlambd_prev[3*N:]
    deltaz = z - z_prev
    deltalambd = lambd - lambd_prev
    dz = dzlambd[:3*N]
    dlambd = dzlambd[3*N:]
    
    return np.dot(deltaz, dz) + np.dot(deltalambd, dlambd) - ds

def augmented_system(zlambd, zlambd_prev, dzlambd, other_parameters, function_params, N, dz, ds):
    return np.concatenate([parameterized_system_split(zlambd, other_parameters, function_params, N, dz), [arclength_constraint(zlambd, zlambd_prev, dzlambd, ds, N)]])

def augmented_jacobian(zlambd, zlambd_old, function_params, N, dz, ds):
    F_jacobian = compute_jacobian(zlambd, other_parameters, function_params, N, dz)

    z = zlambd[:4*N]
    z_prev = zlambd_prev[:4*N]
    lambd = zlambd[4*N:]
    lambd_prev = zlambd_prev[4*N:]
    dz = z - z_prev
    dlambd = lambd - lambd_prev
    arc_jacobian =np.concatenate([2 * dz, 2 * dlambd])

    return np.vstack([F_jacobian, arc_jacobian])

def corrector_step(zlambd, zlambd_prev, dzlambd, other_parameters, function_params, N, dz, ds):
    sol = root(augmented_system, zlambd, args=(zlambd_prev, dzlambd, other_parameters, function_params, N, dz, ds), method='Krylov', options={'maxiter': 100})
    
    if sol.success:
        zlambd_next = sol.x
        return zlambd_next, True
    else:
        zlambd_next = zlambd
        return zlambd, False

def pseudo_arclength_continuation(zlambd_init, function_params, N, dz, ds, S_hat = None, delta_hat = None, kappa = None, gamma=None, max_steps=1000, jac_epislon=1e-8):
    """_summary_

    Args:
        zlambd_init (_type_): _description_
        function_params (_type_): _description_
        N (_type_): _description_
        dz (_type_): _description_
        ds (_type_): _description_
        S_hat (_type_, optional): _description_. Defaults to None.
        delta_hat (_type_, optional): _description_. Defaults to None.
        kappa (_type_, optional): _description_. Defaults to None.
        gamma (_type_, optional): _description_. Defaults to None.
        max_steps (int, optional): _description_. Defaults to 1000.
        jac_epislon (_type_, optional): _description_. Defaults to 1e-8.

    Returns:
        _type_: _description_
    """
    zlambd = real_imag_split_z(zlambd_init, N).astype(float)
    other_parameters = (S_hat, delta_hat, kappa, gamma)
    #interest_indicies = [i for i, param in enumerate(parameters) if param is None]

    results = []
    convergence = []
    for step in range(max_steps):
        # Compute tangent vector
        dzlambd = compute_tangent(zlambd, other_parameters, function_params, N, dz, epsilon=jac_epislon)
        
        #Predictor step
        zlambd_pred = predictor_step(zlambd, dzlambd, ds)
        
        # Corrector step
        zlambd, converged = corrector_step(zlambd_pred, zlambd, dzlambd, other_parameters, function_params, N, dz, ds)

        zlambd_complex = recombine_z(zlambd, N)

        results.append(zlambd_complex)
        convergence.append(converged)
        print(f'Step{step}/{max_steps}:{converged}')

    return np.array(results), np.array(convergence)


def compute_adomian_decomp_terms(Delta, Li, oc, dz, Fk, S_hat, delta_hat, number_of_terms):
    ''' Computes a number of terms in the adomian series
    :param Delta: value of Delta (at steady state?)
    :type Delta: np.array
    :param Li:T^k integral kernel
    :type Li: np.array
    :param oc: T^1 integral kernel
    :type oc: np.array
    :param z: integration/discretization step
    :type z: float
    :param Fk: forcing as array
    :type Fk: np.array
    :param S_hat: S_hat param
    :type S_hat: float
    param delta_hat: delta_hat param
    :type delta_hat: float
    :param number_of_terms: number of terms to compute
    :type number_of_terms: int
    :return: An array of each of the terms in the series (which are functional, hence a multiarray returned)
    :rtype: np.array
    '''

    lambd = 1/((oc @ Delta) * dz - delta_hat - 1/(1j * S_hat)) 
    term_0 = -Fk/(max(Fk)*S_hat)
    terms = [term_0]
    for i in range(number_of_terms):
        new_term = (Li @ (Delta * terms[-1]) * dz)
        terms.append(new_term)

    return np.array(terms), lambd


def inner_product(
        f: np.ndarray,
        g: np.ndarray, 
        R : Callable[[complex], complex] | np.ndarray, 
        Delta: np.ndarray,
        zg: np.ndarray
) -> complex:
    dz = zg[1] - zg[0]
    R = (np.vectorize(R)(zg) if callable(R) else R)
    return np.sum(np.exp(-zg) * R**2 * Delta * f * np.conj(g) * dz) 

#WIP
def compute_A_n(
    eigenvalue: complex,
    eigenfunction: np.ndarray,
    Delta: np.ndarray,
    S_k: float,
    delta_hat: float,
    F_k: np.ndarray,
    R: Callable[[complex], complex] | np.ndarray,
    zg: np.ndarray
) -> np.ndarray:
    """
    Computes the coefficient A_n based on the given parameters and eigenvalues/eigenfunctions.
    """
    # Inner product <F_k(z), phi_n> and <phi_n, phi_n>
    inner_Fk_phi_n = inner_product(1j * F_k/max(F_k), eigenfunction, R, Delta, zg)
    M = inner_product(eigenfunction, eigenfunction, R, Delta, zg)

    # Compute A_n using the given formula
    denom =  1j * S_k * (eigenvalue + delta_hat) - 1
    A_n_val = (inner_Fk_phi_n) / denom
    return A_n_val

def compute_eta_decomp(
    Delta: np.ndarray,
    T_k: np.ndarray,
    T1: np.ndarray,
    S_k: float,
    delta_hat: float,
    F_k: Callable[[Callable | np.ndarray, complex, int], complex] | np.ndarray,
    R: Callable[[complex], complex] | np.ndarray,
    zg: np.ndarray,
    k: int, 
    num_eigenvalues: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes eta(z) as a series expansion using A_n and eigenfunctions of L = Delta * T^k.
    """
    # Compute eigenvalues and eigenfunctions of the operator L = Delta * T^k
    dz = (zg[1] - zg[0])
    L_matrix = T_k * Delta * dz
    omega_c_matrix = np.diag(T1 @ Delta * dz)
    eigenvalues, eigenvectors = np.linalg.eig(L_matrix - omega_c_matrix)

    for i in range(eigenvectors.shape[1]):
        norm = np.sqrt(inner_product(eigenvectors[:, i], eigenvectors[:, i], R, Delta, zg))  # Inner product for norm
        eigenvectors[:, i] /= norm  # Normalize the eigenvector

    F_k = np.vectorize(F_k)(R, zg, k) if callable(F_k) else F_k
    num_eigenvalues = num_eigenvalues or len(zg)
    # Compute eta(z) as a sum of A_n * phi_n(z)
    eta_val = np.zeros_like(zg, dtype=complex)
    A_ns = np.zeros(num_eigenvalues, dtype = np.ndarray)
    for n in range(min(num_eigenvalues, len(eigenvalues))):
        A_n = compute_A_n(eigenvalues[n], eigenvectors[:, n], Delta, S_k, delta_hat, F_k, R, zg)
        eta_val += A_n * eigenvectors[:, n]
        A_ns[n] = A_n

    return eta_val, A_ns, eigenvalues, eigenvectors


def dDelta_dt_from_Delta(
    Delta: np.ndarray,
    Delta_E: np.ndarray | Callable,
    S_k: float,
    delta_hat: float,
    gamma: float, 
    kappa: float,
    T_k: np.ndarray,
    T_1: np.ndarray,
    F_k: Callable[[Callable | np.ndarray, complex, int], complex] | np.ndarray,
    R: Callable[[complex], complex] | np.ndarray,
    zg: np.ndarray,
    k: int, 
    num_eigenvalues: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Delta_E = np.vectorize(Delta_E)(zg) if callable(Delta_E) else Delta_E
    eta, A_ns, eigenvalues, eigenvectors = compute_eta_decomp(Delta, T_k, T_1, S_k, delta_hat, F_k, R, zg, k, num_eigenvalues = num_eigenvalues)
    return dDelta_dt(Delta, eta, Delta_E, gamma, kappa), eigenvalues, eigenvectors

def compute_dDelta_dt_from_range_Delta(
    Deltas: np.ndarray,
    Delta_E: np.ndarray | Callable,
    S_hat: float,
    delta_hat: float,
    gamma: float, 
    kappa: float,
    T_k: np.ndarray,
    T_1: np.ndarray,
    F_k: Callable[[Callable | np.ndarray, complex, int], complex] | np.ndarray,
    R: Callable[[complex], complex] | np.ndarray,
    zg: np.ndarray,
    k: int, 
    num_eigenvalues: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Delta_E = np.vectorize(Delta_E)(zg) if callable(Delta_E) else Delta_E

    dDelta_all = []
    eigenvalues_all = []
    eigenvectors_all = []
    for Delta in Deltas: 
        result = dDelta_dt_from_Delta(Delta, Delta_E, S_hat, delta_hat, gamma, kappa, T_k, T_1, F_k, R, zg, k, num_eigenvalues = num_eigenvalues)
        dDelta_all.append(result[0])
        eigenvalues_all.append(result[1])
        eigenvectors_all.append(result[2])

    dDelta_all = np.array(dDelta_all)
    eigenvalues_all = np.array(eigenvalues_all)
    eigenvectors_all = np.array(eigenvectors_all)

    return dDelta_all, eigenvalues_all, eigenvectors_all

def generate_contours(dDeltas, Deltas, zg, x_axis_level):
    Z = np.nan_to_num(dDeltas, nan=0.0)
    # Define symmetric logarithmic contour levels, while handling zero and near-zero values
    num_levels = 50

    # Ensure no zero values for log scale by setting minimum threshold
    positive_min = Z[Z > 0].min() if np.any(Z > 0) else 1e-3
    negative_min = abs(Z[Z < 0].max()) if np.any(Z < 0) else 1e-3

    # Logarithmic levels for positive and negative values
    positive_levels = np.geomspace(positive_min, Z.max(), num_levels) if Z.max() > 0 else []
    negative_levels = -np.geomspace(negative_min, -Z.min(), num_levels) if Z.min() < 0 else []
    levels = np.concatenate((negative_levels[::-1], [0], positive_levels))

    min_spacing = 0.01
    # Filter levels to enforce minimum spacing
    filtered_levels = [levels[0]]
    for level in levels[1:]:
        if abs(level - filtered_levels[-1]) >= min_spacing:
            filtered_levels.append(level)

    contour = plt.contour(Deltas[:, x_axis_level], zg, dDeltas.T, levels=filtered_levels, cmap='Grays')
    zero_contour = plt.contour(Deltas[:, x_axis_level], zg, dDeltas.T, levels=[0], colors='red', linewidths=2, linestyles='--')
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")



