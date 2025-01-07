from msilib import datasizemask
from typing import Callable, Tuple
import numpy as np
from scipy.optimize import fsolve, root, approx_fprime, minimize, Bounds
from scipy.linalg import norm, solve, lstsq, null_space, eigvals, svd
import const
from dynamical_system import full_system, dDelta_dt, full_system_real
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


#-------------------------------------------------------
#Older Parameter Continuation in Fully Glory
#-------------------------------------------------------

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

# --------------------------------------------------------
# Obselete Adomian Decomposition
# --------------------------------------------------------

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

# --------------------------------------------------------
# Eigenvalue Decomposition
# --------------------------------------------------------

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

def normalize(f, R, Delta, zg):
    inner_prod = inner_product(f, f, R, Delta, zg)
    norm = np.sqrt(np.abs(inner_prod))
    if norm == 0:
        return f
    else: 
        return (f / norm) * np.sign(inner_prod)

    

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
    # Compute eigenvalues and eige 
    dz = (zg[1] - zg[0])
    L_matrix = T_k * Delta * dz
    omega_c_matrix = np.diag(T1 @ Delta * dz)
    eigenvalues, eigenvectors = np.linalg.eig(L_matrix - omega_c_matrix)

    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = normalize(eigenvectors[:, i], R, Delta, zg) # Normalize the eigenvector

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

def generate_contours(dDeltas, Deltas, zg, x_axis_level, min_spacing = 0.01, num_levels = 50):
    Z = np.nan_to_num(dDeltas, nan=0.0)
    # Define symmetric logarithmic contour levels, while handling zero and near-zero values

    # Ensure no zero values for log scale by setting minimum threshold
    positive_min = Z[Z > 0].min() if np.any(Z > 0) else 1e-3
    negative_min = abs(Z[Z < 0].max()) if np.any(Z < 0) else 1e-3

    # Logarithmic levels for positive and negative values
    positive_levels = np.geomspace(positive_min, Z.max(), num_levels) if Z.max() > 0 else []
    negative_levels = -np.geomspace(negative_min, -Z.min(), num_levels) if Z.min() < 0 else []
    levels = np.concatenate((negative_levels[::-1], [0], positive_levels))
    
    # Filter levels to enforce minimum spacing
    filtered_levels = [levels[0]]
    for level in levels[1:]:
        if abs(level - filtered_levels[-1]) >= min_spacing:
            filtered_levels.append(level)

    contour = plt.contour(Deltas[:, x_axis_level], zg, dDeltas.T, levels=filtered_levels, cmap='Grays')
    zero_contour = plt.contour(Deltas[:, x_axis_level], zg, dDeltas.T, levels=[0], colors='red', linewidths=2, linestyles='--')
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

# --------------------------------------------------------
# New Parameter Continuation and Searching 
# --------------------------------------------------------

def build_LDelta(Delta, T_k, T_1):
    """
    Build the matrix representation of L^{\Delta} acting on eta:
    (L^{\Delta} eta)_i = sum_j [Delta_j * T_k[i,j] * eta_j] 
                         - (sum_j Delta_j*T_1[i,j]) * eta_i.
    
    Parameters:
    -----------
    Delta : array of shape (N,)
        Discretized Delta field.
    T_k, T_1 : arrays of shape (N,N)
        Discretized integral operators.
        
    Returns:
    --------
    L : array of shape (N,N)
        The matrix corresponding to L^{\Delta}.
    """
    L_matrix = T_k * Delta
    omega_c_matrix = np.diag(T_1 @ Delta)
    return L_matrix - omega_c_matrix

def solve_eta_for_given_delta(Delta, params, T_k, T_1, F_k):
    """
    Solve for eta given Delta at steady state.
    
    Steady-state condition for eta:
    0 = i*hat_Sk * L^Delta(eta) + (i*hat_Sk*hat_delta - 1)*eta - i*F_k/max(F_k).
    
    Rearranging:
    (i*hat_Sk*L^Delta + (i*hat_Sk*hat_delta - 1)*I)*eta = i*F_k/max(F_k).
    
    Parameters:
    -----------
    Delta : array (N,)
        Current guess (or known) Delta field.
    params : dict
        Contains hat_Sk, hat_delta, etc.
    T_k, T_1 : (N,N) arrays
        Discretized integral operators.
    F_k : (N,) array
        Discretized forcing.
        
    Returns:
    --------
    eta : (N,) array
        Solved eta at steady state corresponding to given Delta.
    """
    iS = 1j * params['hat_Sk']
    
    L = build_LDelta(Delta, T_k, T_1)
    
    A = iS * L + np.eye(L.shape[0]) * (iS * params['hat_delta'] - 1.0)
    rhs = 1j * F_k / max(F_k)
    
    eta = np.linalg.solve(A, rhs)
    return eta


def Delta_residual(Delta, params, T_k, T_1, F_k, Delta_E):
    """
    Compute the residual for Delta at steady state:
    R(Delta) = (Delta_E/max(Delta_E)) - Delta - hat_kappa*Delta*(|eta|^2).
    
    Parameters:
    -----------
    Delta : (N,)
    params : dict
    T_k, T_1 : (N,N)
    F_k : (N,)
    Delta_E : (N,)
    
    Returns:
    --------
    R : (N,)
        Residual vector for Delta.
    """
    eta = solve_eta_for_given_delta(Delta, params, T_k, T_1, F_k)
    R = dDelta_dt(Delta, eta, Delta_E, params['gamma'], params['kappa'])
    return R


def find_steady_state(Delta_init, params, T_k, T_1, F_k, Delta_E, tol=1e-8, maxiter=100):
    """
    Find a steady-state Delta by solving R(Delta)=0 using a simple Newton-like iteration.
    
    Parameters:
    -----------
    Delta_init : (N,)
        Initial guess for Delta.
    params : dict
    T_k, T_1 : (N,N)
    F_k, Delta_E : (N,)
    dz : float
        Discretization step
    tol : float
        Tolerance for convergence.
    maxiter : int
        Maximum iterations.
    
    Returns:
    --------
    Delta : (N,)
        Steady-state solution for Delta.
    success : bool
        Whether convergence was achieved.
    """
    Delta = Delta_init.copy()
    for it in range(maxiter):
        R = Delta_residual(Delta, params, T_k, T_1, F_k, Delta_E)
        resnorm = np.linalg.norm(R)
        if resnorm < tol:
            return Delta, True
        Delta = Delta + R  # Simple relaxation step
    return Delta, False

def parameter_continuation(Delta_init, params, vary_param, param_values, T_k, T_1, F_k, Delta_E, tol=1e-8):
    """
    Simple parameter continuation method. Given a list of param_values for vary_param,
    we solve for steady states at each param value, using the previous solution as initial guess.
    
    Parameters:
    -----------
    Delta_init : (N,)
        Initial guess for Delta at the first param value.
    params : dict
        Parameter dictionary. Will be modified as we step through param_values.
    vary_param : str
        The key in params to vary.
    param_values : array-like
        Values of the parameter to iterate over.
    T_k, T_1 : (N,N)
    F_k, Delta_E : (N,)
    tol : float
    
    Returns:
    --------
    results : list of tuples (param_val, Delta, success)
        Steady-state solutions for each parameter value.
    """
    results = []
    Delta_current = Delta_init.copy()
    for pval in param_values:
        params[vary_param] = pval
        Delta_sol, success = find_steady_state(Delta_current, params, T_k, T_1, F_k, Delta_E, tol=tol)
        results.append((pval, Delta_sol, success))
        if success:
            Delta_current = Delta_sol
        else:
            break
    return results

def generate_4d_matrix(S_hat_vals, delta_hat_vals, kappa_vals, Delta_init, params, T_k, T_1, F_k, Delta_E, tol=1e-8):
    """
    Generate a 4D matrix of parameter space with S_hat, delta_hat, and kappa.
    Each entry corresponds to the resulting Delta (or NaNs if no convergence).

    Parameters:
    -----------
    S_hat_vals : array-like
        Values of S_hat to explore.
    delta_hat_vals : array-like
        Values of delta_hat to explore.
    kappa_vals : array-like
        Values of kappa to explore.
    Delta_init : (N,)
        Initial guess for Delta.
    params : dict
        Parameter dictionary. Will be modified for each parameter set.
    T_k, T_1 : (N,N)
    F_k, Delta_E : (N,)
    tol : float

    Returns:
    --------
    result_matrix : 4D numpy array
        The resulting 4D matrix where each entry contains the steady-state Delta
        or NaNs if no steady state was found.
    """
    S_len, delta_len, kappa_len = len(S_hat_vals), len(delta_hat_vals), len(kappa_vals)
    result_matrix = np.full((S_len, delta_len, kappa_len, len(Delta_init)), np.nan)

    for i, S_hat in enumerate(S_hat_vals):
        for j, delta_hat in enumerate(delta_hat_vals):
            for k, kappa in enumerate(kappa_vals):
                params['hat_Sk'] = S_hat
                params['hat_delta'] = delta_hat
                params['hat_kappa'] = kappa

                Delta_sol, success = find_steady_state(Delta_init, params, T_k, T_1, F_k, Delta_E, tol=tol)
                if success:
                    result_matrix[i, j, k, :] = Delta_sol

    return result_matrix

def plot_parameter_slices(S_hat_vals, delta_hat_vals, kappa_vals, result_matrix):
    """
    Plot 2D parameter slices for S_hat vs delta_hat, S_hat vs kappa, and kappa vs S_hat.

    Parameters:
    -----------
    S_hat_vals : array-like
    delta_hat_vals : array-like
    kappa_vals : array-like
    result_matrix : 4D numpy array
        The resulting 4D matrix where each entry contains the steady-state Delta
        or NaNs if no steady state was found.
    """
    import matplotlib.pyplot as plt

    # Aggregate result_matrix to a scalar field for each parameter pair
    # (e.g., use the norm of Delta as the scalar value)

    # S_hat vs delta_hat (fix kappa index)
    kappa_idx = len(kappa_vals) // 2
    slice_S_delta = np.linalg.norm(result_matrix[:, :, kappa_idx, :], axis=-1)

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_S_delta, extent=[delta_hat_vals[0], delta_hat_vals[-1], S_hat_vals[0], S_hat_vals[-1]], 
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='||Delta||')
    plt.xlabel('delta_hat')
    plt.ylabel('S_hat')
    plt.title('S_hat vs delta_hat (fixed kappa)')
    plt.show()

    # S_hat vs kappa (fix delta_hat index)
    delta_idx = len(delta_hat_vals) // 2
    slice_S_kappa = np.linalg.norm(result_matrix[:, delta_idx, :, :], axis=-1)

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_S_kappa, extent=[kappa_vals[0], kappa_vals[-1], S_hat_vals[0], S_hat_vals[-1]], 
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='||Delta||')
    plt.xlabel('kappa')
    plt.ylabel('S_hat')
    plt.title('S_hat vs kappa (fixed delta_hat)')
    plt.show()

    # kappa vs delta_hat (fix S_hat index)
    S_hat_idx = len(S_hat_vals) // 2
    slice_kappa_delta = np.linalg.norm(result_matrix[S_hat_idx, :, :, :], axis=-1)

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_kappa_delta, extent=[delta_hat_vals[0], delta_hat_vals[-1], kappa_vals[0], kappa_vals[-1]], 
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='||Delta||')
    plt.xlabel('delta_hat')
    plt.ylabel('kappa')
    plt.title('kappa vs delta_hat (fixed S_hat)')
    plt.show()

def compute_jacobian(Delta, params, T_k, T_1, F_k, Delta_E):
    """
    Approximate the Jacobian numerically.
    """
    epsilon = 1e-6
    N = len(Delta)
    J = np.zeros((N, N))
    for i in range(N):
        Delta_perturbed = Delta.copy()
        Delta_perturbed[i] += epsilon
        R_perturbed = Delta_residual(Delta_perturbed, params, T_k, T_1, F_k, Delta_E)
        R = Delta_residual(Delta, params, T_k, T_1, F_k, Delta_E)
        J[:, i] = (R_perturbed - R) / epsilon
    return J

def parameter_continuation_with_bifurcations(Delta_init, params, vary_param, param_range, T_k, T_1, F_k, Delta_E, tol=1e-8, maxiter=100):
    """
    Perform parameter continuation to find bifurcations along a parameter path.

    Parameters:
    -----------
    Delta_init : (N,)
        Initial guess for Delta.
    params : dict
        Parameter dictionary.
    vary_param : str
        The parameter key to vary.
    param_range : array-like
        Range of parameter values to iterate over.
    T_k, T_1 : (N,N)
        Discretized integral operators.
    F_k, Delta_E : (N,)
        Discretized forcing and equilibrium values.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations for each steady state solve.

    Returns:
    --------
    bifurcation_data : list of tuples (param_value, steady_state, bifurcation_detected)
        List of parameter values, corresponding steady states, and bifurcation flags.
    """
    bifurcation_data = []
    Delta_current = Delta_init.copy()

    for param_value in param_range:
        params[vary_param] = param_value

        # Solve for steady state
        Delta_sol, success = find_steady_state(Delta_current, params, T_k, T_1, F_k, Delta_E, tol=tol, maxiter=maxiter)
        if not success:
            bifurcation_data.append((param_value, None, False))
            continue

        # Compute the Jacobian at the steady state
        J = compute_jacobian(Delta_sol, params, T_k, T_1, F_k, Delta_E)
        eigenvalues = np.linalg.eigvals(J)

        # Check for bifurcation: any eigenvalue with zero real part
        bifurcation_detected = any(np.abs(e.real) < tol and e.imag != 0 for e in eigenvalues)
        bifurcation_data.append((param_value, Delta_sol, bifurcation_detected))

        # Update current solution for next step
        Delta_current = Delta_sol

    return bifurcation_data


def plot_continuation_results(param_range, bifurcation_data, vary_param):
    """
    Plot parameter continuation results with bifurcation points, indicating points with no convergence
    and bifurcation lines.

    Parameters:
    -----------
    param_range : array-like
        Range of parameter values.
    bifurcation_data : list of tuples (param_value, steady_state, bifurcation_detected)
        Output of parameter_continuation_with_bifurcations.
    vary_param : str
        Name of the parameter being varied.
    """
    steady_states_mean = [np.mean(data[1]) if data[1] is not None else np.nan for data in bifurcation_data]

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, steady_states_mean, label='Steady State Norm')

   # Highlight bifurcation points with vertical black lines
    for i, (param, _, bifurcation) in enumerate(bifurcation_data):
        if bifurcation:
            plt.axvline(x=param, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
            plt.scatter(param, steady_states_mean[i], color='red', label='Bifurcation' if i == 0 else "")

    # Identify contiguous regions with no convergence
    no_convergence_regions = []
    start = None
    for i, (param, state, _) in enumerate(bifurcation_data):
        if state is None and start is None:
            start = param_range[i-1] if i > 0 else param_range[0]  # Start from the previous parameter
        elif state is not None and start is not None:
            no_convergence_regions.append((start, param_range[i]))
            start = None
    if start is not None:
        no_convergence_regions.append((start, param_range[-1]))

    # Shade contiguous regions with no convergence in smooth light red
    for (start, end) in no_convergence_regions:
        plt.axvspan(start, end, color='red', alpha=0.2, label='No Solution' if start == no_convergence_regions[0][0] else "")

    plt.xlabel(vary_param)
    plt.ylabel('Steady State Norm')
    plt.title(f'Parameter Continuation with Bifurcation Detection ({vary_param})')
    plt.legend()
    plt.grid()
    plt.show()


def parameter_continuation_with_bifurcations_2d(Delta_init, params, vary_param1, param_values1, vary_param2, param_values2, T_k, T_1, F_k, Delta_E, tol=1e-8, maxiter=100):
    """
    Two-parameter continuation method with bifurcation detection. Solves for steady states
    over a grid of two parameters, identifying bifurcations and using solutions from neighboring grid points
    as initial guesses to improve efficiency.

    Parameters:
    -----------
    Delta_init : (N,)
        Initial guess for Delta at the first parameter pair.
    params : dict
        Parameter dictionary. Will be modified as we step through param_values.
    vary_param1 : str
        The first parameter key to vary.
    param_values1 : array-like
        Values of the first parameter to iterate over.
    vary_param2 : str
        The second parameter key to vary.
    param_values2 : array-like
        Values of the second parameter to iterate over.
    T_k, T_1 : (N,N)
        Discretized integral operators.
    F_k, Delta_E : (N,)
        Discretized forcing and equilibrium values.
    tol : float
        Tolerance for convergence.
    maxiter : int
        Maximum number of iterations for each steady state solve.

    Returns:
    --------
    results : dict
        A dictionary where keys are (param1, param2) tuples and values are tuples of
        (Delta_solution, bifurcation_flag, success_flag).
    """
    results = {}
    Delta_current = Delta_init.copy() #TODO: Investigate Delta_current to see if there is a better way of doing continuation here. 

    for i, pval1 in enumerate(param_values1):
        for j, pval2 in enumerate(param_values2):
            # Update parameters
            params[vary_param1] = pval1
            params[vary_param2] = pval2

            # Determine a safe initial guess
            if (i > 0 and results.get((param_values1[i-1], pval2), (None, None, False))[2]):
                prev_solution = results[(param_values1[i-1], pval2)][0]
            elif (j > 0 and results.get((pval1, param_values2[j-1]), (None, None, False))[2]):
                prev_solution = results[(pval1, param_values2[j-1])][0]
            else:
                prev_solution = Delta_init.copy()

            # Solve for steady state
            Delta_sol, success = find_steady_state(prev_solution, params, T_k, T_1, F_k, Delta_E, tol=tol, maxiter=maxiter)

            if not success:
                results[(pval1, pval2)] = (None, False, False)
                continue

            # Compute Jacobian and check for bifurcation
            J = compute_jacobian(Delta_sol, params, T_k, T_1, F_k, Delta_E)
            eigenvalues = np.linalg.eigvals(J)
            bifurcation_detected = any(np.abs(e.real) < tol and e.imag != 0 for e in eigenvalues)

            results[(pval1, pval2)] = (Delta_sol, bifurcation_detected, success)

            # Update current Delta
            if success:
                Delta_current = Delta_sol

    return results

def plot_continuation_results_2d(param_values1, param_values2, results, vary_param1, vary_param2):
    """
    Plot a contour plot of the steady-state solutions in 2D parameter space.
    Highlights bifurcation points and uses a grey background for regions with no convergence.

    Parameters:
    -----------
    param_values1 : array-like
        Values of the first parameter.
    param_values2 : array-like
        Values of the second parameter.
    results : dict
        Output of parameter_continuation_with_bifurcations_2d. Keys are (param1, param2) tuples and
        values are (Delta_solution, bifurcation_flag, success_flag).
    vary_param1 : str
        Name of the first parameter.
    vary_param2 : str
        Name of the second parameter.
    """
    steady_state_norms = np.full((len(param_values1), len(param_values2)), np.nan)
    bifurcation_map = np.zeros((len(param_values1), len(param_values2)), dtype=bool)
    no_convergence_map = np.zeros((len(param_values1), len(param_values2)), dtype=bool)

    # Populate the steady state norms, bifurcation map, and no convergence map
    for i, pval1 in enumerate(param_values1):
        for j, pval2 in enumerate(param_values2):
            result = results.get((pval1, pval2), (None, False, False))
            Delta_sol, bifurcation_flag, success_flag = result
            if success_flag and Delta_sol is not None:
                steady_state_norms[i, j] = np.mean(Delta_sol)
            elif not success_flag:
                no_convergence_map[i, j] = True
            if bifurcation_flag:
                bifurcation_map[i, j] = True

    # Mask non-convergent regions for contour plot
    steady_state_norms_masked = np.ma.masked_where(no_convergence_map, steady_state_norms)

    # Create a grey background
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor((1, 0, 0, 0.3))  # RGBA: light grey with 30% opacity

    # Create a contour plot of the steady-state norm
    X, Y = np.meshgrid(param_values2, param_values1)
    norm_plot = ax.contourf(X, Y, steady_state_norms, levels=20, cmap='viridis', extend='both')
    cbar = plt.colorbar(norm_plot, label='Steady-State Norm')

    # Overlay bifurcation points
    bifurcation_indices = np.where(bifurcation_map)
    plt.scatter(param_values2[bifurcation_indices[1]], param_values1[bifurcation_indices[0]],
                color='black', label='Bifurcation Points', zorder=10)
    plt.scatter([], [], color=(1, 0, 0, 0.3), label='No Convergence')
    plt.xlabel(vary_param2)
    plt.ylabel(vary_param1)
    plt.title(f'Parameter Continuation in 2D: {vary_param1} vs {vary_param2}')
    plt.legend()
    plt.show()


    # ---------------------------------------------
    # New Paramter Continuation 
    # ---------------------------------------------


def compute_jacobian_full_real(vars: np.ndarray, params: dict) -> np.ndarray:
    """
    Numerically computes the Jacobian matrix of the steady-state system.
    
    Args:
        vars: Array containing [Delta, eta.real, eta.imag].
        params: Dictionary of parameters.
        
    Returns:
        Jacobian matrix.
    """
    epsilon = 1e-6
    N = vars.shape[0]
    J = np.zeros((N, N), dtype=complex)
    
    f0 = full_system_real(vars, params) 
    for i in range(N):
        vars_eps = vars.copy()
        vars_eps[i] += epsilon
        f_eps = full_system_real(vars_eps, params) 
        J[:, i] = (f_eps - f0) / epsilon
    return J

def parameter_continuation_with_bifurcation_full(
    initial_vars: np.ndarray,
    params: dict,
    param_name: str,
    param_range: Tuple[float, float],
    step_size: float = 0.1,
    max_steps: int = 1000,
    tolerance: float = 1e-6,
    bifurcation_threshold: float = 1e-3
):
    """
    Continuation with bifurcation detection and tracing parameter curves.

    Args:
        initial_vars: Initial guess for the steady-state variables.
        params: Dictionary of system parameters.
        param_name: Name of the parameter to vary.
        param_range: Tuple of (start, end) values for the parameter.
        step_size: Step size for the parameter continuation.
        max_steps: Maximum number of continuation steps.
        tolerance: Convergence tolerance for `fsolve`.
        bifurcation_threshold: Threshold for detecting bifurcations based on eigenvalues.

    Returns:
        bifurcation_map: List of tuples (parameter, solution, is_bifurcation).
    """
    bifurcation_map = []

    current_vars = initial_vars.copy()
    current_param = param_range[0]
    direction = np.sign(param_range[1] - param_range[0])  # Determine direction of parameter continuation

    for step in range(max_steps):
        # Compute Jacobian and eigenvalues for bifurcation detection
        try:
            J = compute_jacobian_full_real(current_vars, params)
            eigenvals = eigvals(J)
            real_parts = np.real(eigenvals)
            is_bifurcation = np.any(np.abs(real_parts) < bifurcation_threshold)
        except Exception as e:
            print(f"Step {step}: Jacobian computation failed: {e}")
            is_bifurcation = False

        # Update parameter
        new_param = current_param + direction * step_size
        if (direction > 0 and new_param > param_range[1]) or (direction < 0 and new_param < param_range[1]):
            break
        params[param_name] = new_param

        # Predictor step: Use the last valid solution as the initial guess
        initial_guess = current_vars.copy()

        # Corrector step: Solve the steady-state equations with the new parameter
        try:
            sol, info, ier, mesg = fsolve(
                full_system_real,  # System of equations for steady state
                initial_guess,
                args=(params,),
                xtol=tolerance,
                full_output=True
            )
            if ier != 1:
                print(f"Step {step}: fsolve did not converge. Message: {mesg}")
                bifurcation_map.append((current_param, None, is_bifurcation))
            else:
                current_vars = sol  # Update the current solution
                bifurcation_map.append((current_param, sol, is_bifurcation))
        except Exception as e:
            print(f"Step {step}: fsolve failed: {e}")
            bifurcation_map.append((current_param, None, is_bifurcation))

        # Update the current parameter value
        current_param = new_param

    return bifurcation_map


def parameter_continuation_with_bifurcation_full_2d_flat(
    initial_vars: np.ndarray,
    params: dict,
    param_name1: str,
    param_range1: Tuple[float, float],
    step_size1: float,
    param_name2: str,
    param_range2: Tuple[float, float],
    step_size2: float,
    max_steps: int = 100,
    tolerance: float = 1e-8,
    bifurcation_threshold: float = 1e-3
):
    """
    Two-dimensional continuation with bifurcation detection and tracing parameter curves.

    Args:
        initial_vars: Initial guess for the steady-state variables.
        params: Dictionary of system parameters.
        param_name1: Name of the first parameter to vary.
        param_range1: Tuple of (start, end) values for the first parameter.
        param_name2: Name of the second parameter to vary.
        param_range2: Tuple of (start, end) values for the second parameter.
        step_size1: Step size for the first parameter continuation.
        step_size2: Step size for the second parameter continuation.
        max_steps: Maximum number of continuation steps for each parameter.
        tolerance: Convergence tolerance for `fsolve`.
        bifurcation_threshold: Threshold for detecting bifurcations based on eigenvalues.

    Returns:
        results: List of tuples (param1, param2, solution, is_bifurcation).
    """
    results = []

    current_param1 = param_range1[0]
    last_convergent_solution = initial_vars.copy()

    while current_param1 <= param_range1[1]:
        current_param2 = param_range2[0]
        while current_param2 <= param_range2[1]:
            # Update parameters
            params[param_name1] = current_param1
            params[param_name2] = current_param2

            # Determine a safe initial guess
            if results and results[-1][2] is not None:
                initial_guess = results[-1][2]
            else:
                initial_guess = initial_vars.copy()

            # Solve the system
            sol, info, ier, mesg = fsolve(
                full_system_real,  # System of equations for steady state
                initial_guess,
                args=(params,),
                xtol=tolerance,
                full_output=True
            )
            if ier != 1:
                print(f"Continuation failed at ({current_param1}, {current_param2}): {mesg}")
                results.append((current_param1, current_param2, None, False))  # No steady state found
            else:
                # Compute bifurcation detection
                try:
                    J = compute_jacobian_full_real(sol, params)
                    eigenvals = eigvals(J)
                    real_parts = np.real(eigenvals)
                    is_bifurcation = np.any(np.abs(real_parts) < bifurcation_threshold)
                except Exception as e:
                    print(f"Jacobian computation failed at ({current_param1}, {current_param2}): {e}")
                    is_bifurcation = False

                last_convergent_solution = sol
                results.append((current_param1, current_param2, sol, is_bifurcation))

            # Dynamically adjust parameter 2
            current_param2 += step_size2

        # Dynamically adjust parameter 1
        current_param1 += step_size1

    return results


def parameter_continuation_with_bifurcation_full_2d(
    initial_vars: np.ndarray,
    params: dict,
    param_name1: str,
    param_range1: Tuple[float, float],
    step_size1 : float,
    param_name2: str,
    param_range2: Tuple[float, float],
    step_size2: float,
    tolerance: float = 1e-8,
    bifurcation_threshold: float = 1e-3
):
    """
    Two-dimensional continuation with bifurcation detection and tracing parameter curves.

    Args:
        initial_vars: Initial guess for the steady-state variables.
        params: Dictionary of system parameters.
        param_name1: Name of the first parameter to vary.
        param_range1: Tuple of (start, end) values for the first parameter.
        param_name2: Name of the second parameter to vary.
        param_range2: Tuple of (start, end) values for the second parameter.
        step_size1: Step size for the first parameter continuation.
        step_size2: Step size for the second parameter continuation.
        max_steps: Maximum number of continuation steps for each parameter.
        tolerance: Convergence tolerance for `fsolve`.
        bifurcation_threshold: Threshold for detecting bifurcations based on eigenvalues.

    Returns:
        results: Dictionary with keys as (param1, param2) and values as (solution, is_bifurcation).
    """
    results = {}
    last_convergent_solution = initial_vars.copy()

    current_param1 = param_range1[0]
    while current_param1 <= param_range1[1]:
        current_param2 = param_range2[0]
        while current_param2 <= param_range2[1]:
            # Update parameters
            params[param_name1] = current_param1
            params[param_name2] = current_param2

            # Determine a safe initial guess
            if results:
                prev_key1 = (current_param1 - step_size1, current_param2)
                prev_key2 = (current_param1, current_param2 - step_size2)
                if prev_key1 in results and results[prev_key1][0] is not None:
                    initial_guess = results[prev_key1][0]
                    last_convergent_solution = results[prev_key1][0]
                elif prev_key2 in results and results[prev_key2][0] is not None:
                    initial_guess = results[prev_key2][0]
                    last_convergent_solution = results[prev_key2][0]
                else:
                    initial_guess = last_convergent_solution.copy()
            else:
                initial_guess = last_convergent_solution.copy()

            # Solve the system
            sol, info, ier, mesg = fsolve(
                full_system_real,  # System of equations for steady state
                initial_guess,
                args=(params,),
                xtol=tolerance,
                full_output=True
            )
            if ier != 1:
                print(f"Continuation failed at ({current_param1}, {current_param2}): {mesg}")
                results[(current_param1, current_param2)] = (None, False, False)  # No steady state found
            else:
                # Update last convergent solution
                last_convergent_solution = sol.copy()

                # Compute bifurcation detection
                try:
                    J = compute_jacobian_full_real(sol, params)
                    eigenvals = eigvals(J)
                    real_parts = np.real(eigenvals)
                    is_bifurcation = np.any(np.abs(real_parts) < bifurcation_threshold)
                except Exception as e:
                    print(f"Jacobian computation failed at ({current_param1}, {current_param2}): {e}")
                    is_bifurcation = False

                results[(current_param1, current_param2)] = (sol, is_bifurcation, True)

            # Dynamically adjust parameter 2
            current_param2 += step_size2

        # Dynamically adjust parameter 1
        current_param1 += step_size1

    return results



# ----------------------------------------
# Eigenvector Basis Parameter Continuation
# ----------------------------------------

def compute_jacobian_full_imag(vars: np.ndarray, params: dict) -> np.ndarray:
    """
    Numerically computes the Jacobian matrix of the steady-state system.
    
    Args:
        vars: Array containing [Delta, eta.real, eta.imag].
        params: Dictionary of parameters.
        
    Returns:
        Jacobian matrix.
    """
    epsilon = 1e-6
    N = vars.shape[0]
    J = np.zeros((N, N), dtype=complex)
    
    f0 = full_system(vars, None, None, 1, params) 
    for i in range(N):
        vars_eps = vars.copy()
        vars_eps[i] += epsilon
        f_eps = full_system(vars_eps, None, None, 1, params) 
        J[:, i] = (f_eps - f0) / epsilon
    return J


# Assume eigenfunctions are provided as lists or arrays
# eigenfuncs_Delta and eigenfuncs_eta should be lists of functions or discrete arrays evaluated at z points
# For example:
# eigenfuncs_Delta = [phi_1(z), phi_2(z), ..., phi_N(z)]
# eigenfuncs_eta = [psi_1(z), psi_2(z), ..., psi_N(z)]
# Ensure that all eigenfunctions are normalized with respect to the inner product

def project_onto_eigenfunctions(func: np.ndarray, eigenfuncs: list[np.ndarray], R, zg) -> np.ndarray:
    """
    Project a function onto a set of eigenfunctions using the inner product.

    Args:
        func: The function to project, as a discrete array over z.
        eigenfuncs: A list of eigenfunctions, each as a discrete array over z.
        inner_product: A function that takes two arrays and returns their inner product.

    Returns:
        coefficients: Array of coefficients A_n or B_n.

    """
    coefficients = np.array([inner_product(func, phi, R, np.ones_like(zg), zg) for phi in eigenfuncs])
    return coefficients.real, coefficients.imag

def reconstruct_from_coefficients(coefficients: np.ndarray, eigenfuncs: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct a function from its coefficients and eigenfunctions.

    Args:
        coefficients: Array of coefficients A_n or B_n.
        eigenfuncs: A list of eigenfunctions, each as a discrete array over z.

    Returns:
        func: The reconstructed function as a discrete array over z.
    """
    func = np.zeros_like(eigenfuncs[0], dtype=np.complex128)
    for i in range(len(eigenfuncs)):
        func += coefficients[i] * eigenfuncs[:, i]
    return func

def full_system_coefficients(vars_coeffs: np.ndarray, eigenfuncs: list[np.ndarray], params: dict, R, zg) -> np.ndarray:
    """
    System of equations in coefficient space. Reconstructs Delta and eta from coefficients,
    evaluates the full system, and projects back onto the eigenfunctions.

    Args:
        vars_coeffs: Array containing [A_1, A_2, ..., A_N, B_1, B_2, ..., B_N].
        params: Dictionary of system parameters.
        eigenfuncs_Delta: List of eigenfunctions for Delta.
        eigenfuncs_eta: List of eigenfunctions for eta.
        inner_product: Function to compute inner product.

    Returns:
        residuals: Array of residuals for each coefficient equation.
    """
    N = len(eigenfuncs)
    A_coeffs, B_coeffs_real, B_coeffs_imag = np.split(vars_coeffs, 3)

    # Reconstruct Delta and eta in z-space
    Delta = reconstruct_from_coefficients(A_coeffs, eigenfuncs)
    eta = reconstruct_from_coefficients(B_coeffs_real + 1j * B_coeffs_imag, eigenfuncs)

    # Evaluate the full system in z-space
    # Assuming full_system_real accepts Delta and eta as separate arguments
    # and returns a concatenated array of residuals for Delta and eta
    residuals_z = full_system_real(np.concatenate([Delta, eta.real, eta.imag]), params)

    # Project residuals back onto the eigenfunctions to obtain residuals in coefficient space
    residuals_A, _  = project_onto_eigenfunctions(residuals_z[:N], eigenfuncs, R, zg)
    residuals_B_real, residuals_B_imag = project_onto_eigenfunctions(residuals_z[N:2*N] + 1j * residuals_z[2*N:], eigenfuncs, R, zg)

    residuals = np.concatenate([residuals_A, residuals_B_real, residuals_B_imag])

    return residuals


def parameter_continuation_with_bifurcation_coefficients_space(
    initial_vars_z: np.ndarray,
    params: dict,
    param_name1: str,
    param_range1: Tuple[float, float],
    step_size1: float,
    param_name2: str,
    param_range2: Tuple[float, float],
    step_size2: float,
    R, 
    zg,
    tolerance: float = 1e-8,
    bifurcation_threshold: float = 1e-3
):
    """
    Two-dimensional continuation in coefficient space with bifurcation detection.

    Args:
        initial_vars_z: Initial guess for [Delta(z), eta(z)] as a single concatenated array.
        params: Dictionary of system parameters.
        param_name1: Name of the first parameter to vary.
        param_range1: Tuple of (start, end) values for the first parameter.
        step_size1: Step size for the first parameter continuation.
        param_name2: Name of the second parameter to vary.
        param_range2: Tuple of (start, end) values for the second parameter.
        step_size2: Step size for the second parameter continuation.
        eigenfuncs_Delta: List of eigenfunctions for Delta.
        eigenfuncs_eta: List of eigenfunctions for eta.
        inner_product: Function to compute inner product.
        tolerance: Convergence tolerance for `fsolve`.
        bifurcation_threshold: Threshold for detecting bifurcations based on eigenvalues.

    Returns:
        results: Dictionary with keys as (param1, param2) and values as (solution_coeffs, is_bifurcation, converged).
    """
    results = {}

    #compute eigenfunctions
    N = len(zg)
    eigenvalues, eigenfuncs = np.linalg.eig(build_LDelta(np.ones(N), params['T_k'], params['T_1']))
    
    # Project initial Delta and eta onto eigenfunctions to get initial coefficients
    initial_Delta, initial_eta = np.split(initial_vars_z, 2)
    
    initial_A, _ = project_onto_eigenfunctions(initial_Delta, eigenfuncs, R, zg)
    initial_B_real, initial_B_imag = project_onto_eigenfunctions(initial_eta, eigenfuncs, R, zg)
    
    initial_coeffs = np.concatenate([initial_A, initial_B_real, initial_B_imag])
    last_convergent_solution = initial_coeffs.copy()

    current_param1 = param_range1[0]
    while current_param1 <= param_range1[1]:
        current_param2 = param_range2[0]
        while current_param2 <= param_range2[1]:
            # Update parameters
            params[param_name1] = current_param1
            params[param_name2] = current_param2

            # Determine a safe initial guess in coefficient space
            if results:
                prev_key1 = (current_param1 - step_size1, current_param2)
                prev_key2 = (current_param1, current_param2 - step_size2)
                if prev_key1 in results and results[prev_key1][2]:
                    initial_guess = results[prev_key1][0]
                    last_convergent_solution = results[prev_key1][0]
                elif prev_key2 in results and results[prev_key2][2]:
                    initial_guess = results[prev_key2][0]
                    last_convergent_solution = results[prev_key2][0]
                else:
                    initial_guess = last_convergent_solution.copy()
            else:
                initial_guess = last_convergent_solution.copy()

            # Solve the system in coefficient space
            sol, info, ier, mesg = fsolve(
                full_system_coefficients,
                initial_guess,
                args=(eigenfuncs, params, R, zg),
                xtol=tolerance,
                full_output=True
            )

            if ier != 1:
                print(f"Continuation failed at ({current_param1}, {current_param2}): {mesg}")
                results[(current_param1, current_param2)] = (None, False, False)  # No steady state found
            else:
                # Update last convergent solution
                last_convergent_solution = sol.copy()

                # Reconstruct Delta and eta from coefficients for Jacobian computation
                Delta_sol_coeffs, eta_sol_coeffs_real, eta_sol_coeffs_imag = np.split(sol, 3)
                Delta_sol = reconstruct_from_coefficients(Delta_sol_coeffs, eigenfuncs)
                eta_sol = reconstruct_from_coefficients(eta_sol_coeffs_real + 1j * eta_sol_coeffs_imag, eigenfuncs)
                sol_z = np.concatenate([Delta_sol, eta_sol.real, eta_sol.imag])


                # Compute bifurcation detection
                try:
                    J = compute_jacobian_full_real(sol_z, params)  # Assuming this computes Jacobian in z-space
                    eigenvals = eigvals(J)
                    real_parts = np.real(eigenvals)
                    is_bifurcation = np.any(np.abs(real_parts) < bifurcation_threshold)
                except Exception as e:
                    print(f"Jacobian computation failed at ({current_param1}, {current_param2}): {e}")
                    is_bifurcation = False

                results[(current_param1, current_param2)] = (sol, is_bifurcation, True)

            # Dynamically adjust parameter 2
            current_param2 += step_size2

        # Dynamically adjust parameter 1
        current_param1 += step_size1

    return results


def plot_continuation_results_2d_coefs(
    param_values1: np.ndarray,
    param_values2: np.ndarray,
    results: dict[Tuple[float, float], tuple[np.ndarray, bool, bool]],
    vary_param1: str,
    vary_param2: str,
    eigenfunc_indices: list[int] = None
):
    """
    Plot a contour plot of the steady-state solutions in 2D parameter space.
    Highlights bifurcation points and uses a grey background for regions with no convergence.
    Instead of plotting the mean of the entire solution, it plots the mean of specified eigenfunctions.

    Parameters:
    -----------
    param_values1 : array-like
        Values of the first parameter.
    param_values2 : array-like
        Values of the second parameter.
    results : dict
        Output of parameter_continuation_with_bifurcations_2d. Keys are (param1, param2) tuples and
        values are (Delta_solution, bifurcation_flag, success_flag).
    vary_param1 : str
        Name of the first parameter.
    vary_param2 : str
        Name of the second parameter.
    eigenfunc_indices : list of int, optional
        List of indices specifying which eigenfunctions to include in the mean calculation.
        If None, the mean of the entire Delta_solution is computed.
        Indices should be zero-based.
    """
    # Initialize arrays to store computed means and flags
    steady_state_means = np.full((len(param_values1), len(param_values2)), np.nan)
    bifurcation_map = np.zeros((len(param_values1), len(param_values2)), dtype=bool)
    no_convergence_map = np.zeros((len(param_values1), len(param_values2)), dtype=bool)

    # Iterate over all parameter combinations
    for i, pval1 in enumerate(param_values1):
        for j, pval2 in enumerate(param_values2):
            result = results.get((pval1, pval2), (None, False, False))
            Delta_sol, bifurcation_flag, success_flag = result

            if success_flag and Delta_sol is not None:
                if eigenfunc_indices is not None:
                    # Validate indices to prevent out-of-bounds access
                    max_index = len(Delta_sol) - 1
                    valid_indices = [idx for idx in eigenfunc_indices if 0 <= idx <= max_index]
                    if not valid_indices:
                        raise ValueError("No valid eigenfunction indices provided.")
                    
                    # Extract the specified coefficients
                    selected_coeffs = Delta_sol[valid_indices]
                    
                    # Compute the mean of the selected coefficients
                    mean_selected = np.mean(selected_coeffs)
                    steady_state_means[i, j] = mean_selected
                else:
                    # If no specific eigenfunctions are specified, compute the mean of the entire solution
                    steady_state_means[i, j] = np.mean(Delta_sol)
            elif not success_flag:
                no_convergence_map[i, j] = True

            if bifurcation_flag:
                bifurcation_map[i, j] = True

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 9))

        # Create a grey background
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor((1, 0, 0, 0.3))  # RGBA: light grey with 30% opacity

    # Create a contour plot of the steady-state norm
    X, Y = np.meshgrid(param_values2, param_values1)
    norm_plot = ax.contourf(X, Y, steady_state_means, levels=20, cmap='viridis', extend='both')
    cbar = plt.colorbar(norm_plot, label='Steady-State Norm')

    # Overlay bifurcation points
    bifurcation_indices = np.where(bifurcation_map)
    plt.scatter(param_values2[bifurcation_indices[1]], param_values1[bifurcation_indices[0]],
                color='black', label='Bifurcation Points', zorder=10)
    plt.scatter([], [], color=(1, 0, 0, 0.3), label='No Convergence')
    plt.xlabel(vary_param2)
    plt.ylabel(vary_param1)
    plt.title(f'Parameter Continuation in 2D: {vary_param1} vs {vary_param2}')
    plt.legend()
    plt.show()

# ------------------------------------------------------------------------------
# Eigenvalue Multiplicity 
# ------------------------------------------------------------------------------

def get_eigenvalues(params, shared_results):
    """
    Compute or retrieve eigenvalues for the given parameters.
    """
    key = params.tostring()  # Unique key for the parameter set
    if key not in shared_results:
        Delta = params[:-3]
        dz = shared_results['dz']
        L_matrix = shared_results['T_k'] * Delta * dz
        omega_c_matrix = np.diag(shared_results['T1'] @ Delta * dz)
        eigenvalues, eigenvectors = np.linalg.eig(L_matrix - omega_c_matrix)

        # Store results in shared_results
        shared_results[key] = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}

    return shared_results[key]['eigenvalues'], shared_results[key]['eigenvectors']

def eigenvalue_equality_objective(params, shared_results):
    """
    Objective function to enforce eigenvalue equality by minimizing squared differences.
    """
    eigenvalues, _ = get_eigenvalues(params, shared_results)
    eig_diffs = [(eigenvalues[i] - eigenvalues[j])**2 for i in range(len(eigenvalues)) for j in range(i+1, len(eigenvalues))]
    return np.sum(eig_diffs)

def delta_constraint(params, shared_results, Delta_E):
    """
    Constraint to ensure dDelta_dt(Delta, eta, Delta_E, gamma, kappa) = 0.
    """
    eigenvalues, eigenvectors = get_eigenvalues(params, shared_results)  # Ensure consistent eigenvalues
    Delta = params[:-3]
    S_k, delta_hat, kappa = params[-3:]
    eta, _, _, _ = compute_eta_decomp(Delta, shared_results['T_k'], shared_results['T1'], S_k, delta_hat, 
                                      shared_results['F_k'], shared_results['R'], shared_results['zg'], 
                                      shared_results['k'], shared_results['num_eigenvalues'])
    return dDelta_dt(Delta, eta, Delta_E, 1, kappa)

def optimization_wrapper(init_params, T_k, T1, F_k, R, zg, k, Delta_E, num_eigenvalues=None):
    """
    Wrapper function to perform optimization for Delta, S_k, delta_hat, and kappa,
    ensuring eigenvalues are tracked per parameter set.
    """
    dz = zg[1] - zg[0]

    # Shared results dictionary to store intermediate calculations
    shared_results = {
        'T_k': T_k,
        'T1': T1,
        'F_k': F_k,
        'R': R,
        'zg': zg,
        'k': k,
        'dz': dz,
        'num_eigenvalues': num_eigenvalues
    }
    
    # Constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda params: delta_constraint(params, shared_results, Delta_E)
    }]
    
    # Bounds
    N = len(init_params) - 3  # First N values are Delta
    bounds = Bounds(
        [0] * N + [0, 0, 0],  # Lower bounds for Delta, S_k, delta_hat, and kappa
        [np.inf] * N + [np.inf, 1, np.inf]  # Upper bounds for Delta and delta_hat <= 1
    )
    
    # Perform optimization
    result = minimize(
        lambda params: eigenvalue_equality_objective(params, shared_results),
        init_params,
        constraints=constraints,
        bounds=bounds,
        method='SLSQP',
        options={'disp': True, 'maxiter': 1000}
    )
    
    if result.success:
        u, s, vh = svd(build_LDelta(result.x[:-3], T_k, T1))

        # Find singular values close to zero (within tolerance)
        null_mask = (s <= 1e-10)
        null_space = np.compress(null_mask, vh, axis=0)

        # Return the transpose of nullspace to get basis vectors
        return result, null_space.T
    
    return result, None








    









