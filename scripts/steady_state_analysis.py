import numpy as np
from scipy.optimize import fsolve, root, approx_fprime
from scipy.linalg import norm, solve, lstsq, null_space
import const
from dynamical_system import full_system

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
