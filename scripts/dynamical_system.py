import const
import numpy as np
from scipy.special import jv 
from scipy.optimize import root 

def F_k(R, z, k, lambd=0.549,  M=1, h_0=1):
    """
    Function representing the forcing

    :param R: Radius function in array or functional form
    :type R: np.array(float) | function
    :param z: Values of z to evaluate function at
    :type z: np.array(float)
    :param k: Wavenumber
    :type k: int
    :param lambd: Lambda parameter in Bessels (controls topographic shape)
    :type lambd: float
    :param M: Parameter
    :type M: int
    :param h_0: Parameter controlling layer of h_0
    :type h_0: int

    :return: Forcing at each value of z
    :rtype: np.array(z)
    """
    kappa = const.Kappa
    r = R(z) if callable(R) else R
    return (-M * h_0 * jv(k, lambd * r)  /
    ((1 - np.sqrt(1 + 4 * lambd**2)) / 2 - kappa) *
    np.exp(((1 - np.sqrt(1 + 4 * lambd**2)) / 2) * z))/r


def dDelta_dt(Delta, eta, Delta_E, gamma, kappa):
    """
    Function representing d\Delta/dt (nondimensional)

    :param Delta: Vector representing Delta at each layer in the discretization
    :type Delta: np.array(float)
    :param eta: Vector representing eta at each layer
    :type eta: np.array(float)
    :param Delta_E: Vector representing the equilibrium Delta function at each layer
    :type Delta_E: np.array(float)
    :param gamma: time scale parameter
    :type gamma: float
    :param kappa: Forcing parameter
    :type kappa: float

    :return: Value of d\Delta/dt
    :rtype: np.array(float)
    """
    return gamma * (Delta_E/np.max(Delta_E) - Delta - (kappa * Delta * np.conj(eta)*eta))

def deta_dt(Delta, eta, Li, omega_c, S_hat, delta_hat, F_k, dz):
    """
    Function representing d\eta/dt

    :param Delta: Vector representing Delta at each layer in the discretization
    :type Delta: np.array(float)
    :param eta: Vector representing eta at each layer
    :type eta: np.array(float)
    :param Li: Linear operator representing a matrix of T_k computed on the discrete grid
    :type Li: np.array(float)
    :param omega_c: Self-induced velocity integral
    :type omeca_c: np.array(float)
    :param S_k: Parameter
    :type S_k: float
    :param delta_hat: Parameter
    :type delta_hat:  float
    :param F_k: Forcing function
    :type F_k: np.array(float)
    :param dz: dz of discretization
    :type dz: float

    :return: Value of d\Delta/dt
    :rtype: np.array(float)
    """

    '''
    n = len(Delta)
    multiplier_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i, n):
            if (i == j):
                multiplier_matrix[i, i] = Delta[i]
            else:
                multiplier_matrix[i, j] = Delta[j]
                multiplier_matrix[j, i] = Delta[i]

    # Apply the factor of delta to the linear operator (i is the z coordinate, j is the z' coordinate)
    DeltaLi = Li * multiplier_matrix #this is a scaled operator, with z as the rows and z' as columns
    '''

    #First integral. Can compute as Li applied to (Delta * eta)
    term1 = 1j * S_hat * (Li @ (Delta * eta)) * dz

    #Simple multiplication because eta is outside the integral
    term2 = -1j * S_hat * (omega_c @ Delta * dz - delta_hat) * eta

    #Forcing and Damping
    term3 = -1j * F_k/np.max(F_k) - eta

    deta = term1 + term2 + term3
    return deta; 

def full_system(y, constant_params, function_params, dz):
    """
    Function representing equation [d\Delta/dt, d\eta/dt] 

    :param y: vector representing concatenated \Delta, \eta
    :type y: np.array(float)
    :param constant_params: tuple of constant parameters S_hat, delta_hat, kappa, gamma
    :type constant_params: tuple(float)
    :param function_params: tuple of functional (vector) paramters Li, omega_c, Delta_E, F_k
    :type function_params: tuple(np.array(float))
    :param dz: z-step (for discretized integration)
    :type dz: float

    :return: vector representing concatenate d\Delta/dt, d\eta, dt
    :rtype: np.array(float)
    """

    S_hat, delta_hat, kappa, gamma = constant_params
    Li, omegac, Delta_E, F_k = function_params

    Delta, eta = np.split(y, 2)
    Delta = np.real(Delta)
    delta_dt = dDelta_dt(Delta, eta, Delta_E, gamma, kappa)
    eta_dt = deta_dt(Delta, eta, Li, omegac, S_hat, delta_hat, F_k, dz)

    # Return concatenated derivatives
    return np.concatenate([delta_dt, eta_dt])
