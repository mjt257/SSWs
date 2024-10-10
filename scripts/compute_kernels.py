import numpy as np
from scipy.integrate import quad
from scipy.special import iv, kn
from const import *
import time
from multiprocessing import Pool

#Constants
gamma0=np.sqrt(Kappa*(1.-Kappa))
tol = 1e-8

#Defintions
Rs = lambda x,y: (x, y) if y > x else (y, x) # To sort x and y

gamma = lambda m: np.sqrt((m**2.)+1/4.)


def IK(x1, x2, k):
   '''
   Function to compute the product of the modified Bessel Functions of the First and Second kind

   :param x1: argument of modified bessel function of the first kind
   :type x1: float
   :param x2: argument of modified bessel function of the second kind
   :type x2: float
   :param k: order of the bessel functions
   :type k: int
   '''
   mu = 4*k**2
   if x1 > 500.:     # Use large-argument expansions for iv and kn to avoid overflow in exponential
      ix1 = 1. - (mu-1.)/(8.*x1) \
             + (mu-1.)*(mu-9.)/(2*(8.*x1)**2) \
             - (mu-1.)*(mu-9)*(mu-25.)/(6*(8.*x1)**3)
      kx2 = 1. + (mu-1.)/(8.*x2) \
             + (mu-1.)*(mu-9.)/(2*(8.*x2)**2) \
             + (mu-1.)*(mu-9)*(mu-25.)/(6*(8.*x2)**3)
      return ix1 * kx2 * np.exp(x1 - x2) / np.sqrt(4*x1*x2)
   else:
      return iv(k,x1)*kn(k,x2)

def Tk(zi, zj, k, Ri, Rj, dz, symmetric=True):
   """
   Function to compute T(z, z') for given values
   Note: T_k is the S_k kernel with Delta pulled out. 

   :param zi: z
   :type zi: float
   :param zj: z'
   :type zj: float
   :param k: wave number
   :type k: int
   :param Ri: R(z)
   type Ri: float
   :param Rj: R(z')
   :type Rj: float
   :param dz: z step to use in the integration
   :type dz: float
   :param symmetric: flag to denote whether we are computing the kernel symmetrically, default = True
   :type symmetric: boolean, optional

   :return: value of T(z, z'), with I_0 (barotropic) and I_m (baroclinic) part
   :rtype: float
   """
   rm, rp = Rs(Ri, Rj)
   zs = zi + zj
   zd = zi - zj
   omk = 1. - Kappa
   om2k = 1 - 2.*Kappa
   dzh = dz/2.
   csh = np.cosh(dzh)
   snh = np.sinh(dzh)

   # Barotropic part
   I0 = om2k*IK(gamma0*rm, gamma0*rp, k) * \
         np.exp(-omk*zs) * np.sinh(omk*dz) / (omk*dz) #= e^(kappa - 1)(zi + zj) * (e^{(1-kappa)dz} - exp^{(kappa - 1)dz})(2 * (1-kappa) * dz) = e^(kappa * zi + kappa*zj -  zi - zj) (whichever minus persists is z')

   #sinh(omk * dz) = e^([1-kappa]dz), * e^(-omk * zs) = kappa*zi + kappa*zj - zi - zj + dz - kappa * dz
   # Baroclinic part

   # Integral of cos(m (zi - zj))
   def g0p(m): # Integrand for m > 1
      gm = gamma(m) 
      return IK(gm*rm, gm*rp, k) / m

   def g0m(m, w): # Integrand for 0 <= m <= 1
      gm = gamma(m) 
      return IK(gm*rm, gm*rp, k) * w * np.sinc(w * m / np.pi)

   # Need to split domain of integral to avoid numerical issues in quad when |wvar| > 1.
   Dsp1, eDsp1 = quad(g0m, 0., 1., args=(zd + dz,), epsabs=1e-8, epsrel=1e-8)
   Dsm1, eDsm1 = quad(g0m, 0., 1., args=(zd - dz,), epsabs=1e-8, epsrel=1e-8)

   Dsp2, eDsp2 = quad(g0p, 1., np.inf, weight='sin', wvar=zd + dz, epsabs=1e-8, epsrel=1e-8)
   Dsm2, eDsm2 = quad(g0p, 1., np.inf, weight='sin', wvar=zd - dz, epsabs=1e-8, epsrel=1e-8)

   #Dsp1a, eDsp1a = quad(g0p, 0 ,     1., weight='sin', wvar=zd + dz, epsabs=1e-8, epsrel=1e-8)
   #Dsm1a, eDsm1a = quad(g0p, 0.,     1., weight='sin', wvar=zd - dz, epsabs=1e-8, epsrel=1e-8)

   Dsp = Dsp1 + Dsp2
   Dsm = Dsm1 + Dsm2

   # Integral of cos(m (zi + zj) + 2 theta(m))
   def c2t(m): return (4*m**2 - om2k**2) / (4*m**2 + om2k**2)
   def s2t(m): return (4*m*om2k)         / (4*m**2 + om2k**2)

   def cp(m): gm = gamma(m); return IK(gm*rm, gm*rp, k)/gm**2 * \
         (csh*(+2*m*s2t(m) +c2t(m)) + snh*(+c2t(m) +2*m*s2t(m)))
   def cm(m): gm = gamma(m); return IK(gm*rm, gm*rp, k)/gm**2 * \
         (csh*(-2*m*s2t(m) -c2t(m)) + snh*(+c2t(m) +2*m*s2t(m)))
   def sp(m): gm = gamma(m); return IK(gm*rm, gm*rp, k)/gm**2 * \
         (csh*(+2*m*c2t(m) -s2t(m)) + snh*(-s2t(m) +2*m*c2t(m)))
   def sm(m): gm = gamma(m); return IK(gm*rm, gm*rp, k)/gm**2 * \
         (csh*(-2*m*c2t(m) +s2t(m)) + snh*(-s2t(m) +2*m*c2t(m)))

   Scp, eScp = quad(cp, 0, np.inf, weight='cos', wvar=zs + dz, epsabs=1e-8, epsrel=1e-8)
   Scm, eScm = quad(cm, 0, np.inf, weight='cos', wvar=zs - dz, epsabs=1e-8, epsrel=1e-8)
   Ssp, eSsp = quad(sp, 0, np.inf, weight='sin', wvar=zs + dz, epsabs=1e-8, epsrel=1e-8)
   Ssm, eSsm = quad(sm, 0, np.inf, weight='sin', wvar=zs - dz, epsabs=1e-8, epsrel=1e-8)

   Im = np.exp(-zs/2.)/np.pi * ( \
               np.sinh(dzh) / dz**2 * (Dsp - Dsm) \
               + (Scp + Scm + Ssp + Ssm) / (4*dz)) 
   #since zs =  zi + zj, then -zs = (-zi - zj)/2 --
   #  this term in the equation is e^{(z-z')/2}
   #note -- we multiply everything by e^(zi) -- terns this term into zi - zi/2 - zj/2 = zi/2 - zj/2 => zj = z' and zi = z

   res = I0 + Im 
   if symmetric: res = Rj/Ri * res * np.exp(zi) #pretty sure we always want this
   return res

"""
As per the paper, we create a symmetry condition, knowing that
S(z, z')\Delta(z)R(z)^2e^{-z} = S(z', z)\Delta(z')R(z')^e^{-z'}
=> S(z, z')*\Delta(z)/\Delta(z')*[R(z)/R(z')]^2*e{-z + z'} = S(z' z)
"""
def symmetry_coeff(z, zpri, delta_z, delta_zpri, R_z, R_zpri):
   '''
   Computes a coefficient for a given z, zpri such that T_k(z', z) = coeff * T_k(z, z')
   Uses S(z, z')\Delta(z)R(z)^2e^{-z} = S(z', z)\Delta(z')R(z')^e^{-z'} => S(z, z')*\Delta(z)/\Delta(z')*[R(z)/R(z')]^2*e{-z + z'} = S(z' z)

   :param z: z
   :type z: float
   :param zpri: z'
   :type zpri: float
   :param delta_z: Delta(z)
   :type delta_z: float
   :param delta_zpri : Delta(z')
   :type delta_zpri: float
   :param R_z: R(z)
   :type R_z: float
   :param R_zpri: R(z')
   :type R_zpri: float

   :returns: symmetric coeff Delta(z)/Delta(z') * (R(z)/R(z'))^2 * e^(z' - z)
   :rtype: float
   '''
   return delta_z/delta_zpri * ((R_z/R_zpri)**2) * np.exp(zpri - z)


def Li(z_bottom, z_top, N_steps, k, R_func, Delta_func = None) :
   '''
   Compute Li matrix (T_k(z, z') for all z and z' on the discrete grid). Not parallelized. Uses the symmetry condition

   :param z_bottom: z_b; bottom of the vortex
   :type z_botton: float
   :param z_top: z_t; top of the vortex
   :type z_top: float
   :N_steps: number of discrete steps
   :type N_steps: int
   :param k: wavenumber
   :type k: int
   :param R_func: R function
   :type R_func: function
   :param Delta_func: Delta_function, default = None (ones)
   :type Delta_func: function, optional

   :return: tuple of z discretization, omega_c, and Li
   :rtype: tuple(np.array(float))
   '''
   #Define a discretization from z_bottom to z_top with N_steps
   z_layers, dz = np.linspace(z_bottom, z_top, N_steps, retstep=True)

   #Compute arrays for R_func and Delta_func
   R_layers = R_func(z_layers)

   #Accept input of a function or not; otherwise produce general operator for use 
   Delta_layers = np.ones(N_steps) if Delta_func is None else Delta_func(z_layers)

   omega_c = np.zeros((N_steps,N_steps), 'd')
   lin_op = np.zeros((N_steps,N_steps), 'd')

   #z_layers[i] represents z and z_layers[j] represents z'
   for i in range(N_steps):
      z = z_layers[i]
      R_z = R_layers[i]
      Delta_z = Delta_layers[i]

      for j in range(i+1, N_steps):
         zpri = z_layers[j]
         R_zpri = R_layers[j]
         Delta_zpri = Delta_layers[j]

         #Compute S(z, z')
         lin_op[i][j] = Delta_zpri * Tk(z, zpri, k, R_z, R_zpri, dz)
         omega_c[i][j] = Delta_zpri * Tk(z, zpri, 1, R_z, R_zpri, dz)

         #Use symmetry condition to compute S(z', z)
         lin_op[j][i] = symmetry_coeff(z, zpri, Delta_z, Delta_zpri, R_z, R_zpri) * lin_op[i][j]
         omega_c[j][i] = symmetry_coeff(z, zpri, Delta_z, Delta_zpri, R_z, R_zpri) * omega_c[i][j]

      #Compute the ignored diagonal
      lin_op[i][i] = Delta_z * Tk(z, z, k, R_z, R_z, dz)
      omega_c[i][i] = Delta_z * Tk(z, z, 1, R_z, R_z, dz)
    
   return z_layers, omega_c, lin_op

def worker_Li(args):
   '''
   Worker function used when computing the linear operator using multiprocessing. Computes S_k and S_1 from T_k and T_1 (by multiplying by Delta)

   :param args: tuple of i, j, z, zpri, Delta_z, Delta_zpri, R_z, R_zpri, k, dz
   :type args: tuple
   :returns i, j, S_k_ij, S_1_ij
   '''
   i, j, z, zpri, Delta_z, Delta_zpri, R_z, R_zpri, k, dz = args
   S_k_ij = Delta_zpri * Tk(z, zpri, k, R_z, R_zpri, dz)
   S_1_ij = Delta_zpri * Tk(z, zpri, 1, R_z, R_zpri, dz)
   return i, j, S_k_ij, S_1_ij

def parallel_Li(z_bottom, z_top, N_steps, k, R_func, Delta_func=None, num_workers = None):
   '''
   Compute Li matrix (T_k(z, z') for all z and z' on the discrete grid) in a parallelized fashion

   :param z_bottom: z_b; bottom of the vortex
   :type z_botton: float
   :param z_top: z_t; top of the vortex
   :type z_top: float
   :N_steps: number of discrete steps
   :type N_steps: int
   :param k: wavenumber
   :type k: int
   :param R_func: R function
   :type R_func: function
   :param Delta_func: Delta_function, default = None (ones)
   :type Delta_func: function, optional
   :param num_workers: number of workers to used for parallelization, default = None
   :type num_workers: int
   :return: tuple of z discretization, omega_c, and Li
   :rtype: tuple(np.array(float))

   ..Warning: Any call must be wrapped with 'if __name__ == '__main__':'
   '''
   z_layers, dz = np.linspace(z_bottom, z_top, N_steps, retstep=True)
   R_layers = R_func(z_layers)
   Delta_layers = np.ones(N_steps) if Delta_func is None else Delta_func(z_layers)

   #singel layer case
   if np.isnan(dz):
      dz = np.finfo(float).eps*2
    
   # Prepare arguments for multiprocessing
   tasks = []
   #Recall z_layers[i] = z and z_layers[j] = z'
   for i in range(N_steps):
      for j in range(i, N_steps):  # Only compute upper triangular and diagonal
         args = (i, j, z_layers[i], z_layers[j], Delta_layers[i], Delta_layers[j], R_layers[i], R_layers[j], k, dz)
         tasks.append(args)
    
   # Execute in parallel
   with Pool(processes = num_workers) as pool:
      results = pool.map(worker_Li, tasks)
    
   # Initialize matrices
   lin_op = np.zeros((N_steps, N_steps), 'd')
   omega_c = np.zeros((N_steps, N_steps), 'd')

   # Fill matrices based on results
   for i, j, S_k_ij, S_1_ij in results:
      lin_op[i, j] = S_k_ij
      omega_c[i, j] = S_1_ij

      # Apply symmetry condition directly
      if (i != j):
         symmetry_coeff_const = symmetry_coeff(z_layers[i], z_layers[j], Delta_layers[i], Delta_layers[j], R_layers[i], R_layers[j])
         lin_op[j, i] = symmetry_coeff_const * S_k_ij
         omega_c[j, i] = symmetry_coeff_const * S_1_ij
    
   return z_layers, omega_c, lin_op


def computeU(r_axis, zb, zt, N, R_func, Delta_func):
   """
   Function to compute the self-induced velocity (i.e. mean zonal winds) of the wave over a given r and z grid

   :params r_axis: the r-values of computation in the grid
   :type r_axis: np.array(float)
   :param z_b: bottom boundary of discretization
   :type z_b: float
   :param z_t:top boundary of discretization
   :type z_t: float
   :param N: number of discretization steps in the z_grid
   :type N: int
   :param R_func: function representing the radius of the unperturbed vortex (R)
   :type R_func: function
   :param Delta_func: function representing the PV jump (Delta)
   :type Delta_func: function
   :return: the self-induced mean zonal winds
   :rtype: np.array(float)

   .. Computation Steps::
      1. Compute the layers of discretization and the values of each function at those layers
      2.Initialize empty array to store computed values
      3. Run through a loop as follows: 
         a. The process is to iterate through each r value and z value in the grid, computing the integral operator S_1 at the point (r, z, z')
         b. We can leverage the symmetry of the kernel in this calculation
            (i, r_value) represents the index of the r grid and the r_value at the grid, which will be used in place of R(z)
            j, which represents the index for which z = z_layers[j]
            k, which represents the index z for which z' = z_layers[k]
      4. Sum over z', representing a Reimann Sum over the layers, to get a zonal distribution as a function of r,z
   """

   #compute the discretization of the vertical layers, from z_b to z_t over N steps
   z_layers, dz = np.linspace(zb, zt, N, retstep=True)
   R_layers = R_func(z_layers)
   Delta_layers = Delta_func(z_layers)

   #Initialize empty array for operator values
   S1 = np.zeros((len(r_axis), N, N), 'd')

   #Loop as described above
   for i, r_value in enumerate(r_axis):
      print(r_value) #track progress
      for j in range(N):
         for k in range(j+1, N):
            S1[i, j, k] = Delta_layers[k] * Tk(z_layers[j], z_layers[k], 1, r_value, R_layers[k], dz) 
            S1[i, k, j] = symmetry_coeff(z_layers[j], z_layers[k], Delta_layers[j], Delta_layers[k], r_value, R_layers[k]) * S1[i, j, k]
         S1[i, j, j] = Delta_layers[j] * Tk(z_layers[j], z_layers[j], 1, r_value, R_layers[j], dz)

   #Sum across z' axis to get zonal winds
   u = np.sum(S1, axis=2) 

   #multiply u by R_layers.T -- I'm not sure why
   u = u * R_layers.T

   #Return u
   return u

def worker_computeU(args): 
   '''
   Worker function used when computing the linear operator using multiprocessing. Computes S_k and S_1 from T_k and T_1 (by multiplying by Delta)

   :param args: tuple of i, j, z, zpri, r_value, R_zpri, Delta_z, Delta_zpri, dz
   :type args: tuple
   :returns (i, j, S1_Value, None)
   '''
   i, j, k, z, zpri, r_value, R_zpri, Delta_zpri, dz = args
   S1_value = Delta_zpri * Tk(z, zpri, 1, r_value, R_zpri, dz)
   return (i, j, k, S1_value, None)

   '''# Check if z != zpri to handle non-diagonal elements; apply symmetry if zpri < z
   if z != zpri:
      S1_value = Delta_zpri * Tk(z, zpri, 1, r_value, R_zpri, dz)
      # Apply symmetry condition to compute S(z', z) based on S(z, z')
      symmetry_value = None #symmetry_coeff(z, zpri, Delta_z, Delta_zpri, r_value, R_zpri) * S1_value
      return (i, j, k, S1_value, symmetry_value)
   else:
      # Diagonal elements where z == zpri
      return (i, j, k, Delta_zpri * Tk(z, z, 1, r_value, R_zpri, dz), None) '''

def parallel_computeU(r_axis, z_axis, zb, zt, N, R_func, Delta_func, num_workers=None):
   z_layers, dz = np.linspace(zb, zt, N, retstep=True) #z_layers corresponds to the vortex layers
   R_layers = R_func(z_layers)
   Delta_axis = Delta_func(z_axis)
   Delta_layers = Delta_func(z_axis) 

   u = np.zeros((len(r_axis), len(z_axis)), dtype=np.float64)
    
   # Parallel computation
   tasks = [(i, j, k, z_value, z_layers[k], r_value, R_layers[k], Delta_layers[k], dz) 
            for i, r_value in enumerate(r_axis) 
            for j, z_value in enumerate(z_axis)
            for k in range(N)]
    
   with Pool(processes=num_workers) as pool:
      results = pool.map(worker_computeU, tasks)
    
   # Organize results into the u matrix
   S1_matrix = np.zeros((len(r_axis), len(z_axis), N), dtype=np.float64)
   for result in results:
      i, j, k, S1_value, symmetry_value = result
      S1_matrix[i, j, k] = S1_value
      if symmetry_value is not None:
            S1_matrix[i, k, j] = symmetry_value

   #Sum across z' axis to get zonal winds
   u = np.sum(S1_matrix, axis=2) 

   #multiply u by R_layers.T -- I'm not sure why
   #u = u * R_layers.T 

   #Return u
   return u, S1_matrix, z_layers



def computeModes(zs, R, delta, L, k):
# {{{
   from scipy.linalg import eigh
   ws, fns = eigh(L)## solver taking the hermitian property in account : ws contains the opposite of the eigen values, fns the eigen vectors
   om = -ws/k ## eigen values
   io = np.argsort(om)## sort the eigen values (from smaller to bigger)
   om = om[io]
   fc = 1 #fac(zs, R, delta).reshape(-1, 1)## normalize vectors
   fns = fns[:, io]/fc
   return om, fns
# }}}

def construct_eta(eigenfunctions, eigenvalues, k, N, zg):
   zt = zg[-1]
   zb = zg[0]
   #constructs eta as a function of z, t, theta
   eta_hat = np.empty(N, dtype=object)
   for z_func, omega, z_idx in zip(eigenfunctions, eigenvalues, range(N)):
      eta_hat[z_idx] = lambda z, t: z_func[np.floor((N-1)/(zt - zb) * (z - zb)).astype(int)] * np.exp(1j*omega*t)

   eta = lambda z, t, theta: np.real(sum([eta_hat[z_idx](z, t) for z_idx in range(N)]) * np.exp(1j * k * theta))

   return eta

def saveModes(R, delta, k, zb, zt, N, name, path='./'):
# {{{
   rs = np.linspace(0, 8, 65)
   zs = np.linspace(0, 7, 57)
   u, e, tU = computeU(rs, zs, R(zs), delta(zs), zs)

   zg, oc, L, tL = Li(R, delta, zb, zt, N, k)
   om, fns = computeModes(zg, R, delta, L, k)

   fn = path + name + '.npz'
   np.savez(fn, rs=rs, zs=zs, u=u, eu=e, tu=tU, zg=zg, oc=oc, L=L, tL = tL, om=om, fns=fns)
# }}}
   





