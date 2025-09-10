import numpy as np
from numpy.linalg import norm, eigvals, eigh, eig
from scipy.optimize import root, basinhopping
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import qmc


from dynamical_system import full_system_real
from steady_state_analysis import compute_eta_decomp

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

class Solution:
    """
    A solution that stores everything (state + parameters) in one array x.
    We define fuzzy equality (within a tolerance) and a hash based on rounding
    so that we can store these in a Python set or dictionary.

    Typical usage:
      x has shape (3N + 2,)
        - first 3N entries = state vector (Delta, Re(eta), Im(eta)) 
        - last 2 entries   = alpha1, alpha2
    """

    __slots__ = ["x", "tol", "_hash"]

    def __init__(self, x, tol=1e-6):
        """
        x : 1D np.ndarray of length 3N+2
        tol : float, tolerance for considering two solutions 'equal'.
        """
        # Copy x so we don't mutate user's array
        self.x = np.array(x, copy=True, dtype=float)
        self.tol = tol

        # Precompute a hash: we round the entries of x 
        # to some decimal so that small floating errors 
        # do not break the hashing.
        decimals = int(-np.log10(self.tol)) 
        rounded_tuple = tuple(np.round(self.x, decimals=decimals))
        self._hash = hash(rounded_tuple)

    def __eq__(self, other):
        if not isinstance(other, Solution):
            return False
        return np.max(abs(self.x - other.x)) < self.tol

    def __hash__(self):
        # Hash based on rounded norm-consistent key
        return self._hash

    def __repr__(self):
        return (f"Solution("
                f"alpha1={self.alpha1:.3g}, "
                f"alpha2={self.alpha2:.3g}, "
                f"mean={np.split(self.y, 3)[0].mean():.4g}, "
                f"norm={norm(self.y):.4g})")
    
    def __copy__(self):
        return Solution(self.x.copy())
    
    def copy(self):
        return self.__copy__()
    

    def compute_stability(self, params: dict) -> bool:
        """
        Returns True if the solution is linearly stable (all eigenvalues have negative real part).
        """
        N = params['N']
        y = self.y
        local_params = params.copy()
        local_params['hat_Sk'] = self.alpha1
        local_params['kappa'] = self.alpha2

        def fun(z): return full_system_real(z, local_params)
        J = approx_derivative(fun, y, method='2-point', rel_step=1e-6)

        eigs = eigvals(J)
        return np.all(np.real(eigs) < 0)


    @property
    def y(self):
        """
        Return the 'state' part of the solution.
        If you have N in your code, you can slice as needed:
          self.x[:3*N]
        """
        # Here we can't assume we know N at the class level,
        # so let's do a more generic approach:
        # (We know the last 2 entries are parameters.)
        return self.x[:-2]

    @property
    def alpha1(self):
        return self.x[-2]

    @property
    def alpha2(self):
        return self.x[-1]




class Branch:
    """
    A single 'branch' in a multiple-branch continuation.
    Stores solutions in a set, so we don't re-insert near-duplicates.
    """
    def __init__(self, branch_id, initial_solution: Solution):
        self.branch_id = branch_id
        self.open = True
        # We'll store solutions in a set for O(1) membership checks
        self.solutions = set()
        self.current_solution = initial_solution
        self.solutions.add(initial_solution)

    def add_solution(self, new_sol: Solution):
        """
        Add new_sol to the set if it's not already present
        (per fuzzy equality from the Solution class).
        Update current_solution to new_sol if unique,
        or to the existing set member if it's effectively the same.
        """
        self.solutions.add(new_sol)
        self.current_solution = new_sol

    def has_solution(self, sol: Solution):
        """
        Check if 'sol' is in our set (within the solution's tolerance).
        """
        return sol in self.solutions

    def __repr__(self):
        return (f"<Branch {self.branch_id}, #solutions={len(self.solutions)}, "
                f"current=({self.current_solution.alpha1:.3f}, "
                f"{self.current_solution.alpha2:.3f})>")
    


#-------------------------------------------------------
# Initializing \eta and \Delta
#-------------------------------------------------------

def generate_delta_ref(z, alpha=1.0):
    """
    Generate a physically plausible reference delta(z) profile representing
    the polar vortex edge as a sharp PV gradient.

    Parameters
    ----------
    z : array_like
        1D vertical coordinate (e.g., log-pressure or isentropic height).
    alpha : float
        Strength scaling (0 < alpha ≤ 1). alpha = 1 means full-strength vortex;
        smaller alpha mimics a weakened or pre-warming vortex.

    Returns
    -------
    delta_ref : np.ndarray
        Real-valued array, same shape as z. Typically negative (NH).
    """
    z = np.asarray(z)
    z1 = np.percentile(z, 40)   # inner wall center
    z2 = np.percentile(z, 60)   # outer wall center
    sigma1 = 0.05 * (z[-1] - z[0])
    sigma2 = 0.1 * (z[-1] - z[0])

    # Double Gaussian profile to mimic sharp edge + shoulder
    profile = (
        np.exp(-(z - z1)**2 / (2 * sigma1**2)) +
        np.exp(-(z - z2)**2 / (2 * sigma2**2))
    )
    profile /= np.max(profile)  # normalize so max = 1

    # Apply vortex strength and NH sign convention
    return -alpha * profile


def delta_basis_smooth(z, n_delta=5):
    """
    Return first n_delta eigenvectors of -d²/dz² (Neumann BC).
    """
    N = len(z)
    dz = np.diff(z)
    main = np.zeros(N)
    upper = np.zeros(N - 1)
    lower = np.zeros(N - 1)

    # interior
    for i in range(1, N - 1):
        dz_f = dz[i]
        dz_b = dz[i - 1]
        main[i] = -2.0 / (dz_f * dz_b)
        upper[i] = 2.0 / (dz_f * (dz_f + dz_b))
        lower[i - 1] = 2.0 / (dz_b * (dz_f + dz_b))

    # Neumann ends
    main[0] = main[-1] = -2.0 / dz[0] ** 2
    upper[0] = lower[-1] = 2.0 / dz[0] ** 2

    L = np.diag(main) + np.diag(upper, 1) + np.diag(lower, -1)
    eigvals, eigvecs = eigh(-L)               # symmetric
    idx = np.argsort(eigvals)
    basis = eigvecs[:, idx[:n_delta]]
    basis /= np.max(np.abs(basis), axis=0, keepdims=True)
    return basis

def build_eta_basis(delta_ref, kernel_matrix, S_k, n_eta=3):
    M = 1j * S_k * (delta_ref[:, None] * kernel_matrix)
    eigvals, eigvecs = eig(M)
    growth = eigvals.imag
    idx = np.argsort(growth)[::-1]
    basis = eigvecs[:, idx[:n_eta]]
    basis /= np.max(np.abs(basis), axis=0, keepdims=True)
    return basis, eigvals[idx[:n_eta]]

def build_seeds(z, delta_ref, delta_basis, eta_basis,
                n_seed=10, noise_level=0.01, seed=0):
    """
    Create initial guesses for (delta, eta) fields to span function space.

    Parameters
    ----------
    z : array_like, shape (N,)
        Vertical coordinate array.
    delta_ref : array_like, shape (N,)
        Reference delta profile (e.g., staircase PV gradient).
    delta_basis : array_like, shape (N, n_delta)
        Low-order eigenfunctions to span delta space.
    eta_basis : array_like, shape (N, n_eta)
        Unstable complex eigenmodes to span eta space.
    forcing : array_like, shape (N,)
        Forcing term F_k(z).
    kernel_matrix : array_like, shape (N, N)
        Kernel T^k(z, z').
    params : dict
        Dictionary with keys: 'gamma', 'kappa', 'hat_Sk', 'hat_delta'.
    n_seed : int
        Number of initial condition sets to generate.
    noise_level : float
        Amount of small white noise added to delta_ref.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    seeds : list of tuples
        Each element is a (delta0, eta0) tuple, where:
        - delta0 is real-valued, shape (N,)
        - eta0 is complex-valued, shape (N,)
    """
    rng = np.random.default_rng(seed)
    n_delta = delta_basis.shape[1]
    n_eta = eta_basis.shape[1]

    # --- Latin Hypercube Sampling using SciPy --------------------------
    sampler = qmc.LatinHypercube(d=n_delta + 2 * n_eta, seed=rng)
    cube = sampler.random(n=n_seed)  # shape (n_seed, total_dim)

    seeds = []
    for row in cube:
        # split row
        delta_raw = row[:n_delta]
        eta_amp_raw = row[n_delta:n_delta + n_eta]
        eta_phase_raw = row[n_delta + n_eta:]

        # map delta coefficients (signed log-uniform)
        delta_coeff = np.sign(delta_raw - 0.5) * 10**(-1 + 1.3 * delta_raw)
        delta0 = delta_ref + delta_basis @ delta_coeff

        # map eta coefficients (log-uniform amplitudes, uniform phases)
        eta_amps = 10**(-1.3 + 1.6 * eta_amp_raw)
        eta_phases = 2 * np.pi * eta_phase_raw
        eta0 = (eta_basis * eta_amps) @ np.exp(1j * eta_phases)

        # apply small-scale noise to delta only
        delta0 *= 1 + noise_level * rng.standard_normal(len(z))

        seeds.append(np.concatenate([delta0, eta0.real, eta0.imag]))

    return seeds


#-------------------------------------------------------
# Solver Code 
#-------------------------------------------------------

def deflation_factor(y, known_solutions, p=2, eps=1e-12):
    """
    known_solutions: a list of y-vectors (not entire Solutions).
    We push the solver away from any known y in y-space.
    """
    D = 1.0
    from numpy.linalg import norm
    for y_known in known_solutions:
        dist = norm(y - y_known)
        if dist < eps:
            # We are extremely close => blow up
            return np.inf
        D *= (1.0 + 1.0/(dist**p))
    return D

def deflated_residual(y, known_solutions, params):
    F = full_system_real(y, params)
    D = deflation_factor(y, known_solutions)
    if np.isinf(D):
        return 1e6*np.ones_like(F)
    return D * F

def deflated_jacobian(y, known_solutions, params):
    def f_wrapped(x):
        return deflated_residual(x, known_solutions, params)
    return approx_derivative(f_wrapped, y, method='2-point', rel_step=1e-6)

def find_all_solutions_at_params(params, delta_ref=None, delta_basis=None, eta_basis=None, starting_solution = None, n_starts=10, sweep_range=(-0.5, 0.5), tol=1e-6, solver_tol = 1e-6):
    """
    Multi-start deflation approach. 
    We assume alpha1, alpha2 are set in `params` already.
    Returns a list of y-vectors in R^{3N}.
    params = {'hat_Sk', 'kappa', 'hat_delta', 'gamma', 'T_k', 'T_1', 'F_k', 'Delta_E', 'N', 'R', 'k'}
    """
    N = params['N']
    found_solutions = []
    if starting_solution is None:
        found_solutions.append(starting_solution)

    if starting_solution is None:
            starting_solution = build_seeds(
                z=params['zg'],
                delta_ref=delta_ref,
                delta_basis=delta_basis,
                eta_basis=eta_basis,
                n_seed=n_starts,
                noise_level=0.01,
                seed=42
            )[0]
    
    if sweep_range is None:
        if starting_solution is None:
            starting_sols = build_seeds(
                    z=params['zg'],
                    delta_ref=delta_ref,
                    delta_basis=delta_basis,
                    eta_basis=eta_basis,
                    n_seed=n_starts,
                    noise_level=0.01,
                    seed=42
                )
        else:
            starting_sols = [starting_solution] + build_seeds(
                    z=params['zg'],
                    delta_ref=delta_ref,
                    delta_basis=delta_basis,
                    eta_basis=eta_basis,
                    n_seed=n_starts-1,
                    noise_level=0.01,
                    seed=42
                )
    else:

        zg = params['zg']
        #Rather than complete randomness, just cehck near Delta and compute a new \eta
        starting_sols = starting_solution + [np.minimum(np.maximum(starting_solution + np.random.uniform(sweep_range[0], sweep_range[1], N), 0), 1) for _ in range(n_starts-1)]

    for i in range(n_starts):
        y0 = starting_sols[i]

        res = root(
            fun=lambda y: deflated_residual(y, found_solutions, params),
            x0=y0,
            #jac=lambda y: deflated_jacobian(y, found_solutions, params),
            method='hybr',
            #options={'maxiter': 1000},
            tol=solver_tol
        )
        if res.success:
            y_sol = res.x
            # check if new
            is_new = True
            for yk in found_solutions:
                if norm(y_sol - yk) < tol:
                    is_new = False
                    break
            if is_new:
                found_solutions.append(y_sol)
        else:
            continue
            break

    return found_solutions

def G_pseudoarclength_2D(X, X_old, t_old_1, t_old_2, ds_1, ds_2, params):
    """
    X: (3N+2,)
    Returns G(X) in R^{3N+1}.
      - first 3N = full_system_real(y, {alpha1, alpha2})
      - last 1   = dot(X - X_old, t_old) - ds
    """
    N = params['N']
    y       = X[:3*N]
    alpha1  = X[3*N]
    alpha2  = X[3*N+1]

    # local copy of params
    local_params = params.copy()
    local_params['hat_Sk'] = alpha1
    local_params['kappa']  = alpha2

    F_val = full_system_real(y, local_params)  # shape (3N,)

    arc1 = np.dot(X - X_old, t_old_1) - ds_1
    arc2 = np.dot(X - X_old, t_old_2) - ds_2
    return np.concatenate([F_val, [arc1, arc2]])


def pseudoarclength_step_2D(
    X_current, 
    t_old_1, 
    t_old_2, 
    ds_1, 
    ds_2, 
    params,
    solver_method='hybr'
):
    """
    X_current : (3N+2,)
    t_old_1, t_old_2 : two tangent directions in R^{3N+2}.
    ds_1, ds_2       : step sizes along those directions.
    
    We'll do a simple predictor: X_pred = X_current + ds_1 * t_old_1 + ds_2 * t_old_2
    Then solve G=0 as a corrector.
    """
    # predictor
    X_pred = X_current + ds_1 * t_old_1 + ds_2 * t_old_2

    # corrector
    res = root(
        fun=lambda X: G_pseudoarclength_2D(
            X, X_current, t_old_1, t_old_2, ds_1, ds_2, params
        ),
        x0=X_pred,
        # Optionally provide the Jacobian:
        # jac=lambda X: dG_pseudoarclength_2D(X, X_current, t_old_1, t_old_2, ds_1, ds_2, params),
        method=solver_method
    )
    if not res.success:
        return X_pred, (None, None)
    
    X_new = res.x

    # Now we need to define new tangent directions. Typically you'd compute the 
    # 2D tangent plane from the local Jacobian's null space or something similar.
    # For a minimal example, let's define them as the difference from X_current 
    # (but that only yields one direction...). In real codim-2 codes, you'd do an SVD or partial RQ factor.
    # For demonstration, let's do a naive approach to get at least one direction:
    t_new_1 = X_new - X_current
    n_t1 = norm(t_new_1)
    if n_t1 < 1e-14:
        return X_new, (None, None)  # degenerate

    t_new_1 /= n_t1
    # For the second direction, you might keep t_old_2 or do a Gram-Schmidt with t_new_1, etc.

    # We'll just return the single direction t_new_1 plus the old t_old_2 for demonstration.
    # In a real code, you'd do a local rank-2 approach to find two new tangents that remain in the solution's tangent plane.
    t_new_2 = t_old_2.copy()

    return X_new, (t_new_1, t_new_2)


#-----------------------------------------------
#Continuation Code
#-----------------------------------------------

def multiple_branch_continuation_2D(
    branches,
    ds_1,
    ds_2,
    n_steps,
    params,
    n_starts=5,
    sweep_range=(-0.1, 0.1),
    tol_new=1e-6,
    solver_method='hybr',
    solver_tol=1e-6,
    search_params=['hat_Sk', 'kappa'],
    jump_stop = 2
):
    import numpy as np

    # Fix next_branch_id computation
    next_branch_id = max([b.branch_id for b in branches], default=-1) + 1
    branch_tangents = {}
    delta_ref = generate_delta_ref(params['zg'], alpha=1.0)
    delta_basis = delta_basis_smooth(params['zg'], n_delta=5)

    # === Global solution registry ===
    solution_registry = {}  # rounded tuple of solution.x -> branch_id

    for b in branches:
        for s in b.solutions:
            solution_registry[s] = b.branch_id

    for step in range(1, n_steps + 1):
        print(f"===== 2D Continuation step {step} =====")
        new_branches_this_step = []
        unclaimed_solutions = set() # (solution, X_current, branch_id)

        for b in branches:
            if not b.open:
                continue
            X0 = b.current_solution.x
            dim = len(X0)
            t1 = np.zeros(dim); t1[-2] = 1.0
            t2 = np.zeros(dim); t2[-1] = 1.0
            t1 /= np.linalg.norm(t1) if np.linalg.norm(t1) > 1e-14 else 1
            t2 /= np.linalg.norm(t2) if np.linalg.norm(t2) > 1e-14 else 1
            branch_tangents[b.branch_id] = (t1, t2)

        active_branches = sorted([b for b in branches if b.open], key=lambda b: b.branch_id)

        for branch in active_branches:
            if not branch.open:
                continue

            X_current = branch.current_solution.x
            t_old_1, t_old_2 = branch_tangents[branch.branch_id]

            X_new, (t_new_1, t_new_2) = pseudoarclength_step_2D(
                X_current, t_old_1, t_old_2, ds_1, ds_2, params, solver_method=solver_method
            )
            if X_new is None:
                print(f"[Branch {branch.branch_id}] step failed => stopping branch.")
                continue


            N = params['N']
            alpha1_new, alpha2_new = X_new[3 * N], X_new[3 * N + 1]
            if t_new_1 is not None and t_new_2 is not None:
                branch_tangents[branch.branch_id] = (t_new_1, t_new_2)

            local_params = params.copy()
            local_params[search_params[0]] = alpha1_new
            local_params[search_params[1]] = alpha2_new

            eta_basis, eigvals = build_eta_basis(delta_ref, params['T_k'], local_params['hat_Sk'], n_eta=5)
            list_y = find_all_solutions_at_params(
                local_params,
                delta_ref=delta_ref,
                delta_basis=delta_basis,
                eta_basis=eta_basis,
                starting_solution=X_new[:3 * N],
                n_starts=n_starts,
                sweep_range=sweep_range,
                tol=tol_new,
                solver_tol=solver_tol
            )
            all_solutions_found = []
            for y_found in list_y:
                x_cand = np.concatenate([y_found, [alpha1_new, alpha2_new]])
                sol_cand = Solution(x_cand, tol=tol_new)
                all_solutions_found.append(sol_cand)

            unclaimed_for_branch = set([s for s in all_solutions_found if s not in solution_registry])
            unclaimed_solutions |= unclaimed_for_branch

        # Now assign unclaimed solutions to the closest requesting branch
        assigned_keys = set()
        branches_with_assignment = set()
        for s in unclaimed_solutions:
            if s in assigned_keys:
                continue
            candidates = [
                (b, norm(s.x - b.current_solution.x))
                for b in branches
                if b.branch_id not in branches_with_assignment and norm(s.x - b.current_solution.x) < jump_stop
            ]
            if not candidates:
                continue
            min_dist = min(d for _, d in candidates)
            candidates_min = [b for b, d in candidates if abs(d - min_dist) < tol_new]
            b_target = min(candidates_min, key=lambda b: b.branch_id)
            b_target.add_solution(s)
            solution_registry[s] = b_target.branch_id
            assigned_keys.add(s)
            branches_with_assignment.add(b_target.branch_id)
           
            if b_target.open == False:
                b_target.open = True
                print(f"[Branch {b_target.branch_id}] reopened with => {s}")
            else:
                 print(f"[Branch {b_target.branch_id}] updated => {s}")

        # Close branches that did not receive any assignment this round
        for b in branches:
            if b.open and b.branch_id not in branches_with_assignment:
                b.open = False
                print(f"[Branch {b.branch_id}] closed: no new solution assigned")

        # Spawn new branches for any leftover unclaimed solutions
        for s in unclaimed_solutions:
            if s not in assigned_keys:
                new_branch = Branch(next_branch_id, s)
                new_branch.open = True
                new_branches_this_step.append(new_branch)
                solution_registry[s] = new_branch.branch_id
                assigned_keys.add(s)
                print(f"Spawned new Branch {new_branch.branch_id} with solution {s}")
                next_branch_id += 1

        branches.extend(new_branches_this_step)

    return branches


# def multiple_branch_continuation_2D(
#     branches,
#     ds_1,
#     ds_2,
#     n_steps,
#     params,
#     n_starts=5,
#     sweep_range=(-0.1, 0.1),
#     tol_new=1e-6,
#     solver_method='hybr',
#     solver_tol = 1e-6,
#     search_params = ['hat_Sk', 'kappa']
# ):
#     """
#     branches: list of Branch objects (with initial solutions).
#     ds_1, ds_2: step sizes for the two directions
#     n_steps: how many steps to perform
#     params: PDE/ODE config
#     n_starts, sweep_range: for deflation-based find_all_solutions_at_params
#     tol_new: fuzzy equality tolerance for new solutions
#     solver_method: e.g. 'hybr' or 'lm' for scipy.optimize.root
#     """

#     import numpy as np

#     next_branch_id = 1 + max(b.branch_id for b in branches) if branches else 0

#     # We'll store a dictionary of tangents for each branch.
#     # Each value is (t1, t2) in R^{3N+2}.
#     branch_tangents = {}

#     # Compute a default reference delta profile
#     delta_ref = generate_delta_ref(params['zg'], alpha=1.0)  # e.g. double Gaussian profile

#     # Compute Δ basis (smooth, self-adjoint)
#     delta_basis = delta_basis_smooth(params['zg'], n_delta=5)


#     for step in range(1, n_steps+1):
#         print(f"===== 2D Continuation step {step} =====")
#         new_branches_this_step = []

#         # Initialize tangents for each branch
#         for b in branches:
#             if b.open == False:
#                 continue
#             X0 = b.current_solution.x
#             dim = len(X0)  # should be 3N+2
#             # A naive approach: set t1 to bump alpha1, t2 to bump alpha2
#             t1 = np.zeros(dim)
#             t1[-2] = 1.0  # bump alpha1
#             norm_t1 = np.linalg.norm(t1)
#             if norm_t1 > 1e-14:
#                 t1 /= norm_t1

#             t2 = np.zeros(dim)
#             t2[-1] = 1.0  # bump alpha2
#             norm_t2 = np.linalg.norm(t2)
#             if norm_t2 > 1e-14:
#                 t2 /= norm_t2

#             branch_tangents[b.branch_id] = (t1, t2)
        
#         # We take a snapshot to avoid newly spawned branches stepping in the same iteration
#         active_branches = list(branches)

#         for branch in active_branches:
#             if branch.open == False:
#                 continue

#             sol_current = branch.current_solution
#             X_current   = sol_current.x

#             # get tangents for this branch
#             (t_old_1, t_old_2) = branch_tangents[branch.branch_id]

#             # 1) Do a 2D pseudo-arclength step
#             X_new, (t_new_1, t_new_2) = pseudoarclength_step_2D(
#                 X_current,
#                 t_old_1,
#                 t_old_2,
#                 ds_1,
#                 ds_2,
#                 params,
#                 solver_method=solver_method
#             )
#             if X_new is None:
#                 print(f"[Branch {branch.branch_id}] step failed => stopping branch.")
#                 continue

#             # parse out new param
#             N = params['N']
#             alpha1_new = X_new[3*N]
#             alpha2_new = X_new[3*N+1]

#             # 2) If we successfully got new tangents, store them
#             if t_new_1 is not None and t_new_2 is not None:
#                 branch_tangents[branch.branch_id] = (t_new_1, t_new_2)

#             # 3) Now do deflation-based search for all solutions at (alpha1_new, alpha2_new)
#             local_params = params.copy()
#             local_params[search_params[0]] = alpha1_new
#             local_params[search_params[1]]  = alpha2_new

#             # Compute η basis: m most unstable eigenmodes of i S_k * diag(delta_ref) @ T_k
#             eta_basis, eigvals = build_eta_basis(delta_ref, params['T_k'], local_params['hat_Sk'], n_eta=5)

#             # We'll pass X_new (the state) as a "starting_solution" if you like
#             list_y = find_all_solutions_at_params(
#                 local_params,
#                 delta_ref = delta_ref,
#                 delta_basis = delta_basis,
#                 eta_basis = eta_basis,
#                 starting_solution=X_new[:3*N],
#                 n_starts=n_starts,
#                 sweep_range=sweep_range,
#                 tol=tol_new,
#                 solver_tol=solver_tol
#             )
#             print(f"{len(list_y)} solutions found")
#             if len(list_y) == 0:
#                 print(f"[Branch {branch.branch_id}] no solutions => fold or no surface here.")
#                 continue
            
#             # Convert each to a Solution
#             all_solutions_found = []
#             for y_found in list_y:
#                 x_cand = np.concatenate([y_found, [alpha1_new, alpha2_new]])
#                 sol_cand = Solution(x_cand, tol=tol_new)
#                 all_solutions_found.append(sol_cand)

#             # 4) Pick the closest solution to X_new => update the branch
#             def dist_from_xnew(s_cand):
#                 return np.linalg.norm(s_cand.x - X_new)
            
#             s_closest = min(all_solutions_found, key=dist_from_xnew)

#             # 5) Assign solutions to branches (with merging logic)
#             for s_other in all_solutions_found:
#                 if s_other == s_closest:
#                     # Check if the closest solution exists in any other branch
#                     for other_branch in branches:
#                         if other_branch.has_solution(s_closest):
#                             if branch.branch_id > other_branch.branch_id:
#                                 branch.open = False
#                                 print(f"Branch {branch.branch_id} closed due to conflict with Branch {other_branch.branch_id}")
#                             else:
#                                 other_branch.open = False
#                                 print(f"Branch {branch.branch_id} closed due to conflict with Branch {other_branch.branch_id}")
#                             break
#                     if branch.open:
#                         # Add the closest solution to the current branch
#                         branch.add_solution(s_closest)
#                         print(f"[Branch {branch.branch_id}] updated => {branch.current_solution}")
                    
#                 else:
#                     # Check if the solution exists in any branch
#                     is_new_solution = True
#                     for other_branch in branches:
#                         if other_branch.has_solution(s_other):
#                             is_new_solution = False
#                             break

#                     # Spawn a new branch for truly new solutions
#                     if is_new_solution:
#                         new_branch = Branch(next_branch_id, s_other)
#                         new_branch.open = True
#                         new_branches_this_step.append(new_branch)
#                         next_branch_id += 1
#                         print(f"Spawned new Branch {new_branch.branch_id} with solution {s_other}")

#             # Update the branches list with newly spawned branches
#         branches.extend(new_branches_this_step)

#     return branches


def visualize_branches(branches, params, parameter='alpha1', parameter_label=None, value_label='mean(\Delta)', stability_stride=5):
    """
    Visualize the branches by plotting the absolute value of solutions
    against a specified parameter.

    Parameters:
        branches (list): List of Branch objects.
        parameter (str): Parameter to plot on the x-axis ('alpha1' or 'alpha2').
        value_label (str): Label for the y-axis (default: '|Solution|').
    """
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 6))
    for branch in branches:
        # Extract parameter values and absolute solution values for the branch
        param_values = []
        abs_values = []
        stability_flags = []

        sorted_solutions = sorted(branch.solutions, key=lambda s: getattr(s, parameter))

        for idx, sol in enumerate(sorted_solutions):
            if parameter == 'alpha1':
                param_values.append(sol.alpha1)
            elif parameter == 'alpha2':
                param_values.append(sol.alpha2)
            else:
                raise ValueError("Parameter must be 'alpha1' or 'alpha2'")

            abs_values.append(np.split(sol.y, 3)[0].mean())  # Absolute value of the Delta component

            # Stability: compute only every nth point
            if idx % stability_stride == 0:
                is_stable = sol.compute_stability(params)
            stability_flags.append(is_stable)

        param_values = np.array(param_values)
        abs_values = np.array(abs_values)
        stability_flags = np.array(stability_flags)

        # Plot segment-by-segment, using stability flag for linestyle
        for i in range(len(param_values) - 1):
            style = '-' if stability_flags[i] else '--'
            color = colors[branch.branch_id % len(colors)]
            plt.plot(param_values[i:i+2], abs_values[i:i+2], style, color=color)

    plt.xlabel(parameter_label or parameter)
    plt.ylabel(value_label)
    plt.title(f"Branch Visualization ({parameter_label or parameter} vs {value_label}) for N={params['N']}")
    plt.grid()
    plt.tight_layout()
    plt.show()



    #==========================================================
    #OLD HELPERS
    #===========================================================

    def dG_pseudoarclength(X, X_old, t_old, ds, params):
        def fun(Z):
            return G_pseudoarclength(Z, X_old, t_old, ds, params)
        J = approx_derivative(fun, X, method='2-point', rel_step=1e-6)
        return J
