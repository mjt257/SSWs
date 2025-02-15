import numpy as np
from numpy.linalg import norm
from scipy.optimize import root, basinhopping
from scipy.optimize._numdiff import approx_derivative

from dynamical_system import full_system_real

import numpy as np
from numpy.linalg import norm

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
        rounded_tuple = tuple(np.round(self.x, decimals=6))
        self._hash = hash(rounded_tuple)

    def __eq__(self, other):
        if not isinstance(other, Solution):
            return False
        # Fuzzy equality: check L2 norm
        return norm(self.x - other.x) < self.tol

    def __hash__(self):
        # Return the precomputed hash
        return self._hash

    def __repr__(self):
        return (f"Solution("
                f"alpha1={self.alpha1:.3g}, "
                f"alpha2={self.alpha2:.3g}, "
                f"||y||={norm(self.y):.3g})")
    
    def __copy__(self):
        return Solution(self.x.copy())
    
    def copy(self):
        return self.__copy__()

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
        if False and new_sol in self.solutions: #pretty sure we don't these lines
            # It's effectively the same => just set current_solution 
            # to the matched existing one. We find that existing item:
            for s in self.solutions:
                if s == new_sol:
                    self.current_solution = s
                    return
        else:
            # brand new => store it
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

def find_all_solutions_at_params(params, starting_solution = None, n_starts=10, sweep_range=(-0.5, 0.5), tol=1e-6, solver_tol = 1e-6):
    """
    Multi-start deflation approach. 
    We assume alpha1, alpha2 are set in `params` already.
    Returns a list of y-vectors in R^{3N}.
    """
    N = params['N']
    found_solutions = []
    if starting_solution is None:
        found_solutions.append(starting_solution)

    if starting_solution is None:
        zg = np.linspace(1, 5, N)
        amplitude = np.random.uniform(0, 1)
        wavelength = np.random.uniform(5, 20)  # Random wavelength
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        starting_delta = amplitude * np.sin(2 * np.pi * zg / wavelength + phase)

        amplitude = np.random.uniform(0, 1)
        wavelength = np.random.uniform(5, 20)  # Random wavelength
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        eta_real_part = amplitude * np.sin(2 * np.pi * zg / wavelength + phase)
        eta_imaginary_part = amplitude * np.cos(2 * np.pi * zg / wavelength + phase)

        starting_solution = np.concatenate([starting_delta, eta_real_part, eta_imaginary_part])

    for i in range(n_starts):
        if i == 0:
            y0 = starting_solution
        elif sweep_range is None:
            zg = np.linspace(1, 5, N)
            amplitude = np.random.uniform(0, 1)
            wavelength = np.random.uniform(5, 20)  # Random wavelength
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase
            starting_delta = amplitude * np.sin(2 * np.pi * zg / wavelength + phase)

            amplitude = np.random.uniform(0, 1)
            wavelength = np.random.uniform(5, 20)  # Random wavelength
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase
            eta_real_part = amplitude * np.sin(2 * np.pi * zg / wavelength + phase)
            eta_imaginary_part = amplitude * np.cos(2 * np.pi * zg / wavelength + phase)

            y0 = np.concatenate([starting_delta, eta_real_part, eta_imaginary_part])
        else:
            y0 = np.minimum(np.maximum(starting_solution + np.random.uniform(sweep_range[0], sweep_range[1], 3*N), 0), 1)
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

def dG_pseudoarclength(X, X_old, t_old, ds, params):
    def fun(Z):
        return G_pseudoarclength(Z, X_old, t_old, ds, params)
    J = approx_derivative(fun, X, method='2-point', rel_step=1e-6)
    return J

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
    solver_tol = 1e-6
):
    """
    branches: list of Branch objects (with initial solutions).
    ds_1, ds_2: step sizes for the two directions
    n_steps: how many steps to perform
    params: PDE/ODE config
    n_starts, sweep_range: for deflation-based find_all_solutions_at_params
    tol_new: fuzzy equality tolerance for new solutions
    solver_method: e.g. 'hybr' or 'lm' for scipy.optimize.root
    """

    import numpy as np

    next_branch_id = 1 + max(b.branch_id for b in branches) if branches else 0

    # We'll store a dictionary of tangents for each branch.
    # Each value is (t1, t2) in R^{3N+2}.
    branch_tangents = {}



    for step in range(1, n_steps+1):
        print(f"===== 2D Continuation step {step} =====")
        new_branches_this_step = []

        # Initialize tangents for each branch
        for b in branches:
            if b.open == False:
                continue
            X0 = b.current_solution.x
            dim = len(X0)  # should be 3N+2
            # A naive approach: set t1 to bump alpha1, t2 to bump alpha2
            t1 = np.zeros(dim)
            t1[-2] = 1.0  # bump alpha1
            norm_t1 = np.linalg.norm(t1)
            if norm_t1 > 1e-14:
                t1 /= norm_t1

            t2 = np.zeros(dim)
            t2[-1] = 1.0  # bump alpha2
            norm_t2 = np.linalg.norm(t2)
            if norm_t2 > 1e-14:
                t2 /= norm_t2

            branch_tangents[b.branch_id] = (t1, t2)
        
        # We take a snapshot to avoid newly spawned branches stepping in the same iteration
        active_branches = list(branches)

        for branch in active_branches:
            if branch.open == False:
                continue

            sol_current = branch.current_solution
            X_current   = sol_current.x

            # get tangents for this branch
            (t_old_1, t_old_2) = branch_tangents[branch.branch_id]

            # 1) Do a 2D pseudo-arclength step
            X_new, (t_new_1, t_new_2) = pseudoarclength_step_2D(
                X_current,
                t_old_1,
                t_old_2,
                ds_1,
                ds_2,
                params,
                solver_method=solver_method
            )
            if X_new is None:
                print(f"[Branch {branch.branch_id}] step failed => stopping branch.")
                continue

            # parse out new param
            N = params['N']
            alpha1_new = X_new[3*N]
            alpha2_new = X_new[3*N+1]

            # 2) If we successfully got new tangents, store them
            if t_new_1 is not None and t_new_2 is not None:
                branch_tangents[branch.branch_id] = (t_new_1, t_new_2)

            # 3) Now do deflation-based search for all solutions at (alpha1_new, alpha2_new)
            local_params = params.copy()
            local_params['hat_Sk'] = alpha1_new
            local_params['kappa']  = alpha2_new

            # We'll pass X_new (the state) as a "starting_solution" if you like
            list_y = find_all_solutions_at_params(
                local_params,
                starting_solution=X_new[:3*N],
                n_starts=n_starts,
                sweep_range=sweep_range,
                tol=tol_new,
                solver_tol=solver_tol
            )
            print(f"{len(list_y)} solutions found")
            if len(list_y) == 0:
                print(f"[Branch {branch.branch_id}] no solutions => fold or no surface here.")
                continue
            
            # Convert each to a Solution
            all_solutions_found = []
            for y_found in list_y:
                x_cand = np.concatenate([y_found, [alpha1_new, alpha2_new]])
                sol_cand = Solution(x_cand, tol=tol_new)
                all_solutions_found.append(sol_cand)

            # 4) Pick the closest solution to X_new => update the branch
            def dist_from_xnew(s_cand):
                return np.linalg.norm(s_cand.x - X_new)
            
            s_closest = min(all_solutions_found, key=dist_from_xnew)

            # 5) Assign solutions to branches (with merging logic)
            for s_other in all_solutions_found:
                if s_other == s_closest:
                    # Check if the closest solution exists in any other branch
                    for other_branch in branches:
                        if other_branch.has_solution(s_closest):
                            if branch.branch_id > other_branch.branch_id:
                                branch.open = False
                                print(f"Branch {branch.branch_id} closed due to conflict with Branch {other_branch.branch_id}")
                            else:
                                other_branch.open = False
                                print(f"Branch {branch.branch_id} closed due to conflict with Branch {other_branch.branch_id}")
                            break
                    if branch.open:
                        # Add the closest solution to the current branch
                        branch.add_solution(s_closest)
                        print(f"[Branch {branch.branch_id}] updated => {branch.current_solution}")
                    
                else:
                    # Check if the solution exists in any branch
                    is_new_solution = True
                    for other_branch in branches:
                        if other_branch.has_solution(s_other):
                            is_new_solution = False
                            break

                    # Spawn a new branch for truly new solutions
                    if is_new_solution:
                        new_branch = Branch(next_branch_id, s_other)
                        new_branch.open = True
                        new_branches_this_step.append(new_branch)
                        next_branch_id += 1
                        print(f"Spawned new Branch {new_branch.branch_id} with solution {s_other}")

            # Update the branches list with newly spawned branches
        branches.extend(new_branches_this_step)

    return branches
