import numpy as np 
from dynamical_system import F_k, full_system
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display

#Simulate equations under differnet parameter sets through discretized integration
def simulate(
        kappa, gamma, S_hat, delta_hat, 
        Delta_0_func, eta_0_func, Delta_E, R_func, k,
        zb, zt, N, T_final,
        L, oc,
        solver = 'BDF'
    ):
    z, dz = np.linspace(zb, zt, N, retstep=True)  # Compute spatial grid

    #Apply functions to it
    DeltaE = Delta_E(z)
    Fk = np.vectorize(F_k)(R_func, z, k)

    #Compute initial conditions on grid
    Delta_0 = np.vectorize(Delta_0_func)(z).astype(float) if callable(Delta_0_func) else Delta_0_func
    eta_0 = np.vectorize(eta_0_func)(z).astype(complex) if callable(eta_0_func) else eta_0_func

    # Initial condition vector
    y0 = np.concatenate([Delta_0, eta_0])

    # Time span for the simulation
    t_span = (0, T_final)

    # Solve the system
    constant_params = (S_hat, delta_hat, kappa, gamma)
    function_params = (L, oc, DeltaE, Fk)
    solution = solve_ivp(lambda t, y: full_system(y, constant_params, function_params, dz), t_span, y0,  method=solver)

    sol_t = solution.t
    sol_Delta, sol_eta = np.split(solution.y, 2)
    return sol_t, sol_Delta, sol_eta

def generate_plots(
        kappa, gamma, S_hat, delta_hat, 
        Delta_0_func, eta_0_func, Delta_E, R_func, k,
        zb, zt, N, T_final,
        L, oc,
        zg, fig, axs, c1 = None, c2 = None,
        solver = 'BDF', levels = 10
    ):

    sol_t, sol_Delta, sol_eta = simulate(kappa, gamma, S_hat, delta_hat, Delta_0_func, eta_0_func, Delta_E, R_func, k, zb, zt, N, T_final, L, oc, solver)

    t_grid, zg_grid = np.meshgrid(sol_t, zg)

    # Plotting sol_Delta as a contour plot
    c1 = axs[0, 0].contourf(t_grid, zg_grid, sol_Delta.real, levels, cmap='viridis')
    axs[0, 0].set_title('Contour of sol_Delta')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Layer')

    # Plotting amplitude of sol_eta as a contour plot
    c2 = axs[0, 1].contourf(t_grid, zg_grid, np.abs(sol_eta), levels, cmap='plasma')
    axs[0, 1].set_title('Contour of amplitude of eta')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Layer')

    # Plotting phase of sol_eta
    c3 = axs[1, 0].contourf(t_grid, zg_grid, np.angle(sol_eta), levels, cmap='viridis')
    axs[1, 0].set_title('Contour of phase of eta')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Layer')


    # Selecting 5 evenly spaced layers
    num_layers = sol_Delta.shape[0]
    representative_layers = np.linspace(0, num_layers - 1, 5, dtype=int)

    # Plotting representative layers for sol_Delta
    for i in representative_layers:
        axs[1, 1].plot(sol_t, sol_Delta[i, :].T.real, label=f'Layer {zg[i]:.2f}')
    axs[1, 1].set_title('sol_Delta over Time for Selected Layers')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('sol_Delta')
    axs[1, 1].legend()

    # Set y-axis limits for non-contour plots
    axs[1, 1].set_ylim(0, 1)

    
    fig.suptitle(f'Eta, Delta for kappa={kappa}, gamma={gamma}, S_hat={S_hat}, delta_hat={delta_hat}')

    return axs, c1, c2, c3, sol_Delta, sol_eta


def generate_and_show_plots(
    kappa, gamma, S_hat, delta_hat, Delta0_func, eta0_func, Delta_E, R, k, zb, zt, N, T_final, L, oc, zg, solver='RK45', levels = 30, save=False
):
    '''
    Generates and shows plots

    :param kappa: forcing parameter
    :type kappa: float
    :param gamma: time scale paramter
    :type gamma: float
    :param S_hat: S_hat parameter
    :type S_hat: float
    :param delta_hat: parameter
    :type delta_hat: float
    :param Delta0_func: function representing the initial value of Delta
    :type Delta0_func: function
    :param eta0_func: function representing the initial value of eta
    :type eta0_func: function
    :param Delta_E: equilibrium value of Delta
    :type Delta_E: function
    :param R: R function
    :type R: function
    :param k: wavenumber
    :type k: int
    :param zb: bottom of vortex
    :type zb: float
    :param zt: top of vortex
    :type zt: float
    :param N: discretization number
    :type N: int
    :param T_final: final time of simulation
    :type T_final: int
    :param L: linear operator
    :type L: np.array(float)
    :param oc: omega_c
    :type oc: np.array(float)
    :param zg: z-grid
    :type zg: np.array(float)
    :param solver: integration solver to use, default='RK45'
    :type solver: string, optional
    :param levels: number of contour levels, default=30
    :type levels: int, optional
    :param save: whether to save the plot, default=True
    :type save: boolean, optional
    :returns: sol_Delta, sol_eta
    :rtype: np.array(float), np.array(float)
    '''
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    c1 = c2 = c3 = None

    #'RK45'
    axs, c1, c2, c3, sol_Delta, sol_eta = generate_plots(kappa, gamma, S_hat, delta_hat, Delta0_func, eta0_func, Delta_E, R, k, zb, zt, N, T_final, L, oc, zg, fig, axs, solver='RK45', levels = 30)

    fig.colorbar(c1, ax=axs[0, 0], cmap='viridis')
    fig.colorbar(c2, ax=axs[0, 1], cmap='plasma')
    fig.colorbar(c3, ax=axs[1, 0], cmap='viridis')

    # Adjust layout
    plt.tight_layout()

    if save:
        plt.savefig(f'../figure_cache/static/simulation_kappa={kappa}_gamma={gamma}_S_hat={S_hat}_delta_hat={delta_hat}_N={N}.png')

    # Show the plot
    plt.show()



    return sol_Delta, sol_eta


def generate_animated_plots(
    animation_values, param_to_animate, kappa, gamma, S_hat, delta_hat, Delta0, eta0, Delta_E, R, k, zb, zt, N, T_final, L, oc, zg, num_frames = 50, solver='RK45', levels = 30, save=False
):
    '''
    Generates and shows plots

    :param animation_values: values of the parameter to animate
    :type animation_values: np.array(float)
    :param param_to_animate: which parameter we are animating
    :type param_to_animate: string
    :param kappa: forcing parameter
    :type kappa: float
    :param gamma: time scale paramter
    :type gamma: float
    :param S_hat: S_hat parameter
    :type S_hat: float
    :param delta_hat: parameter
    :type delta_hat: float
    :param Delta0: function representing the initial value of Delta
    :type Delta0: function
    :param eta0: function representing the initial value of eta
    :type eta0: function
    :param Delta_E: equilibrium value of Delta
    :type Delta_E: function
    :param R: R function
    :type R: function
    :param k: wavenumber
    :type k: int
    :param zb: bottom of vortex
    :type zb: float
    :param zt: top of vortex
    :type zt: float
    :param N: discretization number
    :type N: int
    :param T_final: final time of simulation
    :type T_final: int
    :param L: linear operator
    :type L: np.array(float)
    :param oc: omega_c
    :type oc: np.array(float)
    :param zg: z-grid
    :type zg: np.array(float)
    :param num_frames: number of animation frames, default = 50
    :type num_frames: int
    :param solver: integration solver to use, default='RK45'
    :type solver: string, optional
    :param levels: number of contour levels, default=30
    :type levels: int, optional
    :param save: whether to save the plot, default=True
    :type save: boolean, optional
    :returns: sol_Delta, sol_eta
    :rtype: np.array(float), np.array(float)
    '''
    global axs, c1, c2, c3
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Initialize the lines for Delta and abs of Eta
    c1 = c2 = c3 = None

    fig.colorbar(c1, ax=axs[0, 0], cmap='viridis')
    fig.colorbar(c2, ax=axs[0, 1], cmap='plasma')
    fig.colorbar(c3, ax=axs[1, 0], cmap='viridis')


    def init():
        for ax in axs.flat:
            ax.clear()
        return axs.flat

    # Function to update the animation
    def update(frame, param_to_animate, kappa, gamma, S_hat, delta_hat, Delta0, eta0, Delta_E, R, k, zb, zt, N, T_final, L, oc, zg, fig, levels):
        global axs, c1, c2, c3  

        if param_to_animate == 'kappa':
            kappa = animation_values[frame % len(animation_values)]
        elif param_to_animate == 'gamma':
            gamma = animation_values[frame % len(animation_values)]
        elif param_to_animate == 'S_hat':
            S_hat = animation_values[frame % len(animation_values)]
        elif param_to_animate == 'delta_hat':
            delta_hat = animation_values[frame % len(animation_values)]
        elif param_to_animate == 'Delta0':
            Delta0 = animation_values[frame % len(animation_values)]
        elif param_to_animate == 'eta0':
            eta0 = animation_values[frame % len(animation_values)]

        for ax in axs.flat:
            ax.clear()

        axs, c1, c2, c3, _, _ = generate_plots(kappa, gamma, S_hat, delta_hat, Delta0, eta0, Delta_E, R, k, zb, zt, N, T_final, L, oc, zg, fig, axs, c1=c1, c2=c2, solver='RK45', levels = levels)

        return axs.flat

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, 
        fargs=(param_to_animate, kappa, gamma, S_hat, delta_hat, Delta0, eta0, Delta_E, R, k, zb, zt, N, T_final, L, oc, zg, fig, levels), blit=False, repeat=True)

    # Display the animation
    html_anim = HTML(ani.to_jshtml())
    display(html_anim)

    if save:
        ani.save(filename=f'../figure_cache/animated/MultiLayer Delta and AbsEta for varying {param_to_animate}.html', writer="html")

    return html_anim

   
