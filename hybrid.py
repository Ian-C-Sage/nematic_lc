from freedericksz import *

def run_it():
    cell_thickness=10e-6
    grid_points=101
    k_11=1e-10 # LC elastic constant
    k_33=1e-11 # LC elastic constant
    eps_par=15.0
    eps_perp=5.0
    voltage=0
    tilt1=0.0
    tilt2=np.pi/2
    profile=np.zeros((grid_points, 1), float)
    profile[0, :]=tilt1
    profile[grid_points-1, :]=tilt2

    z_vals=np.linspace(0, cell_thickness, grid_points) # grid_points values
    input_profile=initialise_theta(z_vals, cell_thickness, tilt1, tilt2, np.pi/10.0)
    sol_struct=root(el_wrapper, input_profile[1:-1], [tilt1, tilt2, cell_thickness, k_33, k_11, eps_par, eps_perp, voltage], method='hybr', options={'factor':0.1})
    for t_index, theta_val in enumerate(sol_struct.x):
        profile[t_index+1, 0]=theta_val

    fig, ax = plt.subplots()
    ax.plot(z_vals, profile[:, 0])
    plt.show()

if __name__=='__main__':
    run_it()
