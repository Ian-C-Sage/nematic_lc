from tn_routines import *

def run_it():
    """ Wrapper for a main routine to provide exemplary output"""
    grid_points=51
    twist=np.radians(180)
    tilt1=pi/90.0 # 2 degree pretilt
    tilt2=pi/90.0
    max_tilt=np.pi/36
    cell_thickness=10e-6
    z_vals=np.linspace(0, cell_thickness, grid_points)
    voltage=6.0
    k_11=11.7e-12 # LC elastic constants
    k_22=8.8e-12
    k_33=19.5e-12
    eps_par=19.5 # LC permittivities
    eps_perp=5.17
    inv_P=twist/(2*np.pi*cell_thickness) # Inverse chiral pitch length

    # Define a starting configuration with the correct twist and slight
    # tilt
    a=initialise_phi(grid_points, twist)
    b=initialise_theta(grid_points, tilt1, tilt2, max_tilt)
    c=np.concatenate((b, a))
    # The root function accepts a "guess" configuration which excludes
    # the fixed boundaries - and returns a solution of the same length.
    # Here is the starting guess.
    euler_cen=np.concatenate((b[1:-1], a[1:-1]))
    v_range=[a/100 for a in range(0, 501)] # 0V to 5V in small steps
    profile=np.zeros((2*grid_points, len(v_range)), float) # An array to put the answers in

    for v_index, voltage in enumerate(v_range):
        sol_struct=root(el_wrapper, euler_cen, [tilt1, tilt2, twist, cell_thickness, k_33, k_22, k_11, eps_par, eps_perp, voltage, inv_P], method='hybr')
        # Put the answer into its array, and update the "guess" for each step, to
        # the solution found at the previous voltage.
        for t_index, theta_val in enumerate(sol_struct.x[:int(len(euler_cen)/2)]):
            profile[t_index+1, v_index]=theta_val
            euler_cen[t_index]=theta_val
        for t_index, phi_val in enumerate(sol_struct.x[int(len(euler_cen)/2):]):
            profile[t_index+3+int(len(euler_cen)/2), v_index]=phi_val
            euler_cen[t_index+int(len(euler_cen)/2)]=phi_val
    # Put the boundary values into the solution array
    rows=np.shape(profile)[0]
    profile[0, :]=tilt1
    profile[int(rows/2)-1, :]=tilt2
    profile[int(rows/2), :]=0.0
    profile[-1, :]=twist
    # And plot a few things
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5))
    col_list=[]
    for index, value in enumerate(v_range):
        if 2*value==int(2*value):
            col_list.append(index)
    for col in col_list:
        ax1.plot(z_vals, profile[:int(rows/2), col], label=str(v_range[col])+'V')
        ax2.plot(z_vals, profile[int(rows/2):, col])
    ax3.plot(v_range[:], profile[int(rows/4), :])
    ax1.set_ylabel('Through-cell tilt')
    ax2.set_ylabel('Through-cell twist')
    ax3.set_xlabel('V')
    ax3.set_ylabel('Mid-plane tilt')
    ax1.legend()
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    plt.show()

if __name__=='__main__':
    run_it()
