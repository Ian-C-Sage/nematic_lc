"""Solve the continuum equation set, describing the relaxation of a twist-free
nematic layer from an initially distorted configuration to a relaxed state in
the presence of an applied field."""

from math import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

eps0=8.854e-12

def initialise_theta(z_vals, cell_thickness, tilt1, tilt2, max_tilt):
    """The routine is used to seed a smoothly tilted director configuration as a starting point for
    the numerical solution routines, to avoid the trivial solution tilt=0 (for all z) which is
    valid at all applied fields if tilt1=-tilt2"""
    thetas=np.array([tilt1+(tilt2-tilt1)*(zval/cell_thickness)+max_tilt*sin(pi*(zval/cell_thickness)) for zval in z_vals])
    return thetas

def voltage_distribution(cell_thickness, theta_vals, voltage, eps_par, eps_perp):
    """Given the LC properties and tilt profile, calculate the non-uniform
    potential distribution across a nematic layer resulting from the LC
    anisotropy. Interpolate theta values in gaps. We use the fact that the
    dielectric displacement is constant through the layer."""

    # For N grid points, there are N-1 dielectric layers. Interpolate the tilt angles.
    gap_thetas=[(theta_vals[i]+theta_vals[i+1])/2.0 for i in range(len(theta_vals)-1)]
    cos_theta=np.cos(gap_thetas) # cos(theta)
    cos_theta_sq=cos_theta*cos_theta # cos(theta)^2
    eps=eps_perp*cos_theta_sq+eps_par*(1-cos_theta_sq) # Effective relative permittivity for a tilted sub-layer
    cap=eps0*eps/(cell_thickness/(len(theta_vals)-1)) # Capacitance of each sub-layer, per unit area
    total_cap=1/np.sum(1/cap) # Capacitance of the whole stack of layers, in series
    charge_density=voltage*total_cap # Charge on the stack, per unit area, under potential of voltage
    potential_drop=charge_density/cap # voltage drop across each sub-layer
    potential=np.array(theta_vals)
    potential[0]=0.0
    for i in range(len(potential_drop)):
        potential[i+1]=potential[i]+potential_drop[i]
    potential[-1]=voltage
    return potential_drop

def voltage_distribution2(cell_thickness, theta_vals, voltage, eps_par, eps_perp):
    """Given the LC properties and tilt profile, calculate the non-uniform
    potential distribution across a nematic layer resulting from the LC
    anisotropy. Interpolate epsilon values in gaps."""

    # For N grid points, there are N-1 dielectric layers. Interpolate the permittivities.
    #gap_thetas=[(theta_vals[i]+theta_vals[i+1])/2.0 for i in range(len(theta_vals)-1)]
    cos_theta=np.cos(theta_vals) # cos(theta)
    cos_theta_sq=cos_theta*cos_theta # cos(theta)^2
    eps=eps_perp*cos_theta_sq+eps_par*(1-cos_theta_sq) # Effective relative permittivity at each grid point
    gap_eps=np.array([(eps[i]+eps[i+1])/2.0 for i in range(len(theta_vals)-1)])
    cap=eps0*gap_eps/(cell_thickness/(len(theta_vals)-1)) # Capacitance of each sub-layer, per unit area
    total_cap=1/np.sum(1/cap) # Capacitance of the whole stack of layers, in series
    charge_density=voltage*total_cap # Charge on the stack, per unit area, under potential of voltage
    potential_drop=charge_density/cap # voltage drop across each sub-layer
    potential=np.array(theta_vals)
    potential[0]=0.0
    for i in range(len(potential_drop)):
        potential[i+1]=potential[i]+potential_drop[i]
    potential[-1]=voltage
    return potential_drop

def euler_lagrange(theta_vals, cell_thickness, k_33, k_11, eps_par, eps_perp, voltage):
    """ The Euler-Lagrange equation is defined in the .wxmx file. This routine takes a vector
    argument of length grid_points and returns a vector or length grid_points-2; the wrapper
    function takes care of this and also handles the constant arguments correctly"""
    grid_spacing=cell_thickness/(len(theta_vals)-1)
    # We'll be using a finite difference approximation to the derivative terms dtheta/dz and
    # d^2theta/dz^2. The equations we need to solve are centred on theta[1] to theta[-2];
    # theta[0] and theta[-1] are constant boundary values. Approximate the derivatives by central
    # differences, and to aid this, define the theta ranges centred on i+1 and i-1.
    theta_plus=np.array(theta_vals[2:])
    theta_minus=np.array(theta_vals[:-2])
    theta_mid=np.array(theta_vals[1:-1])
    dtheta_dz=(theta_plus-theta_minus)/(2*grid_spacing)
    d2theta_dz2=(theta_plus+theta_minus-2*theta_mid)/(grid_spacing*grid_spacing)
    cos_theta=np.cos(theta_mid)
    cos_theta2=cos_theta*cos_theta
    sin_theta=np.sin(theta_mid)
    term1=((k_11-k_33)*cos_theta2+k_33)*d2theta_dz2
    term2=(k_33-k_11)*sin_theta*cos_theta*dtheta_dz*dtheta_dz
    vd=voltage_distribution(cell_thickness, theta_vals, voltage, eps_par, eps_perp)
    field=np.array([(vd[i]+vd[i+1])/(2.0*grid_spacing) for i in range(len(vd)-1)])
    term3=(eps_par-eps_perp)*eps0*field*field*cos_theta*sin_theta
    el_result=term1+term2+term3
    return el_result

def el_wrapper(theta_cen, params):
    """Wrapper routine to put the euler_lagrange equations in to the correct format,
    to be called by "solve" """
    theta_vals=np.array([0 for i in range(len(theta_cen)+2)], float)
    theta_vals[1:-1]=theta_cen
    #theta_vals[0]=tilt1
    #theta_vals[-1]=tilt2
    theta_vals[0]=params[0]
    theta_vals[-1]=params[1]
    el_result=euler_lagrange(theta_vals, params[2], params[3], params[4], params[5], params[6], params[7])
    return el_result

def run_it():
    """ Wrapper for a main routine to provide exemplary output"""
    cell_thickness=10e-6
    grid_points=101
    # Constants for E7
    k_11=11.7e-12 
    k_33=19.5e-12 
    eps_par=19.5
    eps_perp=5.17
    z_vals=np.linspace(0, cell_thickness, grid_points) # grid_points values
    tilt1=np.radians(0.0)
    tilt2=np.radians(0.0)
    init_tilt=np.pi/5.0 # Too large or small a value here results in failure
    input_profile=initialise_theta(z_vals, cell_thickness, tilt1, tilt2, init_tilt)
    #v_range=[20, 15, 10, 9, 8, 7, 6, 5, 4, 3.5, 3]
    #v_range=[5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 0.0]
    v_range=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    #v_range.reverse()
    #v_range=[2.0, 3.0]
    profile=np.zeros((grid_points, len(v_range)), float)
    profile[0, :]=tilt1
    profile[grid_points-1, :]=tilt2
    for v_index, voltage in enumerate(v_range):
        sol_struct=root(el_wrapper, input_profile[1:-1], [tilt1, tilt2, cell_thickness, k_33, k_11, eps_par, eps_perp, voltage], method='hybr')
        for t_index, theta_val in enumerate(sol_struct.x):
            profile[t_index+1, v_index]=theta_val
            # Convergence to the correct solution is a little delicate; careful
            # choice of the starting configuration solves the problem.
            # Only use the result as the starting configuration to the next
            # voltage step, if the mid-plane tilt is greater than that set by
            # initialise_theta, above
            if sol_struct.x[int((grid_points-2)/2)]>init_tilt:
                input_profile[t_index+1]=theta_val
        #print(sol_struct.x[int((grid_points-2)/2)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sol_array=sol_struct.x
    sol_array=np.insert(sol_array, 0, tilt1)
    sol_array=np.append(sol_array, tilt2)
    mid_tilt=profile[grid_points//2, :]
    for index in range(len(v_range)):
        ax1.plot(z_vals, profile[:, index], label=str(v_range[index])+'V')
    ax1.legend()
    ax1.set_ylabel('Through-cell tilt')
    ax2.plot(v_range, mid_tilt)
    ax2.set_xlabel('V')
    ax2.set_ylabel('Mid-plane tilt')
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()


if __name__=='__main__':
    run_it()
