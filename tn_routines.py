"""Solve the continuum equation set, describing the relaxation of a twisted
nematic layer from an initially distorted configuration to a relaxed state in
the presence of an applied field."""

from math import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

eps0=8.854e-12

def initialise_theta(grid_points, tilt1, tilt2, max_tilt):
    """The routine is used to seed a smoothly tilted director configuration as a starting point for
    the numerical solution routines, to avoid the trivial solution tilt=0 (for all z) which is
    valid at all applied fields if tilt1=tilt2=0"""
    thetas=np.array([tilt1+(tilt2-tilt1)*(index/(grid_points-1))+max_tilt*sin(pi*(index/(grid_points-1))) for index in range(grid_points)])
    return thetas

def initialise_phi(z_vals, twist):
    """Initialise the phi values to a uniform ramp from 0 to twist"""
    phis=np.linspace(0.0, twist, z_vals)
    return phis

def voltage_distribution(cell_thickness, theta_vals, voltage, eps_par, eps_perp):
    """Given the LC properties and tilt profile, calculate the non-uniform
    potential distribution across a nematic layer resulting from the LC
    anisotropy. Interpolate theta values in gaps."""

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


def euler_lagrange(euler_vals, cell_thickness, k_33, k_22, k_11, eps_par, eps_perp, voltage, inv_P):
    """ The Euler-Lagrange equations are defined in the .wxmx file. This routine takes a vector
    argument of length 2*grid_points and returns a vector or length 2*grid_points-4; the wrapper
    function takes care of this and also handles the constant arguments correctly"""
    
    # We'll be using a finite difference approximation to the derivative terms dtheta/dz and
    # d^2theta/dz^2. The equations we need to solve are centred on theta[1] to theta[-2];
    # theta[0] and theta[-1] are constant boundary values. Approximate the derivatives by central
    # differences, and to aid this, define the theta ranges (theta_plus, theta_minus) centred on
    # i+1 and i-1 respectively.
    base_length=int(len(euler_vals)/2)
    theta_vals=euler_vals[:base_length]
    theta_plus=np.array(theta_vals[2:])
    theta_minus=np.array(theta_vals[:-2])
    theta_mid=np.array(theta_vals[1:-1])
    grid_spacing=cell_thickness/(len(theta_vals)-1)
    dtheta_dz=(theta_plus-theta_minus)/(2*grid_spacing)
    d2theta_dz2=(theta_plus+theta_minus-2*theta_mid)/(grid_spacing*grid_spacing)
    
    phi_vals=euler_vals[base_length:]
    phi_plus=np.array(phi_vals[2:])
    phi_minus=np.array(phi_vals[:-2])
    phi_mid=np.array(phi_vals[1:-1])
    dphi_dz=(phi_plus-phi_minus)/(2*grid_spacing)
    d2phi_dz2=(phi_plus+phi_minus-2*phi_mid)/(grid_spacing*grid_spacing)

    # These are vectorized functions, which operate on the whole list of theta values
    cos_theta=np.cos(theta_mid)
    cos_theta_sq=cos_theta*cos_theta
    cos_theta_cu=cos_theta_sq*cos_theta
    cos_theta_qd=cos_theta_sq*cos_theta_sq
    sin_theta=np.sin(theta_mid)
    
    vd=voltage_distribution(cell_thickness, theta_vals, voltage, eps_par, eps_perp)
    field=np.array([(vd[i]+vd[i+1])/(2.0*grid_spacing) for i in range(len(vd)-1)])
    
    el1_term1=-2*k_33*cos_theta_cu*sin_theta*dphi_dz*dphi_dz
    el1_term2=2*k_22*cos_theta_cu*sin_theta*dphi_dz*dphi_dz
    el1_term3=k_33*cos_theta*sin_theta*dphi_dz*dphi_dz
    el1_term4=-4*np.pi*k_22*cos_theta*sin_theta*dphi_dz*inv_P
    el1_term5=-k_33*cos_theta_sq*d2theta_dz2
    el1_term6=k_11*cos_theta_sq*d2theta_dz2
    el1_term7=k_33*d2theta_dz2
    el1_term8=k_33*cos_theta*sin_theta*dtheta_dz*dtheta_dz
    el1_term9=-k_11*cos_theta*sin_theta*dtheta_dz*dtheta_dz
    el1_term10=-eps_perp*field*field*eps0*cos_theta*sin_theta
    el1_term11=eps_par*field*field*eps0*cos_theta*sin_theta
    el1_result=el1_term1+el1_term2+el1_term3+el1_term4+el1_term5+el1_term6+el1_term7+el1_term8+el1_term9+el1_term10+el1_term11

    el2_term1=-k_33*cos_theta_qd*d2phi_dz2
    el2_term2=k_22*cos_theta_qd*d2phi_dz2
    el2_term3=k_33*cos_theta_sq*d2phi_dz2
    el2_term4=4*k_33*cos_theta_cu*sin_theta*dtheta_dz*dphi_dz
    el2_term5=-4*k_22*cos_theta_cu*sin_theta*dtheta_dz*dphi_dz
    el2_term6=-2*k_33*cos_theta*sin_theta*dtheta_dz*dphi_dz
    el2_term7=4*np.pi*k_22*cos_theta*sin_theta*dtheta_dz*inv_P
    el2_result=el2_term1+el2_term2+el2_term3+el2_term4+el2_term5+el2_term6+el2_term7

    el_result=np.concatenate((el1_result, el2_result)) # String together the theta and phi values
    return el_result

def el_wrapper(euler_cen, params):
    """Wrapper routine to put the euler_lagrange equations into the correct format,
    to be called by "solve"
    params=[tilt1, tilt2, twist, cell_thickness, k_33, k_22, k_11, eps_par, eps_perp, voltage, inv_P]"""
    base_length=int(len(euler_cen)/2)
    theta_vals=np.array([0 for i in range(base_length+2)], float)
    theta_vals[1:-1]=euler_cen[:base_length]
    theta_vals[0]=params[0]
    theta_vals[-1]=params[1]

    phi_vals=np.array([0 for i in range(base_length+2)], float)
    phi_vals[1:-1]=euler_cen[base_length:]
    phi_vals[0]=0.0
    phi_vals[-1]=params[2]
    euler_vals=np.concatenate((theta_vals, phi_vals))
    el_result=euler_lagrange(euler_vals, params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10])
    return el_result

def demo():
    """ Wrapper for a main routine to provide exemplary output"""
    grid_points=101
    twist=np.pi/2 # TN cell
    tilt1=pi/90.0 # 2 degree pretilt
    tilt2=pi/90.0
    max_tilt=np.pi/36
    cell_thickness=10e-6
    z_vals=np.linspace(0, cell_thickness, grid_points)
    k_11=11.7e-12 # LC elastic constants
    k_22=8.8e-12
    k_33=19.5e-12
    eps_par=19.5 # LC permittivities
    eps_perp=5.17
    inv_P=1/(300e-6) # Inverse chiral pitch length

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
    demo()
