# Solve the continuum equation set, describing the relaxation of a twist-free
# nematic layer from an initially distorted configuration to a relaxed state in
# the presence of an applied field.

import numpy as np
from math import pi, sin, log
import matplotlib.pyplot as plt
from scipy.optimize import root

eps0=8.854e-12
cell_thickness=10e-6
voltage=30.0
grid_points=51
k_11=1e-10 # LC elastic constant
k_33=2e-10 # LC elastic constant
eps_par=15.0
eps_perp=5.0
gamma=10.0 # LC viscosity
sim_length=10.0 # Units of time
delta_t=0.0005 # Time increment
record_every=2500 # How many time increments between plots

z_vals=np.linspace(0, cell_thickness, grid_points) # grid_points values
tilt1=np.pi/18000.0
tilt2=np.pi/18000.0

# Return a smoothly varying tilt profile at the z values specified in z_vals, matching boundary
# conditions at z=0 and z=cell_thickness, and with a maximum deviation from straight line
# interpolation of max_tilt
#
# The routine is used to seed a smoothly tilted director configuration as a starting point for
# the numerical solution routines, to avoid the trivial solution tilt=0 (for all z) which is
# valid at all applied fields if tilt1=-tilt2
def initialise_theta(z_vals, cell_thickness, tilt1, tilt2, max_tilt):
    thetas=np.array([tilt1+(tilt2-tilt1)*(zval/cell_thickness)+max_tilt*sin(pi*(zval/cell_thickness)) for zval in z_vals])
    return(thetas)
                 
a=initialise_theta(z_vals, cell_thickness, tilt1, tilt2, np.pi/2.0)


# Given the LC properties and tilt profile, calculate the non-uniform
# potential distribution across a nematic layer resulting from the LC
# anisotropy. Interpolate theta values in gaps.
def voltage_distribution(d, theta_vals, voltage, eps_par, eps_perp):
    # For N grid points, there are N-1 dielectric layers. Interpolate the tilt angles.
    gap_thetas=[(theta_vals[i]+theta_vals[i+1])/2.0 for i in range(len(theta_vals)-1)]
    ct=np.cos(gap_thetas) # cos(theta)
    ct_sq=ct*ct # cos(theta)^2
    eps=eps_perp*ct_sq+eps_par*(1-ct_sq) # Effective relative permittivity for a tilted sub-layer
    cap=eps0*eps/(d/(len(theta_vals)-1)) # Capacitance of each sub-layer, per unit area
    total_cap=1/np.sum(1/cap) # Capacitance of the whole stack of layers, in series
    Q=voltage*total_cap # Charge on the stack, per unit area, under potential of voltage
    potential_drop=Q/cap # voltage drop across each sub-layer
    potential=np.array(theta_vals)
    potential[0]=0.0
    for i in range(len(potential_drop)):
        potential[i+1]=potential[i]+potential_drop[i]
    potential[-1]=voltage
    #potential_drop[0]=potential_drop[0]/2.0
    #potential_drop[-1]=potential_drop[-1]/2.0
    return potential_drop

# Given the LC properties and tilt profile, calculate the non-uniform
# potential distribution across a nematic layer resulting from the LC
# anisotropy. Interpolate epsilon values in gaps.
def voltage_distribution2(d, theta_vals, voltage, eps_par, eps_perp):
    # For N grid points, there are N-1 dielectric layers. Interpolate the permittivities.
    #gap_thetas=[(theta_vals[i]+theta_vals[i+1])/2.0 for i in range(len(theta_vals)-1)]
    ct=np.cos(theta_vals) # cos(theta)
    ct_sq=ct*ct # cos(theta)^2
    eps=eps_perp*ct_sq+eps_par*(1-ct_sq) # Effective relative permittivity at each grid point
    gap_eps=np.array([(eps[i]+eps[i+1])/2.0 for i in range(len(theta_vals)-1)])
    cap=eps0*gap_eps/(d/(len(theta_vals)-1)) # Capacitance of each sub-layer, per unit area
    total_cap=1/np.sum(1/cap) # Capacitance of the whole stack of layers, in series
    Q=voltage*total_cap # Charge on the stack, per unit area, under potential of voltage
    potential_drop=Q/cap # voltage drop across each sub-layer
    potential=np.array(theta_vals)
    potential[0]=0.0
    for i in range(len(potential_drop)):
        potential[i+1]=potential[i]+potential_drop[i]
    potential[-1]=voltage
    #potential_drop[0]=potential_drop[0]/2.0
    #potential_drop[-1]=potential_drop[-1]/2.0
    return potential_drop

# (ε_par-ε_perp)*('diff(V,z,1))^2*ε_0*cos(θ)*sin(θ)+(K*(θ[i+1]-2*θ[i]+θ[i-1]))/h^2='diff(θ,t,1).γ
def euler_lagrange(theta_vals, cell_thickness, k_33, k_11, eps_par, eps_perp, voltage):
    num_eqs=len(theta_vals)-2
    h=cell_thickness/(len(theta_vals)-1)
    theta_plus=theta_vals[2:]
    theta_minus=theta_vals[:-2]
    theta_mid=theta_vals[1:-1]
    dtheta_dz=(theta_plus-theta_minus)/(2*h)
    d2theta_dz2=(theta_plus+theta_minus-2*theta_mid)/(h*h)
    ct=np.cos(theta_mid)
    ct2=ct*ct
    st=np.sin(theta_mid)
    term1=((k_33-k_11)*ct2+k_11)*d2theta_dz2
    term2=(k_11-k_33)*st*ct*dtheta_dz*dtheta_dz
    vd=voltage_distribution2(cell_thickness, theta_vals, voltage, eps_par, eps_perp)
    field=np.array([(vd[i]+vd[i+1])/(2.0*h) for i in range(len(vd)-1)])

    term3=(eps_par-eps_perp)*eps0*field*field*ct*st
    el=term1+term2+term3
    return el

def el_wrapper(theta_cen, params):
    theta_vals=np.array([0 for i in range(len(theta_cen)+2)], float)
    theta_vals[1:-1]=theta_cen
    theta_vals[0]=tilt1
    theta_vals[-1]=tilt2
    el=euler_lagrange(theta_vals, params[2], params[3], params[4], params[5], params[6], params[7])
    return el

#v_range=[20, 15, 10, 9, 8, 7, 6, 5, 4, 3.5, 3]
v_range=[5.0, 4.5, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0]
profile=np.zeros((grid_points, len(v_range)), float)
profile[0, :]=tilt1
profile[grid_points-1, :]=tilt2
for v_index, voltage in enumerate(v_range):
    b=root(el_wrapper, a[1:-1], [tilt1, tilt2, cell_thickness, k_33, k_11, eps_par, eps_perp, voltage], method='hybr', options={'factor':0.1})
    for t_index, theta_val in enumerate(b.x):
        profile[t_index+1, v_index]=theta_val
        a[t_index+1]=theta_val
"""
v=voltage_distribution(cell_thickness, a, 1.0, 10, 5)
v2=voltage_distribution2(cell_thickness, a, 1.0, 10, 5)
"""
fig, ax = plt.subplots()
c=b.x
c=np.insert(c, 0, tilt1)
c=np.append(c, tilt2)
for index in range(len(v_range)):
    ax.plot(z_vals, profile[:, index], label=str(v_range[index])+'V')
ax.legend()
plt.show()


