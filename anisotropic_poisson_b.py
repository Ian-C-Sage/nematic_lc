# Standalone routine illustrating solution of the 2d generalised Poisson equation
# for a system including an anisotropic dielectric medium. The calculation uses
# Gunter's assymmetric method Journal of Computational Physics 209 (2005) 354–370
# and Journal of Computational Physics Volume 272, 1 September 2014, Pages 526-549
# The configuration modelled here is an IPS cell.
# See anisotropic_poisson.py for a similar model of a cell with a through-cell field.

import numpy as np
from scipy.sparse import coo_array, csr_array
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

# LC director configuration
rot_angle=45.0
azimuth=0.0

# Simulation space geometry
x_size=40.0
y_size=25.0
resolution=20

cyclic_boundaries=1

x_intervals=x_size*resolution
y_intervals=y_size*resolution

top_lc=7.5
bot_lc=17.5

x_points=int(x_intervals+1-(cyclic_boundaries & 1))
y_points=int(y_intervals+1-(cyclic_boundaries & 2)/2)

# Geometry of each electrode. Dirichlet boundaries also go here.
electrode1=[bot_lc*resolution, bot_lc*resolution+4, 5*resolution, 15*resolution]
electrode1=[int(a) for a in electrode1]
electrode2=[bot_lc*resolution, bot_lc*resolution+4, 25*resolution, 35*resolution]
electrode2=[int(a) for a in electrode2]
electrodes=[electrode1, electrode2]

fixed_potentials=[1.0, -1.0] # Potentials at each electrode

# Geometry of the Neumann boundaries.
neumann1=[0, 1, 0, x_points] # Top edge
neumann2=[y_points-1, y_points, 0, x_points] # Bottom edge
neumann=[neumann1, neumann2]

h=1e-6/resolution
h2=h*h

# These are default values in the device map, to indicate what needs
# to be done at this location. Other locations are fixed potentials.
default_v=10000 # Potential has to be calculated
default_n=20000 # Neumann boundary with zero perpendicular field

# Blank device map, initialised to default_v
v_mesh=default_v*np.ones((y_points, x_points))

# Arrays to hold varous device and LC property maps
e_par=np.ones((y_points, x_points))
e_perp=np.ones((y_points, x_points))
theta=np.zeros((y_points, x_points))
phi=np.zeros((y_points, x_points))
e11=np.ones((y_points, x_points))
e12=np.ones((y_points, x_points))
e22=np.ones((y_points, x_points))

num_electrodes=len(fixed_potentials)

# Make a "map" of the device geometry
# First insert the Dirichlet regions
for count in range(num_electrodes):
    v_mesh[electrodes[count][0]:electrodes[count][1], electrodes[count][2]:electrodes[count][3]]=fixed_potentials[count];

# Next, the Neumann boundaries
num_neumann=len(neumann)
for count in range(num_neumann):
    v_mesh[neumann[count][0]:neumann[count][1], neumann[count][2]:neumann[count][3]]=default_n

# Set the dielectric constants for the LC and substrates
#e_par[:int((y_size-lc_thickness)*resolution), :]=5.0;
#e_par[int((y_size-lc_thickness)*resolution):, :]=15.0;
#e_perp[:, :]=5.0;

e_par[:, :]=5.0
e_perp[:, :]=5.0
e_par[int(top_lc*resolution):int(bot_lc*resolution), :]=15.0

# Set the LC configuration. A dummy configuration is also set
# covering the substrate(s), but has no effect as e_par=e_perp
# in that area
theta[:, :]=0.0
theta[int(top_lc*resolution):int(bot_lc*resolution), :]=np.radians(rot_angle)
phi[:, :]=0.0


# Vectorized trigonometric functions of the Euler angles
ct=np.cos(theta)
st=np.sin(theta)
ct2=ct*ct
st2=st*st
cp=np.cos(phi)
sp=np.sin(phi)
cp2=cp*cp
sp2=sp*sp

# Components of the dielectric tensor. The reference direction
# for the Euler angles is along the y-axis, ie, "vertical"
# See euler_rotations.wxmx
eps_11=cp2*st2*e_par+(sp2+cp2*ct2)*e_perp;
eps_12=cp*ct*st*(e_perp-e_par);
eps_22=ct2*e_par+st2*e_perp;

# Now assemble the sparse system matrix
triplet_row=[]
triplet_col=[]
triplet_val=[]
y_vector=np.zeros((x_points*y_points, 1))
sys_row=-1 # A counter for the row of the matrix we are dealing with

for row in range(y_points):
    for col in range(x_points):
        sys_row=sys_row+1
        if v_mesh[row, col]<default_v: # Dirichlet boundary
            triplet_row.append(sys_row)
            triplet_col.append(sys_row)
            triplet_val.append(1.0)
            y_vector[sys_row]=v_mesh[row, col]
        elif v_mesh[row, col]==default_n: # Neumann boundary
            triplet_row.append(sys_row)
            triplet_col.append(sys_row)
            triplet_val.append(1.0)
            if col==0:
                triplet_col.append(sys_row+1)
            elif col==x_points-1:
                triplet_col.append(sys_row-1)
            elif row==0:
                triplet_col.append(sys_row+x_points)
            elif row==y_points-1:
                triplet_col.append(sys_row-x_points)
            else:
                print("Neumann boundary at illegal position:", row, col)
            triplet_row.append(sys_row)
            triplet_val.append(-1.0)
        elif v_mesh[row, col]==default_v:
            # Inner point; potential to be calculated
            # See gunter_asymmetric.wxmx
            if row==0:
                n_row=y_points-1
            else:
                n_row=row-1
            if row==y_points-1:
                s_row=0
            else:s_row=row+1
            if col==0:
                w_col=x_points-1
            else:
                w_col=col-1
            if col==x_points-1:
                e_col=0
            else:
                e_col=col+1
            n_e11=(eps_11[row, col]+eps_11[n_row, col])/2
            n_e12=(eps_12[row, col]+eps_12[n_row, col])/2
            n_e22=(eps_22[row, col]+eps_22[n_row, col])/2
            s_e11=(eps_11[row, col]+eps_11[s_row, col])/2
            s_e12=(eps_12[row, col]+eps_12[s_row, col])/2
            s_e22=(eps_22[row, col]+eps_22[s_row, col])/2
            w_e11=(eps_11[row, col]+eps_11[row, w_col])/2
            w_e12=(eps_12[row, col]+eps_12[row, w_col])/2
            w_e22=(eps_22[row, col]+eps_22[row, w_col])/2
            e_e11=(eps_11[row, col]+eps_11[row, e_col])/2
            e_e12=(eps_12[row, col]+eps_12[row, e_col])/2
            e_e22=(eps_22[row, col]+eps_22[row, e_col])/2
            # Vs
            triplet_row.append(sys_row)
            triplet_col.append((s_row)*x_points+col)
            triplet_val.append((w_e12+4*s_e22-e_e12)/(4*h2))
            # Vsw
            triplet_row.append(sys_row)
            triplet_col.append((s_row)*x_points+w_col)
            triplet_val.append((w_e12+s_e12)/(4*h2))
            # Vn
            triplet_row.append(sys_row)
            triplet_col.append((n_row)*x_points+col)
            triplet_val.append(-(w_e12-4*n_e22-e_e12)/(4*h2))
            # Vnw
            triplet_row.append(sys_row)
            triplet_col.append((n_row)*x_points+w_col)
            triplet_val.append(-(w_e12+n_e12)/(4*h2))
            # Vw
            triplet_row.append(sys_row)
            triplet_col.append((row)*x_points+w_col)
            triplet_val.append((4*w_e11+s_e12-n_e12)/(4*h2))
            # Vc
            triplet_row.append(sys_row)
            triplet_col.append(sys_row)
            triplet_val.append(-(w_e11+s_e22+n_e22+e_e11)/(h2))
            # Ve
            triplet_row.append(sys_row)
            triplet_col.append((row)*x_points+e_col)
            triplet_val.append(-(s_e12-n_e12-4*e_e11)/(4*h2))
            # Vse
            triplet_row.append(sys_row)
            triplet_col.append((s_row)*x_points+e_col)
            triplet_val.append(-(s_e12+e_e12)/(4*h2))
            # Vne
            triplet_row.append(sys_row)
            triplet_col.append((n_row)*x_points+e_col)
            triplet_val.append((n_e12+e_e12)/(4*h2))

A=coo_array((triplet_val, (triplet_row, triplet_col)), shape=(x_points*y_points, x_points*y_points)).tocsr()
X=linalg.spsolve(A, y_vector)
pot=X.reshape((y_points, x_points))

# Next find the field components by difference of
# adjacent potentials. First the x-component.
pot_l=pot[:, :x_points-1]
pot_r=pot[:, 1:]
field_x=(pot_r-pot_l)/h

# Now average adjacent rows of the field array, to
# re-centre the sample points and equaize the number
# of points along x and y. See Nagel for details.
field_xu=field_x[1: , :]
field_xd=field_x[:y_points-1, :]
field_x=(field_xu+field_xd)/2.0

# Handle the y-component in the same way
pot_u=pot[1: , :]
pot_d=pot[:y_points-1, :]
field_y=(pot_u-pot_d)/h
field_yl=field_y[:, :x_points-1]
field_yr=field_y[:, 1:]
field_y=(field_yl+field_yr)/2.0

scalar_field=np.sqrt(field_x*field_x+field_y*field_y)

# Now plot the results
# First the electric potential
fig1, ax1=plt.subplots()
ax1.imshow(pot, cmap='jet', interpolation='nearest')
# Overlay a quiver plot of the director orientation
xp=np.shape(ct)[1]
yp=np.shape(ct)[0]
# Limit the quiver plot to the LC area only
st=st[int(top_lc*resolution):int(bot_lc*resolution), :]
ct=ct[int(top_lc*resolution):int(bot_lc*resolution), :]
X,Y = np.meshgrid(np.arange(xp), np.arange(top_lc*resolution, bot_lc*resolution))
n=int(1.3*resolution)
# Plot the director as a pair of vectors, pointing in opposite directions
ax1.quiver(X[::n,::n], Y[::n, ::n], st[::n, ::n], ct[::n, ::n], angles='xy', scale_units='xy', scale=0.8/resolution, headlength=3, headwidth=1e-6)
ax1.quiver(X[::n,::n], Y[::n, ::n], -st[::n, ::n], -ct[::n, ::n], angles='xy', scale_units='xy', scale=0.8/resolution, headlength=3, headwidth=1e-6)
ax1.set_title("Potential with LC director overlay")

fig2, ax2=plt.subplots()
ax2.imshow(scalar_field, vmin=0, vmax=300000, cmap='jet', interpolation='nearest')

xp=np.shape(field_x)[1]
yp=np.shape(field_x)[0]
X,Y = np.meshgrid(np.arange(xp), np.arange(yp))
n=int(1.3*resolution)
ax2.quiver(X[::n,::n], Y[::n, ::n], -field_x[::n, ::n], -field_y[::n, ::n], angles='xy', scale_units='xy', scale=120000/resolution, headlength=3)
ax2.set_title("Scalar field with vector field overlay")

#fig3, ax3=plt.subplots()
#ax3.imshow(e_par)

plt.show()
