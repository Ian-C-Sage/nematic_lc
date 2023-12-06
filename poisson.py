# Implementation of James R Nagel, IEEE Antennas and Propagation Magazine,
# Vol. 56, No.4, August 2014 or more exactly to 2012 web version, figure 5
# Dielectric medium is uniform!

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

grid_x=500
grid_y=500
#grid_x=8
#grid_y=8

top_electrode_y=200
bot_electrode_y=300
electrode_width=250

electrode_left=int((grid_x-electrode_width)/2)
electrode_right=int((grid_x+electrode_width)/2)

# Make a "map" of the device configuration
# 20 - inner point; potential to be calculated
# 0-9 - Dirichlet boundary
# 10 - Neumann boundary
dev_map=20*np.ones((grid_y, grid_x), int)
dev_map[0, :]=2
dev_map[top_electrode_y, electrode_left:electrode_right+1]=0
dev_map[grid_y-1, :]=2
dev_map[bot_electrode_y, electrode_left:electrode_right+1]=1
dev_map[:, 0]=10
dev_map[:, grid_x-1]=10

dirichlet_potentials=[1.0, -1.0, 0.0]

matrix_rank=grid_x*grid_y

coo_row=[]
coo_col=[]
coo_val=[]
b=[]

# Assemble the sparse system matrix, initially in triplet format
for dev_row in range(grid_y):
    for dev_col in range(grid_x):
        matrix_row=dev_row*grid_x+dev_col
        code=dev_map[dev_row, dev_col]
        if code in range(10):
            coo_row.append(matrix_row)
            coo_col.append(matrix_row)
            coo_val.append(1.0)
            b.append(dirichlet_potentials[code])
        elif code==20:
            coo_row.append(matrix_row)
            coo_col.append(matrix_row)
            coo_val.append(-4.0)
            coo_row.append(matrix_row)
            coo_col.append(matrix_row+1)
            coo_val.append(1.0)
            coo_row.append(matrix_row)
            coo_col.append(matrix_row-1)
            coo_val.append(1.0)
            coo_row.append(matrix_row)
            coo_col.append(matrix_row+grid_x)
            coo_val.append(1.0)
            coo_row.append(matrix_row)
            coo_col.append(matrix_row-grid_x)
            coo_val.append(1.0)
            b.append(0.0)
        elif code==10:
            coo_row.append(matrix_row)
            coo_col.append(matrix_row)
            coo_val.append(1.0)
            if dev_col==0:
                coo_row.append(matrix_row)
                coo_col.append(matrix_row+1)
                coo_val.append(-1.0)
            else:
                coo_row.append(matrix_row)
                coo_col.append(matrix_row-1)
                coo_val.append(-1.0)
            b.append(0.0)
# Convert the sparse matrix to csr format, suitable for linear
# algebra operations.
A=sp.coo_array((coo_val, (coo_row, coo_col))).tocsr()

# Solve for the vector of potentials            
X=linalg.spsolve(A, b)
# Reshape the vector into the device configuration
pot=np.reshape(X, (grid_y, grid_x))

# Next find the field components by difference of
# adjacent potentials. First the x-component.
pot_l=pot[:, :grid_x-1]
pot_r=pot[:, 1:]
field_x=pot_r-pot_l

# Now average adjacent rows of the field array, to
# re-centre the sample points and equaize the number
# of points along x and y. See Nagel for details.
field_xu=field_x[1: , :]
field_xd=field_x[:grid_y-1, :]
field_x=(field_xu+field_xd)/2.0

# Handle the y-component in the same way
pot_u=pot[1: , :]
pot_d=pot[:grid_y-1, :]
field_y=pot_u-pot_d
field_yl=field_y[:, :grid_x-1]
field_yr=field_y[:, 1:]
field_y=(field_yl+field_yr)/2.0

scalar_field=np.sqrt(field_x*field_x+field_y*field_y)
fig1, ax1=plt.subplots()
ax1.imshow(pot, cmap='jet', interpolation='nearest')
fig2, ax2=plt.subplots()
ax2.imshow(scalar_field, cmap='jet', vmin=0, vmax=0.035, interpolation='nearest')
X,Y = np.meshgrid(np.arange(500), np.arange(500))
n=20
ax2.quiver(X[::n,::n], Y[::n, ::n], -field_x[::n, ::n], -field_y[::n, ::n], angles='xy', scale_units='xy', scale=0.0015, headlength=3)

plt.show()
