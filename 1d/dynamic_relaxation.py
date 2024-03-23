# Solve the continuum equation set, describing the relaxation of a twist-free
# nematic layer from an initially distorted configuration to a uniform state in
# the absence of an applied field.
# The algorithm used is a relaxation method using sparse matrix implementation.
# The solver can be chosen to be explicit, implicit or Crank-Nicholson. The
# explicit method will be unstable at large values of delta_t or more correctly
# for values of lambda>0.5; implicit methods merely become increasingly
# inaccurate.

import numpy as np
from math import pi, sin, log
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve

solver=''
#solver='explicit'
#solver='implicit'
solver='crank-nicholson'

cell_thickness=10e-6
grid_points=101
K=1e-10 # LC elestic constant
gamma=10.0 # LC viscosity
sim_length=10.0 # Units of time
delta_t=0.0005 # Time increment
record_every=2500 # How many time increments between plots

# Define solvers using three common algorithms. See the main documentation for a
# discussion of these. All are based on sparse matrices; these are first assembled
# in row/column/value triples, then converted to the compressed format used by sparse
# linear algebra libraries.
def explicit_solver(start_config, lambda_param, sim_length, delta_t, record_every):
    num_points=len(start_config)
    # Assemble the B matrix in sparse triplet format
    # The main diagonal
    row_index=[i for i in range(1, num_points-1)]
    col_index=[i for i in range(1, num_points-1)]
    cell_vals=[1-2*lambda_param for i in range(1, num_points-1)]
    # The upper diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(2, num_points)]
    cell_vals=cell_vals+[lambda_param for i in range(1, num_points-1)]
    # The lower diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(0, num_points-2)]
    cell_vals=cell_vals+[lambda_param for i in range(1, num_points-1)]
    # The boundary values
    row_index=row_index+[0, num_points-1]
    col_index=col_index+[0, num_points-1]
    cell_vals=cell_vals+[1.0, 1.0]
    #print(row_index)
    #print(col_index)
    #print(cell_vals)
    # Convert the triplet representation to compressed format
    B=coo_array((cell_vals, (row_index, col_index)), shape=(num_points, num_points)).tocsc()
    #print(B.toarray())
    num_outputs=int(1+sim_length/(delta_t*record_every))
    #results=zeros(num_points, num_outputs)
    results = [[0]*num_points for i in range(num_outputs)]
    results[0]=start_config.copy()
    inp=start_config.copy()
    output_ticker=0
    output_counter=1
    for count in range(num_steps):
        outp=B@inp
        #print(outp)
        output_ticker=output_ticker+1
        if output_ticker==record_every:
            output_ticker=0
            results[output_counter]=outp.copy()
            output_counter=output_counter+1
        inp=outp.copy()
    return results
    
            
def implicit_solver(start_config, lambda_param, sim_length, delta_t, record_every):
    num_points=len(start_config)
    # Assemble the A matrix in sparse format
    # The main diagonal
    row_index=[i for i in range(1, num_points-1)]
    col_index=[i for i in range(1, num_points-1)]
    cell_vals=[1+2*lambda_param for i in range(1, num_points-1)]
    # The upper diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(2, num_points)]
    cell_vals=cell_vals+[-lambda_param for i in range(1, num_points-1)]
    # The lower diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(0, num_points-2)]
    cell_vals=cell_vals+[-lambda_param for i in range(1, num_points-1)]
    # The boundary values
    row_index=row_index+[0, num_points-1]
    col_index=col_index+[0, num_points-1]
    cell_vals=cell_vals+[1.0, 1.0]
    #print(row_index)
    #print(col_index)
    #print(cell_vals)
    A=coo_array((cell_vals, (row_index, col_index)), shape=(num_points, num_points)).tocsc()
    #print(A.toarray())
    num_outputs=int(1+sim_length/(delta_t*record_every))
    #results=zeros(num_points, num_outputs)
    results = [[0]*num_points for i in range(num_outputs)]
    results[0]=start_config.copy()
    inp=start_config.copy()
    output_ticker=0
    output_counter=1
    for count in range(num_steps):
        outp=spsolve(A, inp)
        #print(outp)
        output_ticker=output_ticker+1
        if output_ticker==record_every:
            output_ticker=0
            results[output_counter]=outp.copy()
            output_counter=output_counter+1
        inp=outp.copy()
    return results

def cn_solver(start_config, lambda_param, sim_length, delta_t, record_every):
    num_points=len(start_config)
    # Assemble the A matrix in sparse format
    # The main diagonal
    row_index=[i for i in range(1, num_points-1)]
    col_index=[i for i in range(1, num_points-1)]
    cell_vals=[2+2*lambda_param for i in range(1, num_points-1)]
    # The upper diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(2, num_points)]
    cell_vals=cell_vals+[-lambda_param for i in range(1, num_points-1)]
    # The lower diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(0, num_points-2)]
    cell_vals=cell_vals+[-lambda_param for i in range(1, num_points-1)]
    # The boundary values
    row_index=row_index+[0, num_points-1]
    col_index=col_index+[0, num_points-1]
    cell_vals=cell_vals+[1.0, 1.0]
    #print(row_index)
    #print(col_index)
    #print(cell_vals)
    A=coo_array((cell_vals, (row_index, col_index)), shape=(num_points, num_points)).tocsc()
    #print(A.toarray())
    # Assemble the B matrix in sparse format
    # The main diagonal
    row_index=[i for i in range(1, num_points-1)]
    col_index=[i for i in range(1, num_points-1)]
    cell_vals=[2-2*lambda_param for i in range(1, num_points-1)]
    # The upper diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(2, num_points)]
    cell_vals=cell_vals+[lambda_param for i in range(1, num_points-1)]
    # The lower diagonal
    row_index=row_index+[i for i in range(1, num_points-1)]
    col_index=col_index+[i for i in range(0, num_points-2)]
    cell_vals=cell_vals+[lambda_param for i in range(1, num_points-1)]
    # The boundary values
    row_index=row_index+[0, num_points-1]
    col_index=col_index+[0, num_points-1]
    cell_vals=cell_vals+[1.0, 1.0]
    #print(row_index)
    #print(col_index)
    #print(cell_vals)
    B=coo_array((cell_vals, (row_index, col_index)), shape=(num_points, num_points)).tocsc()
    #print(B.toarray())
    num_outputs=int(1+sim_length/(delta_t*record_every))
    #results=zeros(num_points, num_outputs)
    results = [[0]*num_points for i in range(num_outputs)]
    results[0]=start_config.copy()
    inp=start_config.copy()
    output_ticker=0
    output_counter=1
    for count in range(num_steps):
        outp=spsolve(A, B@inp)
        #print(outp)
        output_ticker=output_ticker+1
        if output_ticker==record_every:
            output_ticker=0
            results[output_counter]=outp.copy()
            output_counter=output_counter+1
        inp=outp.copy()
    return results


z_vals=np.linspace(0, cell_thickness, grid_points)
#print(z_vals)
tilt1=0.0
tilt2=0.0
start_theta=[tilt1+(tilt2-tilt1)*xval/cell_thickness+sin(pi*xval/cell_thickness) for xval in z_vals]
#print(start_theta)
#fig, ax = plt.subplots()
#ax.plot(z_vals, start_theta)
#plt.show()
h=cell_thickness/(grid_points-1)

lambda_param=(K*delta_t)/(h*h*gamma)
print("lambda=", lambda_param)
num_outputs=int(1+sim_length/(delta_t*record_every))
num_steps=int(sim_length/delta_t)
t_vals=np.linspace(0, sim_length, num_outputs)
print(t_vals)
if solver=='crank-nicholson':
    print("Using Crank-Nicholson solver\n")
    result=cn_solver(start_theta, lambda_param, sim_length, delta_t, record_every)
elif solver=='implicit':
    print("Using implicit solver\n")
    result=implicit_solver(start_theta, lambda_param, sim_length, delta_t, record_every)
elif solver=='explicit':
    print("Using explicit solver\n")
    result=explicit_solver(start_theta, lambda_param, sim_length, delta_t, record_every)
else:
    print("No valid solver specified\n")
ts=[]
for i in range(len(t_vals)):
    ts.append(log(result[i][int((grid_points+1)/2)]))
num_plots=len(result)
fig, (ax1, ax2) = plt.subplots(1, 2)
for i in range(num_plots):
    ax1.plot(z_vals, result[i], label="t="+str(t_vals[i]))
ax1.legend()
ax1.set_xlabel("z")
ax1.set_ylabel("Theta")
ax2.plot(t_vals, ts)
ax2.set_xlabel("t")
ax2.set_ylabel("log(Theta_max)")
plt.show()
