
# coding: utf-8

# This Notebook is a translation of Taufiq's original Bempp 2 code for Bempp 3.x.

# In[1]:

import bempp.api
import numpy as np

bempp.api.enable_console_logging()


# In[62]:

# Conductivity
k = 1E7

# Switch to dense mode
bempp.api.global_parameters.assembly.boundary_operator_assembly_type = 'dense'


# In[64]:

grid = bempp.api.shapes.sphere(1, h=0.1) # Creates a regular sphere
space = bempp.api.function_space(grid, "P", 1) # Space of piecewise constant functions


# In[65]:

# Assemble the operators
ident = bempp.api.operators.boundary.sparse.identity(space, space, space)
adlp = bempp.api.operators.boundary.laplace.adjoint_double_layer(space, space, space)

lhs = .5 * (k+1) / (k - 1) * ident + adlp


# In[66]:

# Grid functions

# Right hand sides
funs = []

for index in range(3):
    def normal_fun(point, normal, domain_index, result):
        result[0] = normal[index]
    funs.append(bempp.api.GridFunction(space, fun=normal_fun))


# In[67]:

# Solve systems

sols = []

for index in range(3):
    sols.append(bempp.api.linalg.gmres(lhs, funs[index], use_strong_form=True, tol=1E-8)[0])


# In[68]:

# Get products of solution functions with mass matrices

proj = [sol.projections() for sol in sols]


# In[69]:

# Compute the moments

moments = []

for index in range(3):
    def fun(point, normal, domain_index, result):
        result[0] = point[index]
    moments.append(bempp.api.GridFunction(space, fun=fun).coefficients)


# In[70]:

# Now compute the polarization tensor

mpt = np.empty((3, 3), dtype='float64')

for ind1 in range(3):
    for ind2 in range(3):
        mpt[ind1, ind2] = np.dot(proj[ind1], moments[ind2])
        
print(mpt)


# In[ ]:

print(mpt)


# In[ ]:




# In[ ]:



