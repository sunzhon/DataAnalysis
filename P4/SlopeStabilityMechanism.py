#! /usr/bin/pyenv python
#! -coding utf-8-


import math
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sympy.plotting import plot3d

# Define variables  #
l0, l1, l2, L, H= sp.symbols('l_0 l_1 l_2 L H')
θ1f,θ2f=sp.symbols(r'\theta_1^f \theta_2^f')
α = sp.symbols(r'\alpha')

# Geometric equations #
y1 = L/2+l1*sp.sin(θ1f) - l2*sp.sin(θ2f-θ1f)
y2 = L/2-l1*sp.sin(θ1f) + l2*sp.sin(θ2f-θ1f)

h=H/2+ l0 + l1*sp.cos(θ1f) +l2*sp.cos(θ2f-θ1f)

x1=y1-h*sp.tan(α)
x2=y2+h*sp.tan(α)

# Solve the function #
out=sp.solve([x1-x2],[α])
func = out[α]
print(sp.simplify(func))

# Use Lilibot configurations into this model #
slopeModel_lilibot=func.subs({l0:0.04,l1:0.07,l2:0.086,L:0.32,H:0.06}) #Lilibot
slopeModel_laikago=func.subs({l0:0.08,l1:0.41,l2:0.45,L:0.55,H:0.15}) #Laikago
print(slopeModel_lilibot)

# Plot the relationship of Lilibot between anterior exxtreme positions and slope inclination
axs1 =plot3d(slopeModel_lilibot,(θ1f,0.0/180*math.pi, 80.0/180*math.pi),(θ2f, 50.0/180*math.pi, 110.0/180*math.pi),xlabel=r'$\theta^a_1$ [degree]',ylabel=r'$\theta^a_2$ [degree]',zlabel=r'$\eta [degree]$',surface_color=[1.0,0.1,0.],show=False)
axs2 =plot3d(slopeModel_laikago,(θ1f,0.0/180*math.pi, 85.0/180*math.pi),(θ2f, 50.0/180*math.pi, 110.0/180*math.pi),xlabel=r'$\theta^a_1$ [degree]',ylabel=r'$\theta^a_2$ [degree]',zlabel=r'$\eta [degree]$',surface_color=[0.,0.1,1.0],show=False)

axs1.append(axs2[0])
#axs1.show()
#plot Lilibot and Laikago together


# substitute a particular anterior extreme position 
Max_slope=slopeModel_lilibot.evalf(subs={θ1f:80.0/180.0*math.pi,θ2f:50.0/180.0*math.pi})
Min_slope=slopeModel_lilibot.evalf(subs={θ1f:0.0/180.0*math.pi,θ2f:110.0/180.0*math.pi})
print('Lilibot max slope (80,50) slope angle:',Max_slope*180.0/math.pi)
print('Lilibot min slope (0,110) slope angle:',Min_slope*180.0/math.pi)
Max_slope=slopeModel_laikago.evalf(subs={θ1f:85.0/180.0*math.pi,θ2f:50.0/180.0*math.pi})
Min_slope=slopeModel_laikago.evalf(subs={θ1f:0.0/180.0*math.pi,θ2f:110.0/180.0*math.pi})
print('Laikago (85,50) slope angle:',Max_slope*180.0/math.pi)
print('Laikago (0.0,110) slope angle:',Min_slope*180.0/math.pi)


# get the partial derivation of the slope function
dx=sp.diff(slopeModel_lilibot,θ1f)
dy=sp.diff(slopeModel_lilibot,θ2f)
print('dx:',dx)
print('dy:',dy)


print('dx at a point:',dx.subs({θ1f:55/180*3.14,θ2f:95/180*3.14}))
print('dy at a point:',dy.subs({θ1f:55/180*3.14,θ2f:95/180*3.14}))

#Max step length

dθ1,dθ2 = sp.symbols(r'\delta\theta_1 \delta\theta_2')
S=sp.symbols('S')

Pxf=-l1*sp.sin(θ1f)+l2*sp.sin(θ2f-θ1f)
Pzf=l1*sp.cos(θ1f)+l2*sp.cos(θ2f-θ1f)

Pz=l1*sp.cos(θ1f-dθ1)+l2*sp.cos(θ2f-dθ2-θ1f+dθ1)

Px=-l1*sp.sin(θ1f-dθ1)+l2*sp.sin(θ2f-dθ2-θ1f+dθ1)

#dtheta1=sp.solve([Pzf-Pz],dθ1)
S=Pxf-Px
print('Max step length:',S)

