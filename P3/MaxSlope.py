#! /usr/bin/pyenv python
#! -coding utf-8-


import math
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sympy.plotting import plot3d


l0, l1, l2, L, H= sp.symbols('l_0 l_1 l_2 L H')
θ1f,θ2f=sp.symbols(r'\theta_1^f \theta_2^f')
α = sp.symbols(r'\alpha')

y1=L/2+l1*sp.sin(θ1f) - l2*sp.sin(θ2f-θ1f)
y2=L/2-l1*sp.sin(θ1f) + l2*sp.sin(θ2f-θ1f)

h=H/2+ l0 + l1*sp.cos(θ1f) +l2*sp.cos(θ2f-θ1f)

x1=y1-h*sp.tan(α)
x2=y2+h*sp.tan(α)

out=sp.solve([x1-x2],[α])
func = out[α]
print(sp.simplify(func))

func_final=func.subs({l0:0.04,l1:0.07,l2:0.086,L:0.16,H:0.001})
print(func_final)


axs=plot3d(func_final,(θ1f,-45/180*3.14,math.pi/2-10.0/180*3.14),(θ2f, 30/180*3.14,math.pi/2+math.pi/3.0),xlabel=r'$\theta^F_1$ [rad]',ylabel=r'$\theta^F_2$ [rad]',zlabel=r'$\alpha_{max}$')


s=func_final.evalf(subs={θ1f:math.pi/2-10/180*3.14,θ2f:60.0/180*3.14})
print(s*180/math.pi)

dx=sp.diff(func_final, θ1f)
dy=sp.diff(func_final,θ2f)
print('dx:',dx)
print('dy:',dy)

print('-----------------')
s=func_final.evalf(subs={θ1f:55/180*3.14,θ2f:95.0/180*3.14})
print(s*180/math.pi)

print(dx.subs({θ1f:55/180*3.14,θ2f:95/180*3.14}))
print(dy.subs({θ1f:55/180*3.14,θ2f:95/180*3.14}))
#  max step length

dθ1,dθ2 = sp.symbols(r'\delta\theta_1 \delta\theta_2')
S=sp.symbols('S')

Pxf=-l1*sp.sin(θ1f)+l2*sp.sin(θ2f-θ1f)
Pzf=l1*sp.cos(θ1f)+l2*sp.cos(θ2f-θ1f)

Pz=l1*sp.cos(θ1f-dθ1)+l2*sp.cos(θ2f-dθ2-θ1f+dθ1)

Px=-l1*sp.sin(θ1f-dθ1)+l2*sp.sin(θ2f-dθ2-θ1f+dθ1)

#dtheta1=sp.solve([Pzf-Pz],dθ1)
S=Pxf-Px

print(S)

