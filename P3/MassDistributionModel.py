#! /usr/bin/pyenv python
#! -coding utf-8-


import math
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sympy.plotting import plot3d


T, S, L,t= sp.symbols('T S L t')

func=(L/2+2*S/T*t)/(L/2-2*S/T*t)
func1=func.subs({T:0.5,S:0.02,L:0.16})
sp.plotting.plot(func1,(t,0,0.25))

out=sp.integrate(func,(t, 0, T/2))
out1=out/(T/2)

out1=out.subs({L:0.16})
print(out1)

axs=plot3d(out1,(T,0.5,4.5),(S, 0.005,0.025))



