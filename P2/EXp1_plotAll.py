#!/usr/bin/pyenv python
import Exp1_ANC
Exp1_ANC.fig.set_size_inches(6,6,forward=True)
Exp1_ANC.fig.savefig("ANC.png",bbox_inches='tight')

import Exp1_ACITerm
Exp1_ACITerm.fig.set_size_inches(4,8,forward=True)
Exp1_ACITerm.fig.savefig("ACITerm.png",bbox_inches='tight')

import Exp1_attitude
Exp1_attitude.fig.set_size_inches(4,8,forward=True)
Exp1_attitude.fig.savefig("attitude.png",bbox_inches='tight')

import Exp1_CPG
Exp1_CPG.fig.set_size_inches(12,8,forward=True)
Exp1_CPG.fig.savefig("CPG.png",bbox_inches='tight')

import Exp1_gait
#Exp1_gait.fig.set_size_inches(8,2,forward=True)
Exp1_gait.fig.savefig("gait.png",bbox_inches='tight')

import Exp1_GRF
Exp1_GRF.fig.set_size_inches(6,8,forward=True)
Exp1_GRF.fig.savefig("GRF.png",bbox_inches='tight')

import Exp1_SAdaptation
Exp1_SAdaptation.fig.set_size_inches(6,8,forward=True)
Exp1_SAdaptation.fig.savefig("SAdaptation.png",bbox_inches='tight')

import Exp1_SFTerm
Exp1_SFTerm.fig.set_size_inches(6,8,forward=True)
Exp1_SFTerm.fig.savefig("SFTerm.png",bbox_inches='tight')


