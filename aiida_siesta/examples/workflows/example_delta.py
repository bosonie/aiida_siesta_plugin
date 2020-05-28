#!/usr/bin/env runaiida

#Not required by AiiDA
import os.path as op
import sys

#AiiDA classes and functions
from aiida.engine import submit
from aiida.orm import load_code
from aiida.orm import (Dict,Str)
from aiida_siesta.data.psf import PsfData
from aiida_siesta.workflows.DeltaTest import DeltaWorkflow

try:
    codename = sys.argv[1]
except IndexError:
    codename = 'SiestaHere@localhost'

#The code
code = load_code(codename)

#Resources
options = Dict(
    dict={
        "max_wallclock_seconds": 12*60*60,
        #'withmpi': True,
        #'account': "tcphy113c",
        #'queue_name': "DevQ",
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
    })

#The submission
#All the inputs of a Siesta calculations are listed in a dictionary
#Note the different use of options compared to ../plugins/siesta/example_first.py
elements = ["Tl"]

for element in elements:

    inputs = {
        'element': Str(element),
        'code': code,
        'options': options,
        'metadata' : {'label': 'Test delta full primo'}
    }

    process = submit(DeltaWorkflow, **inputs)
    print("Submitted workchain; ID={}".format(process.pk))
    print("For information about this workchain type: verdi process show {}".format(process.pk))
    print("For a list of running processes type: verdi process list")

#H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn
#Sb Te I Xe Cs Ba Lu Hf Ta W Re Os I Pt Au Hg Tl Pb Bi Po Rn
#"H", "He", "Li", "Be", "B", "C", "N"
#"O", "F", "Ne", "Na", "Mg", "Al", "Si"
#"P", "S", "Cl", "Ar", "K", "Ca", "Sc"
#"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni" 
#"Cu", "Zn", "Ga", "Ge", "As", "Se", "Br"
#"Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo"
#"Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
#"Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
#"Lu", "Hf", "Ta", "W", "Re", "Os", "I",
#"Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Rn"

