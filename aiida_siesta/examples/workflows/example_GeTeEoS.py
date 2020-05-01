#!/usr/bin/env runaiida

#Not required by AiiDA
import os.path as op
import sys
import pymatgen as mg

#AiiDA classes and functions
from aiida.engine import submit
from aiida.orm import load_code
from aiida.orm import (Float, Dict, StructureData, KpointsData)
from aiida_siesta.data.psf import PsfData
from aiida_siesta.workflows.GeTeEoS import EqOfStateGeTe
from aiida_siesta.data.psml import get_pseudos_from_structure

# This example shows the use of the IsotropicEosFast
# Requires a working aiida profile and the set up of
# a code (it submits the WorkChain to the daemon).
# To run it: runaiida example_eos.py codename
# The inputs are the same of ../plugins/siesta/example_first.py
# with the addition of an optional "volume_per_atom",
# the starting volume around which the EoS is calculated

try:
    codename = sys.argv[1]
except IndexError:
    codename = 'SiestaHere@localhost'

#The code
code = load_code(codename)

#The structure.
stru=mg.Structure.from_file("/home/ebosoni/GeTeSTM/POSCAR")
structure=StructureData(pymatgen_structure=stru)

#The parameters
parameters = Dict(
    dict={
        'xc-functional': 'GGA',
        'xc-authors': 'PBE',
        'max-scfiterations': 500,
        'dm-numberpulay': 4,
        'dm-mixingweight': 0.03,
        'meshcutoff': "900 Ry",
        'dm-tolerance': 1.e-4,
        'Solution-method': 'diagon',
        'electronic-temperature': '25 meV',
        'write-forces': True,
        '%block PAO-PolarizationScheme':
        """
        Ge non-perturbative \n%endblock PAO-PolarizationScheme"""

    })


#The basis set
basis = Dict(dict={
'pao-energy-shift': '300 meV',
'%block pao-basis-sizes': """
Ge DZP
Te DZP
%endblock pao-basis-sizes""",
    })

#The kpoints
kpoints = KpointsData()
kpoints.set_kpoints_mesh([24, 24, 24])

#The pseudopotentials
pseudos=get_pseudos_from_structure(family_name="nc-fr-04_pbe_stringent_psml",structure=structure)

#Resources
options = Dict(
    dict={
        "max_wallclock_seconds": 40360,
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
inputs = {
    'structure': structure,
    'parameters': parameters,
    'code': code,
    'basis': basis,
    'kpoints': kpoints,
    'pseudos': pseudos,
    'options': options,
    'metadata': {
        "label": "PBE stringent DZP"},
}

process = submit(EqOfStateGeTe, **inputs)
print("Submitted workchain; ID={}".format(process.pk))
print(
    "For information about this workchain type: verdi process show {}".format(
        process.pk))
print("For a list of running processes type: verdi process list")
