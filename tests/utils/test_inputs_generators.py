#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
from aiida_siesta.groups.pseudos import PsmlFamily
from aiida.plugins import WorkflowFactory, CalculationFactory


def test_siestacalc_inpgen(aiida_profile, fixture_code, generate_structure, generate_psml_fam):
    """Test the validation of subclasses of `InputsGenerator`."""

    generate_psml_fam("nc-sr-04_pbe_standard_psml","Si") #This will stay for the entire session!

    from aiida_siesta.utils.inputs_generators import SiestaCalculationInputsGenerator

    inp_gen = SiestaCalculationInputsGenerator(CalculationFactory("siesta.siesta"))
    structure = generate_structure()
    protocol = inp_gen.get_default_protocol_name()
    code = fixture_code("siesta.siesta")
    code.store()
    calc_engines = {"siesta": {'code': code.uuid, 'options': {"resources": {"num_mpiprocs_per_machine": 1}, "max_wallclock_seconds": 360}}}

    build = inp_gen.get_filled_builder(structure, calc_engines, protocol, spin="polarized")

    assert "parameters" in build


def test_baseworkchain_inpgen(aiida_profile, fixture_code, generate_structure):
    """Test the validation of subclasses of `InputsGenerator`."""

    from aiida_siesta.utils.inputs_generators import BaseWorkChainInputsGenerator

    inp_gen = BaseWorkChainInputsGenerator(WorkflowFactory("siesta.base"))
    structure = generate_structure()
    protocol = inp_gen.get_default_protocol_name()
    code = fixture_code("siesta.siesta")
    code.store()
    calc_engines = {"siesta": {'code': code.uuid, 'options': {"resources": {"num_mpiprocs_per_machine": 1}, "max_wallclock_seconds": 360}}} 

    build = inp_gen.get_filled_builder(structure, calc_engines, protocol, spin="polarized")

    assert "parameters" in build

#def test_bandgapworkchain_inpgen(aiida_profile, fixture_code, generate_structure):
#    """Test the validation of subclasses of `InputsGenerator`."""

#    from aiida_siesta.utils.inputs_generators import BandgapWorkChainInputsGenerator

#    inp_gen = BandgapWorkChainInputsGenerator(WorkflowFactory("siesta.bandgap"))
#    structure = generate_structure()
#    protocol = inp_gen.get_default_protocol_name()
#    code = fixture_code("siesta.siesta")
#    code.store()
#    calc_engines = {"siesta": {'code': code.uuid, 'options': {"resources": {"num_mpiprocs_per_machine": 1}, "max_wallclock_seconds": 360}}}

#    build = inp_gen.get_filled_builder(structure, calc_engines, protocol, bands_path_generator="seekpath")

#    assert "parameters" in build

def test_eosworkchain_inpgen(aiida_profile, fixture_code, generate_structure):
    """Test the validation of subclasses of `InputsGenerator`."""

    from aiida_siesta.utils.inputs_generators import EosWorkChainInputsGenerator

    inp_gen = EosWorkChainInputsGenerator(WorkflowFactory("siesta.eos"))
    structure = generate_structure()
    protocol = inp_gen.get_default_protocol_name()
    code = fixture_code("siesta.siesta")
    code.store()
    calc_engines = {"siesta": {'code': code.uuid, 'options': {"resources": {"num_mpiprocs_per_machine": 1}, "max_wallclock_seconds": 360}}}

    build = inp_gen.get_filled_builder(structure, calc_engines, protocol, relaxation_type="atoms_only")

    assert "parameters" in build

def test_stmworkchain_inpgen(aiida_profile, fixture_code, generate_structure):
    """Test the validation of subclasses of `InputsGenerator`."""

    from aiida_siesta.utils.inputs_generators import StmWorkChainInputsGenerator

    inp_gen = StmWorkChainInputsGenerator(WorkflowFactory("siesta.stm"))
    structure = generate_structure()
    protocol = inp_gen.get_default_protocol_name()
    code = fixture_code("siesta.siesta")
    code.store()
    code2 = fixture_code("siesta.stm")
    code2.store()
    calc_engines = {"siesta": {'code': code.uuid, 'options': {"resources": {"num_mpiprocs_per_machine": 1}, "max_wallclock_seconds": 360}},
                    "stm": {'code': code2.uuid, 'options': {"resources": {"num_mpiprocs_per_machine": 1}, "max_wallclock_seconds": 360}}
            }

    build = inp_gen.get_filled_builder(structure, calc_engines, protocol, stm_mode="constant-height", stm_value=2)

    assert "parameters" in build
