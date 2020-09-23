from aiida_siesta.workflows.utils.generator_absclass import InputsGenerator
from aiida.plugins import WorkflowFactory
from aiida_siesta.groups.pseudos import PsmlFamily

#@pytest.fixture
#def protocol_registry():
#
#    class SubInputsGenerator(InputsGenerator):
#
#        _protocols = {'efficiency': {'description': 'description'}, 'precision': {'description': 'description'}}
#        _default_protocol = 'efficiency'
#
#    return SubProtocolRegistry()


def test_validation(aiida_profile):
    """Test the validation of subclasses of `InputsGenerator`."""

    #Here I fake the pseudofamilies
    #PsmlFamily.objects.get_or_create("nc-sr-04_pbe_standard-psf")
    #PsmlFamily.objects.get_or_create("nc-sr-04_pbe_stringent-psf")
    PsmlFamily.objects.get_or_create("nc-sr-04_pbe_standard_psml")

    class SubInputsGenerator(InputsGenerator):

        _calc_types = None

        def get_inputs_dict(self, structure, calc_engines, protocol, **kwargs):

            pass

        def get_filled_builder(self, structure, calc_engines, protocol, **kwargs):

            pass

    import pytest
    with pytest.raises(RuntimeError):
        SubInputsGenerator(WorkflowFactory("siesta.base"))

    class SubInputsGenerator(InputsGenerator):

        _calc_types = {"si":{"code": "pp", "resources": "tt"}}

        def get_inputs_dict(self, structure, calc_engines, protocol, **kwargs):

            pass

        def get_filled_builder(self, structure, calc_engines, protocol, **kwargs):

            pass

    import pytest
    with pytest.raises(TypeError):
        SubInputsGenerator()


    class SubInputsGenerator(InputsGenerator):

        _calc_types = {"si":{"code": "pp", "resources": "tt"}}

    import pytest
    with pytest.raises(TypeError):
        SubInputsGenerator(WorkflowFactory("siesta.base"))

    class SubInputsGenerator(InputsGenerator):

        _calc_types = {"si":{"code": "pp", "resources": "tt"}}

        def get_inputs_dict(self, structure, calc_engines, protocol, **kwargs):

            pass

        def get_filled_builder(self, structure, calc_engines, protocol, **kwargs):

            pass

    subinpgen=SubInputsGenerator(WorkflowFactory("siesta.base")) 
    assert  subinpgen.how_to_pass_computation_options()[1] == {"si":{"code": "pp", "resources": "tt"}}
