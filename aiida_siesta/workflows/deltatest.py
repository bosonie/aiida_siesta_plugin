import numpy as np
from aiida import orm
from aiida.engine import WorkChain, calcfunction, ToContext
from aiida_siesta.workflows.eos import EqOfStateFixedCellShape


def get_reference(element):
    """
    Extract the reference Vo B0 B1 from a file
    """
    f = open('/home/ebosoni/DeltaTest/WIEN2k.txt', 'r')
    for line in f:
        a = line.split()
        if a[0] == element:
            vol = a[1]
            b0 = a[2]  #In GPa
            b1 = a[3]
    return float(vol), float(b0), float(b1)


@calcfunction
def calcDelta(v0wF, b0wF, b1wF, v0fF, b0fF, b1fF):
    """
    Calculate the Delta value, function copied from the official DeltaTest repository.
    I don't understand what it does, but it works.
    """
    v0w = v0wF.value
    b0w = b0wF.value
    b1w = b1wF.value
    v0f = v0fF.value
    b0f = b0fF.value
    b1f = b1fF.value

    #The reference bulk modulus is in GPa, need to convert back to eV/ang^3
    b0w = b0w * 10.**9. / 1.602176565e-19 / 10.**30.

    #vref = 30.
    #bref = 100. * 10.**9. / 1.602176565e-19 / 10.**30.

    Vi = 0.94 * (v0w + v0f) / 2.
    Vf = 1.06 * (v0w + v0f) / 2.

    a3f = 9. * v0f**3. * b0f / 16. * (b1f - 4.)
    a2f = 9. * v0f**(7. / 3.) * b0f / 16. * (14. - 3. * b1f)
    a1f = 9. * v0f**(5. / 3.) * b0f / 16. * (3. * b1f - 16.)
    a0f = 9. * v0f * b0f / 16. * (6. - b1f)

    a3w = 9. * v0w**3. * b0w / 16. * (b1w - 4.)
    a2w = 9. * v0w**(7. / 3.) * b0w / 16. * (14. - 3. * b1w)
    a1w = 9. * v0w**(5. / 3.) * b0w / 16. * (3. * b1w - 16.)
    a0w = 9. * v0w * b0w / 16. * (6. - b1w)

    x = [0, 0, 0, 0, 0, 0, 0]

    x[0] = (a0f - a0w)**2
    x[1] = 6. * (a1f - a1w) * (a0f - a0w)
    x[2] = -3. * (2. * (a2f - a2w) * (a0f - a0w) + (a1f - a1w)**2.)
    x[3] = -2. * (a3f - a3w) * (a0f - a0w) - 2. * (a2f - a2w) * (a1f - a1w)
    x[4] = -3. / 5. * (2. * (a3f - a3w) * (a1f - a1w) + (a2f - a2w)**2.)
    x[5] = -6. / 7. * (a3f - a3w) * (a2f - a2w)
    x[6] = -1. / 3. * (a3f - a3w)**2.

    y = [0, 0, 0, 0, 0, 0, 0]

    y[0] = (a0f + a0w)**2 / 4.
    y[1] = 3. * (a1f + a1w) * (a0f + a0w) / 2.
    y[2] = -3. * (2. * (a2f + a2w) * (a0f + a0w) + (a1f + a1w)**2.) / 4.
    y[3] = -(a3f + a3w) * (a0f + a0w) / 2. - (a2f + a2w) * (a1f + a1w) / 2.
    y[4] = -3. / 20. * (2. * (a3f + a3w) * (a1f + a1w) + (a2f + a2w)**2.)
    y[5] = -3. / 14. * (a3f + a3w) * (a2f + a2w)
    y[6] = -1. / 12. * (a3f + a3w)**2.

    Fi = np.zeros_like(Vi)
    Ff = np.zeros_like(Vf)

    Gi = np.zeros_like(Vi)
    Gf = np.zeros_like(Vf)

    for n in range(7):
        Fi = Fi + x[n] * Vi**(-(2. * n - 3.) / 3.)
        Ff = Ff + x[n] * Vf**(-(2. * n - 3.) / 3.)

        Gi = Gi + y[n] * Vi**(-(2. * n - 3.) / 3.)
        Gf = Gf + y[n] * Vf**(-(2. * n - 3.) / 3.)

    Delta = 1000. * np.sqrt((Ff - Fi) / (Vf - Vi))
    #Deltarel = 100. * np.sqrt((Ff - Fi) / (Gf - Gi))
    #Delta1 = 1000. * np.sqrt((Ff - Fi) / (Vf - Vi)) \
    #    / (v0w + v0f) / (b0w + b0f) * 4. * vref * bref

    return orm.Float(Delta)  #, Deltarel, Delta1


#def ret_delta(Vref, Bref, B1ref, Vo, Bo, B1):
#    """
#    Calculates the Delta value, implemented by me, easy to undestand,
#    but it uses scipy.
#    """
#    from scipy.integrate import quad
#    Bref = float(Bref) * 10.**9. / 1.602176565e-19 / 10.**30.
#    Vm = (Vo + Vref) / 2
#    I = quad(diff_murn_squared, 0.94 * Vm, 1.06 * Vm, args=(Vo, Bo, B1, Vref, Bref, B1ref))
#    #print p.label, Vo, Bo, Bp, np.sqrt(I[0]/Vm/0.12)*1000, "meV"
#    return np.sqrt(I[0] / Vm / 0.12) * 1000


def diff_murn_squared(x, Vo, Bo, Bp, Vo2, Bo2, Bp2):
    """
    Return the difference between two Birch-Murnagahn functions.
    """
    r = (Vo / x)**(2. / 3.)
    r1 = (Vo2 / x)**(2. / 3.)
    return (
        9. / 16. * Bo * Vo * (r - 1.)**2 * (2. + (Bp - 4.) * (r - 1.)) -
        (9. / 16. * Bo2 * Vo2 * (r1 - 1.)**2 * (2. + (Bp2 - 4.) * (r1 - 1.)))
    )**2


def structure_init(element):
    """
    Workfunction to create structure of a given element taking it from a reference
    list of scructures and a reference volume.
    :param element: The element to create the structure with.
    :return: The structure and the kpoint mesh (from file, releted to the structure!).
    """
    import pymatgen as mg

    f = open('/home/ebosoni/DeltaTest/WIEN2k.txt', 'r')
    for line in f:
        a = line.split()
        if a[0] == element:
            vol = a[1]

    in_structure = mg.Structure.from_file("/home/ebosoni/DeltaTest/CIFs/{0}.cif".format(element), primitive=False)
    newreduced = in_structure.copy()
    newreduced.scale_lattice(float(vol) * in_structure.num_sites)
    structure = orm.StructureData(pymatgen_structure=newreduced)

    return structure


def kpgen(structure, distance):
    """
    K-points generator. I will implement also a copy from file way in the future.
    """
    kpoints_mesh = orm.KpointsData()
    kpoints_mesh.set_cell_from_structure(structure)
    kpoints_mesh.set_kpoints_mesh_from_density(distance=distance)

    return kpoints_mesh


def get_basis(element):
    """
    Basis set generator. Implement the exceptions due to problems with PseudoDojo pseudos.
    """
    basis_dict = {
        'pao-energy-shift': '100 meV',  #default is 0.02 Ry = 272 meV
        '%block pao-basis-sizes': "\n {} DZP\n%endblock pao-basis-sizes".format(element),
    }

    if element in ["Li", "Be", "Mg", "Na", "Ga", "Ge", "As", "Se", "In"]:
        basis_dict['%block PAO-PolarizationScheme'] = "\n {} non-perturbative\n%endblock PaoPolarizationScheme".format(
            element
        )

    return orm.Dict(dict=basis_dict)


def get_parameters(element):
    """
    Parameters generator. Implements the choice of spin polarization according to the stndards
    of the delta test.
    """
    parameters = {
        'xc-functional': 'GGA',
        'xc-authors': 'PBE',
        'mesh-cutoff': "500 Ry",
        'max-scf-iterations': 500,
        'scf-mixer-history': 5,
        'scf-mixer-weight': 0.1,
        'scf-dm-tolerance': 1.e-5,
        'electronic-temperature': '25 meV',  #294 K, 0.00183747 Ry
        'write-forces': True,
    }
    if element in ["O", "Cr", "Mn", "Fe", "Co", "Ni"]:
        parameters["spin"] = "polarized"
        if element in ["Fe", "Co", "Ni"]:
            parameters["dm-init-spin-af"] = False
        elif element in ["Mn", "Cr"]:
            parameters["write-mulliken-pop"] = 1
            parameters["dm-init-spin-af"] = True
        else:
            parameters["%block dm-init-spin"] = "\n1 + \n2 + \n3 - \n4 - \n%endblock dmintspin"

    return orm.Dict(dict=parameters)


class DeltaWorkflow(WorkChain):

    @classmethod
    def define(cls, spec):
        super(DeltaWorkflow, cls).define(spec)
        spec.input("element", valid_type=orm.Str)
        spec.input("code", valid_type=orm.Code)
        spec.input('options', valid_type=orm.Dict)

        spec.outline(cls.inpsetup, cls.run_eqs, cls.return_results)

        spec.output('EosData', valid_type=orm.Dict, required=True)
        spec.output('DeltaValue', valid_type=orm.Float, required=True)

        spec.exit_code(301, 'EOS_FIT_FAIL', message='Problem in the fit of the EoS data')

    def inpsetup(self):
        structure = structure_init(self.inputs.element.value)
        kpoints = kpgen(structure, 0.062)
        parameters = get_parameters(self.inputs.element.value)
        basis = get_basis(self.inputs.element.value)
        pseudo_family = orm.Str("nc-sr-04_pbe_stringent_psml")
        self.ctx.inputs = {
            'code': self.inputs.code,
            'structure': structure,
            'pseudo_family': pseudo_family,
            'parameters': parameters,
            'basis': basis,
            'kpoints': kpoints,
            'options': self.inputs.options,
            'metadata': {
                'label': self.inputs.element.value
            }
        }

    def run_eqs(self):

        inputs = self.ctx.inputs
        #future = self.submit(SiestaBaseWorkChain, **inputs)
        future = self.submit(EqOfStateFixedCellShape, **inputs)
        el = self.inputs.element.value
        self.report('Launching EqOfStateFixedCellShape,<{0}>, for element {1}'.format(future.pk, el))

        return ToContext(eos_calc=future)

    def return_results(self):
        self.report('Concluded EqOfStateFixedCellShape for {}'.format(self.inputs.element.value))
        Vref, Bref, B1ref = get_reference(self.inputs.element.value)
        outres = self.ctx.eos_calc.outputs.results_dict
        self.out('EosData', outres)
        if "fit_res" not in outres.get_dict():
            self.report('EqOfStateFixedCellShape failed to perform Birch-Murnaghan fit')
            return self.exit_codes.EOS_FIT_FAIL
        fitres = outres.get_dict()["fit_res"]
        Vo = float(fitres['Vo(ang^3/atom)'])
        Bo = float(fitres['Bo(eV/ang^3)'])
        B1 = float(fitres['B1'])
        #delta = ret_delta(Vref, Bref, B1ref, Vo, Bo, B1)
        delta2 = calcDelta(
            orm.Float(Vref), orm.Float(Bref), orm.Float(B1ref), orm.Float(Vo), orm.Float(Bo), orm.Float(B1)
        )
        self.out('DeltaValue', delta2)
        self.report('End of DeltaWorkflow for {}'.format(self.inputs.element.value))