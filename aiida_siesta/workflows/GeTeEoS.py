import numpy as np
from pymatgen import Lattice, Structure
from aiida.common import AttributeDict
from aiida.engine import WorkChain, calcfunction, ToContext
from aiida.orm import Float, StructureData, Dict, ArrayData
from aiida_siesta.calculations.tkdict import FDFDict
from aiida_siesta.workflows.base import SiestaBaseWorkChain
from aiida.engine import while_, if_


@calcfunction
def scale_to_vol(stru, vol):
    """
    Calcfunction to scale a structure to a target volume. Uses pymatgen.
    :param stru: An aiida structure
    :param vol: The target volume per atom in angstroms
    :return: The new scaled AiiDA structure
    """

    in_structure = stru.get_pymatgen()
    new = in_structure.copy()
    new.scale_lattice(float(vol) * in_structure.num_sites)
    structure = StructureData(pymatgen_structure=new)

    return structure


@calcfunction
def rescale(structure, scale):
    """
    Calcfunction to rescale a structure by a scaling factor. Uses ase.
    :param structure: An AiiDA structure to rescale
    :param scale: The scale factor
    :return: The rescaled structure
    """

    the_ase = structure.get_ase()
    new_ase = the_ase.copy()
    new_ase.set_cell(the_ase.get_cell() * float(scale), scale_atoms=True)
    new_structure = StructureData(ase=new_ase)

    return new_structure


@calcfunction
def get_info(outstress, outpar, struct):
    """
    Calcfunction creating a dictionary with selected inputs and results of a SiestaBaseWC, usefull for
    various post processes: the positions analysis and the creation of the Equation of State.
    :param struct: aiida structure of a SiestaBaseWC
    :param outpar: the output_parameters of a SiestaBaseWC
    :param outstress: the forces_and_stress of a SiestaBaseWC
    :return: A dictionary containing volume per atom, energy per atom, angle of the cell, side of the  
             cell, difference of the diagonal stress and the positional parameter 
             (pos Te[pos,pos,pos] Ge[1-pos,1-pos,1pos])
    """

    evdict = {}
    pystr = struct.get_pymatgen_structure()
    evdict["alpha"] = pystr.lattice.alpha
    evdict["latt_const"] = pystr.lattice.a
    for site in pystr.sites:
        if site.specie.name == "Te":
            pos = site.a
    evdict["pos"] = pos
    evdict["vol"] = struct.get_cell_volume() / len(struct.sites)
    evdict["vol_units"] = 'ang^3/atom'
    evdict["en"] = outpar['E_KS'] / len(struct.sites)
    evdict["en_units"] = outpar['E_KS_units'] + '/atom'
    vectst = outstress.get_array("stress")
    evdict["stress_diff"] = vectst[0][0] - vectst[2][2]
    #add units stress
    resultdict = Dict(dict=evdict)

    return resultdict


def delta_project_BM_fit(volumes, energies):  #pylint: disable=invalid-name
    """
    The fitting procedure implemented in this function was copied from the Delta Project Code.
    https://github.com/molmod/DeltaCodesDFT/blob/master/eosfit.py
    It is introduced to fully uniform the delta test procedure
    with the one performed with other codes, moreover it has the upside to not use scypi.
    """

    #Does the fit always succeed?
    fitdata = np.polyfit(volumes**(-2. / 3.), energies, 3, full=True)
    ssr = fitdata[1]
    sst = np.sum((energies - np.average(energies))**2.)
    residuals0 = ssr / sst
    deriv0 = np.poly1d(fitdata[0])  #pylint: disable=invalid-name
    deriv1 = np.polyder(deriv0, 1)  #pylint: disable=invalid-name
    deriv2 = np.polyder(deriv1, 1)  #pylint: disable=invalid-name
    deriv3 = np.polyder(deriv2, 1)  #pylint: disable=invalid-name

    volume0 = 0
    x = 0
    for x in np.roots(deriv1):
        if x > 0 and deriv2(x) > 0:
            E0 = deriv0(x)  #pylint: disable=invalid-name
            volume0 = x**(-3. / 2.)
            break

    #Here something checking if the fit is good!
    #The choice of residuals0 > 0.01 it is not supported by a real scientific reason, just from experience.
    #Values ~ 0.1 are when fit random numbers, ~ 10^-5 appears for good fits. The check on the presence of
    #a minmum covers the situations when an almost linear dependence is fitted (very far from minimum)
    if volume0 == 0 or residuals0 > 0.01:
        return residuals0, volume0
    derivV2 = 4. / 9. * x**5. * deriv2(x)  #pylint: disable=invalid-name
    derivV3 = (-20. / 9. * x**(13. / 2.) * deriv2(x) - 8. / 27. * x**(15. / 2.) * deriv3(x))  #pylint: disable=invalid-name
    bulk_modulus0 = derivV2 / x**(3. / 2.)
    bulk_deriv0 = -1 - x**(-3. / 2.) * derivV3 / derivV2
    return E0, volume0, bulk_modulus0, bulk_deriv0


@calcfunction
def fit_eos(arda):
    """
    Calcfunction that collects all the E vs V, performs the birch_murnaghan fit 
    and creates a dictionary with all the relevant results. Uses scipy.optimize and numpy.
    :param arda: ArrayData containing the eos data.
    :return: A dictionary containing the eos data and the fit parameters of the murnagan fit.
    """
    vol = arda.get_array("eos")[:,0]
    ener = arda.get_array("eos")[:,2]
    vol = vol.astype('float64')
    ener = ener.astype('float64')
    try:
        E0, volume0, bulk_modulus0, bulk_deriv0 = delta_project_BM_fit(vol, ener)  #pylint: disable=invalid-name
        fit_res = {}
        fit_res["Eo(eV/atom)"] = E0
        fit_res["Vo(ang^3/atom)"] = volume0
        fit_res["Bo(eV/ang^3)"] = bulk_modulus0
        fit_res["Bo(GPa)"] = bulk_modulus0 * 160.21766208
        fit_res["B1"] = bulk_deriv0
    except:  # nopep8 #pylint: disable=bare-except
        fit_res = {}
        #residuals0, volume0 = delta_project_BM_fit(volumes, energies)
        #In the future we could use these info to improve help,
        #residuals0 is a np array

    if fit_res:
        result_dict = Dict(dict={'eos_data': arda.get_array("eos"), "fit_res": fit_res})
    else:
        result_dict = Dict(dict={'eos_data': arda.get_array("eos")})

    return result_dict


@calcfunction
def fit_pos(**calcs):
    """
    Calcfunction that collects all the E vs pos (the position value), performs the birch_murnaghan fit 
    and creates a dictionary with all the relevant results. Uses scipy.optimize and numpy.
    :param calcs: Dictionaries result of get_info
    :return: A dictionary containing a list of relevant info and the results of the murnagan fit.
    P.S. Using the Birch-Murnagan for the position parameter is a streach, not sure has theoretical fundation.
    """

    data = []
    ener = []
    pos = []
    for cal in calcs:
        arg = calcs[cal]
        ener.append(arg["en"])
        pos.append(arg["pos"])
        data.append([arg["vol"], arg["en"], arg["vol_units"], arg["en_units"], arg["pos"]])

    energies = np.array(ener)
    pos = np.array(pos)
    try:
        E0, volume0, bulk_modulus0, bulk_deriv0 = delta_project_BM_fit(pos, energies)
        fit_res = {}
        fit_res["Eo(eV/atom)"] = E0
        fit_res["minpos"] = volume0
    except:
        fit_res = {}

    if fit_res is None:
        result_dict = Dict(dict={'all_data': data})
    else:
        result_dict = Dict(dict={'all_data': data, "fit_res": fit_res})

    return result_dict


@calcfunction
def create_structure(volume, angle, pos):
    ang = angle.value
    cos = np.cos(np.radians(ang))
    base = (volume.value / (np.sqrt(2 * cos**3 - 3 * cos**2 + 1)))**(1 / 3)
    lattice = Lattice.from_parameters(a=base, b=base, c=base, alpha=ang, beta=ang, gamma=ang)
    thex = pos.value
    truct = Structure(lattice, ["Te", "Ge"], coords=[[thex, thex, thex], [1 - thex, 1 - thex, 1 - thex]])

    return StructureData(pymatgen_structure=truct)


class EqOfStateGeTe(WorkChain):
    """
    WorkChain to calculate the equation of state of GeTe. All the structure parameters are manually
    changed, no use of Siesta relax option. This is because Siesta doesn't have symmetries.
    The three parameters defining a GeTe structure are:
     pos: the positional parameter defining the position of the two atoms (Te[pos,pos,pos] Ge[1-pos,1-pos,1pos])
     angle: the angle of the rombohedral cell
     volume: the volume of the cell
    For a fixed volume and angle, the pos giving minimum energy (equivalent at minimum forces I hope) is 
    calculated. The procedure is repeted for 5 angles at a fixed volume. The analysis of this calculations
    indicate the angle (for each volume) at which the stress tensor is hydrostatic. Finally the E(V) data
    are analyzed in order to find the equilibrium volume.
    """

    @classmethod
    def define(cls, spec):
        super(EqOfStateGeTe, cls).define(spec)
        spec.input("volume_per_atom", valid_type=Float, required=False, help="Starting vol")
        spec.expose_inputs(SiestaBaseWorkChain, exclude=('metadata',))
        spec.inputs._ports['pseudos'].dynamic = True  #Temporary fix to issue #135 plumpy
        spec.outline(
            cls.initio,
            while_(cls.should_run_vol)(
                while_(cls.should_run_angles)(
                    cls.run_base_wcs, 
                    cls.analyse_pos, 
                    if_(cls.should_run_final_pos)(
                        cls.run_final_pos,
                        cls.stress_to_var,
                        ),
                    ),
                cls.analyse_stress,
                cls.analyse_pos,
                if_(cls.should_run_final_pos)(cls.run_final_pos,),
                cls.final_angle,
                ),
            cls.process_eos,
            if_(cls.should_run_eq_vol)(
                while_(cls.should_run_angles)(
                    cls.run_base_wcs,
                    cls.analyse_pos,
                    if_(cls.should_run_final_pos)(
                        cls.run_final_pos,
                        cls.stress_to_var,
                        ),
                    ),
                cls.analyse_stress,
                cls.analyse_pos,
                if_(cls.should_run_final_pos)(cls.run_final_pos,),
                cls.final_res,
                )
        )

        spec.output('eos_dict', valid_type=Dict, required=True)
        spec.output('equilibrium_structure', valid_type=StructureData, required=False)
        spec.output('final_structure_info', valid_type=Dict, required=False)
        spec.output('final_forces_stress', valid_type=ArrayData, required=False)

    def initio(self):
        self.ctx.scales = (0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06)
        self.ctx.angscale = (0.99, 1., 1.01, 1.02)
        self.ctx.atompos = (0.226, 0.228, 0.230, 0.232, 0.234, 0.236, 0.238)
        self.ctx.countangl = 0
        self.ctx.countvol = 0
        self.ctx.stresscoll = []
        self.ctx.eoscoll = []
        self.ctx.changevol = True

        self.report("Starting EqOfStateGeTe Workchain")

        if "pseudo_family" not in self.inputs:
            if not self.inputs.pseudos:
                raise ValueError('neither an explicit pseudos dictionary nor a pseudo_family was specified')
        if "volume_per_atom" in self.inputs:
            self.ctx.s0 = scale_to_vol(self.inputs.structure, self.inputs.volume_per_atom)
        else:
            self.ctx.s0 = self.inputs.structure

        test_input_params = FDFDict(self.inputs.parameters.get_dict())
        for k, v in sorted(test_input_params.get_filtered_items()):
            if k in ('mdtypeofrun', 'mdnumcgsteps', 'mdrelaxcellonly'):
                raise ValueError('Not supposed to relax!!!')
                #better to strip them automatically in the future

    def should_run_vol(self):
        return self.ctx.countvol < len(self.ctx.scales)

    def should_run_angles(self):
        return self.ctx.countangl < len(self.ctx.angscale)

    def run_base_wcs(self):
        """
        run the SiestaBaseWorkChain at fixed volume, fixed, angle, changing only atom dinstance.
        """
        calcs = {}
        #pystru = self.ctx.s0.get_pymatgen_structure()
        scalev = self.ctx.scales[self.ctx.countvol]
        scaledstru = rescale(self.ctx.s0, Float(scalev))
        pystru = scaledstru.get_pymatgen_structure()
        if self.ctx.changevol:
            self.ctx.volnow = pystru.volume
        ang = pystru.lattice.angles[1]
        scalea = self.ctx.angscale[self.ctx.countangl]
        self.ctx.angnow = ang * scalea
        self.report('Analysing volume {0}, and angle {1}'.format(self.ctx.volnow, self.ctx.angnow))
        for thex in self.ctx.atompos:
            inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
            inputs["structure"] = create_structure(Float(self.ctx.volnow), Float(self.ctx.angnow), Float(thex))
            future = self.submit(SiestaBaseWorkChain, **inputs)
            self.report('Launching SiestaBaseWorkChain<{0}>, at position of {1}'.format(future.pk, thex))
            calcs[("a" + str(self.ctx.angnow) + "_" + str(thex))] = future
        return ToContext(**calcs)  #Here it waits

    def analyse_pos(self):
        """
        For a fixed volume and angle, analyses the E vs pos curve to get the minimum. pos is the positional
        parameter that defines (Te[pos,pos,pos] Ge[1-pos,1-pos,1pos]). With this procedure we get the final
        equilibrium atom positions for a fixed volume and angle.
        """
        self.report('Calculation with different atom positions are over. Analysis start')
        collectwcinfo = {}
        for label in self.ctx.atompos:
            wcnod = self.ctx[("a" + str(self.ctx.angnow) + "_" + str(label))]
            info = get_info(wcnod.outputs.forces_and_stress, wcnod.outputs.output_parameters, wcnod.inputs.structure)
            collectwcinfo["po" + str(label).replace(".", "_")] = info
        res_dict = fit_pos(**collectwcinfo)
        self.ctx.okfitpos = False
        if "fit_res" in res_dict.attributes:
            self.ctx.okfitpos = True
            self.ctx.okthex = res_dict.get_attribute("fit_res")["minpos"]
        else:
            self.ctx.countangl = self.ctx.countangl + 1

    def should_run_final_pos(self):
        """
        Only if the fit E vs pos is succesfull, I run a final SiestaBaseWorkChain at the equilibrium
        atom positions
        """
        return self.ctx.okfitpos

    def run_final_pos(self):
        """
        The run of the SiestaBaseWorkChain at the equilibrium atom positions
        """
        thex = self.ctx.okthex
        self.report('The equilibrium position is {}'.format(thex))
        inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
        inputs["structure"] = create_structure(Float(self.ctx.volnow), Float(self.ctx.angnow), Float(thex))
        future = self.submit(SiestaBaseWorkChain, **inputs)
        self.report(
            'Launching SiestaBaseWorkChain<{0}>, last at vol {1} and ang {2}'.format(
                future.pk, self.ctx.volnow, self.ctx.angnow
            )
        )
        return ToContext(finang=future)

    def stress_to_var(self):
        """
        Collects the stress on a variable that will be later analysed
        """
        self.report('Collecting results at vol {0} and ang {1}'.format(self.ctx.volnow, self.ctx.angnow))
        wcnod = self.ctx.finang
        info = get_info(wcnod.outputs.forces_and_stress, wcnod.outputs.output_parameters, wcnod.inputs.structure)
        self.ctx.stresscoll.append([info["alpha"], info["stress_diff"]])
        self.ctx.countangl = self.ctx.countangl + 1

    def analyse_stress(self):
        self.report('Calculations with different angles are over for volume {}.'.format(self.ctx.volnow))
        self.report('{}'.format(self.ctx.stresscoll))
        arrstrang = np.array(self.ctx.stresscoll)
        fitdata = np.polyfit(arrstrang[:, 0], arrstrang[:, 1], 1, full=True)
        okangle = -fitdata[0][1] / fitdata[0][0]
        self.ctx.angnow = okangle
        self.report('The angle {0} is the favourable for volume {1}'.format(okangle, self.ctx.volnow))
        calcs = {}
        for thex in self.ctx.atompos:
            inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
            inputs["structure"] = create_structure(Float(self.ctx.volnow), Float(self.ctx.angnow), Float(thex))
            future = self.submit(SiestaBaseWorkChain, **inputs)
            self.report('Launching SiestaBaseWorkChain<{0}>, at position of {1}'.format(future.pk, thex))
            calcs[("a" + str(self.ctx.angnow) + "_" + str(thex))] = future
        return ToContext(**calcs)  #Here it waits

    #here I reput the same analyse_pos, should_run_final_pos and run_final_pos as before

    def final_angle(self):
        wcnod = self.ctx.finang
        info = get_info(wcnod.outputs.forces_and_stress, wcnod.outputs.output_parameters, wcnod.inputs.structure)
        self.ctx.eoscoll.append([info["vol"], info["vol_units"], info["en"], info["en_units"]])
        scal = self.ctx.scales[self.ctx.countvol]
        self.report("Concluded all analisys for volume {0}, meaning scale {1}".format(self.ctx.volnow, scal))
        self.ctx.countvol = self.ctx.countvol + 1
        self.ctx.countangl = 0

    def process_eos(self):
        self.report('All 7 calculations finished. Post process starts')

        arda = ArrayData()
        arda.set_array(name="eos", array=np.array(self.ctx.eoscoll))
        res_dict = fit_eos(arda)

        self.out('eos_dict', res_dict)

        if "fit_res" in res_dict.attributes:
            self.report('Birch-Murnaghan fit was succesfull, running analysis on eq volume')
            self.ctx.volnow = res_dict["fit_res"]["Vo(ang^3/atom)"] * 2
            self.ctx.countangl = 0
            self.ctx.countvol = 0
            self.ctx.changevol = False
        else:
            self.report("WARNING: Birch-Murnaghan fit failed, returning the eos data only,End of WorkChain")

    def should_run_eq_vol(self):
        return not self.ctx.changevol

# Here we redo everything in the while_(should_run_angles), with of course the right volume.

    def final_res(self):
        wcnod = self.ctx.finang
        info = get_info(wcnod.outputs.forces_and_stress, wcnod.outputs.output_parameters, wcnod.inputs.structure)
        self.out('final_structure_info', info)
        self.out('equilibrium_structure', wcnod.inputs.structure) 
        self.out('final_forces_stress', wcnod.outputs.forces_and_stress)

        self.report('End of EqOfState Workchain')
