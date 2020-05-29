from aiida import orm
from aiida.engine import while_, submit, WorkChain, ToContext, calcfunction
from aiida_siesta.workflows.DeltaTest import DeltaWorkflow


@calcfunction
def get_out(**collect):
    res = {}
    for el in collect:
        res[el] = collect[el].value
    return orm.Dict(dict=res)


#@calcfunction
#def get_info(calc_results):
#    print(calc_results)
#    if not calc_results.is_finished_ok:
#        return orm.Str("fail")
#    else:
#        return calc_results.outputs.DeltaValue


class RunAllDelta(WorkChain):

    @classmethod
    def define(cls, spec):
        super(RunAllDelta, cls).define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input('options', valid_type=orm.Dict)

        spec.outline(cls.inpsetup,
                     while_(cls.should_run_delta)(
                         cls.run_delta,
                         cls.return_res_to_var,
                     ), cls.finish)

        spec.output('DeltaValues', valid_type=orm.Dict, required=True)

    def inpsetup(self):
        self.ctx.lista = ["Mn", "Fe"]
        self.ctx.counter = 0
        self.ctx.collection = {}

    def should_run_delta(self):
        return self.ctx.counter < len(self.ctx.lista)

    def run_delta(self):
        element = self.ctx.lista[self.ctx.counter]
        code = self.inputs.code
        options = self.inputs.options
        inputs = {
            'element': orm.Str(element),
            'code': code,
            'options': options,
            'metadata': {
                'label': '{}'.format(element)
            }
        }

        future = self.submit(DeltaWorkflow, **inputs)
        self.report("Start Delta calculation for {}".format(element))
        return ToContext(result=future)

    def return_res_to_var(self):
        if self.ctx.result.is_finished_ok:
            self.ctx.collection[self.ctx.result.label] = self.ctx.result.outputs.DeltaValue
        else:
            self.ctx.collection[self.ctx.result.label] = orm.Str("fail")
        self.ctx.counter = self.ctx.counter + 1

    def finish(self):
        self.out("DeltaValues", get_out(**self.ctx.collection))
