import nengo
import nengo.spa as spa

spinnaker = True

dimensions = 16

class Sequence(spa.SPA):
    def __init__(self):
        spa.SPA.__init__(self)
        #Specify the modules to be used
        self.cortex = spa.Buffer(dimensions=dimensions)

        #Specify the action mapping
        actions = spa.Actions(
            'dot(cortex, A) --> cortex = B',
            'dot(cortex, B) --> cortex = C',
            'dot(cortex, C) --> cortex = D',
            'dot(cortex, D) --> cortex = E',
            'dot(cortex, E) --> cortex = A'
            )

        self.bg = spa.BasalGanglia(actions=actions)
        self.thal = spa.Thalamus(self.bg)

        #Specify the input
        def start(t):
            if t<0.05: return 'A'
            else: return '0'

        self.input = spa.Input(cortex=start)



model = Sequence(label='Sequence_Module')

with model:
    #Probe things that will be plotted
    pActions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    pUtility = nengo.Probe(model.bg.input, synapse=None)

#Make a simulator and run it
if not spinnaker:
    sim = nengo.Simulator(model)
else:
    import nengo_spinnaker

    config = nengo_spinnaker.Config()
    for n in model.input.input_nodes.values():
        config[n].f_of_t = True

    sim = nengo_spinnaker.Simulator(model, config=config)

sim.run(0.5)

from matplotlib import pyplot as plt
figure = plt.figure()

p1 = figure.add_subplot(2,1,1)
p1.plot(sim.trange(), sim.data[pActions])
p1.set_ylabel('Action')
p1.set_ylim([-0.1, 1.1])

p2 = figure.add_subplot(2,1,2)
p2.plot(sim.trange(), sim.data[pUtility])
p2.set_ylabel('Utility')
p2.set_ylim([-0.1, 1.1])

plt.show(block=True)
