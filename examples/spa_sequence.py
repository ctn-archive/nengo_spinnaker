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

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))
p1 = fig.add_subplot(3,1,1)

"""
p1.plot(spa.similarity(sim.data, pState))
p1.legend(pState.target.vocab.keys, fontsize='x-small')
p1.set_ylabel('State')
"""

p2 = fig.add_subplot(3,1,2)
p2.plot(sim.data[pActions])
p2.set_ylabel('Action')

p3 = fig.add_subplot(3,1,3)
p3.plot(sim.data[pUtility])
p3.set_ylabel('Utility')
fig.subplots_adjust(hspace=0.2)
plt.show()
