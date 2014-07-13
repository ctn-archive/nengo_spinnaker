
dimensions = 64  # dimensionality of the cortical representation
subdimensions = 32
   # for SPAUN, dimensions=512
pstc = 0.01

spinnaker = True

import numpy as np

import nengo
import nengo.spa as spa
import nengo_spinnaker

motor_vocab = spa.Vocabulary(dimensions)
motor_vocab.add('ONE', np.eye(dimensions)[2])
motor_vocab.add('TWO', np.eye(dimensions)[3])
motor_vocab.add('THREE', np.eye(dimensions)[4])

model = spa.SPA()
with model:
    model.vision = spa.Buffer(dimensions=dimensions, subdimensions=subdimensions)
    model.phrase = spa.Buffer(dimensions=dimensions, subdimensions=subdimensions)
    model.motor = spa.Buffer(dimensions=dimensions, vocab=motor_vocab, subdimensions=subdimensions)
    model.noun = spa.Memory(dimensions=dimensions, subdimensions=subdimensions)
    model.verb = spa.Memory(dimensions=dimensions, subdimensions=subdimensions)

    model.bg = spa.BasalGanglia(spa.Actions(
        'dot(vision, WRITE) --> verb=vision',
        'dot(vision, ONE+TWO+THREE) --> noun=vision',
        '0.5*(dot(vision, NONE-WRITE-ONE-TWO-THREE) + dot(phrase, WRITE*VERB))'
             '--> motor=phrase*~NOUN',
        ))
    model.thal = spa.Thalamus(model.bg)

    model.cortical = spa.Cortical(spa.Actions(
        'phrase=noun*NOUN',
        'phrase=verb*VERB',
        ))

    def vision_input(t):
        if t<0.5: return 'WRITE'
        elif t<1.0: return 'ONE'
        elif t<1.5: return 'NONE'
        elif t<2.0: return 'WRITE'
        elif t<2.5: return 'TWO'
        elif t<3.0: return 'NONE'
        elif t<3.5: return 'THREE'
        elif t<4.0: return 'WRITE'
        elif t<4.5: return 'NONE'
        else: return '0'
    model.input = spa.Input(vision=vision_input)


with model:
    #Probe things that will be plotted
    pActions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    pUtility = nengo.Probe(model.bg.input, synapse=None)
    pMotor = nengo.Probe(model.motor.state.output, synapse=0.01)
    pNoun = nengo.Probe(model.noun.state.output, synapse=0.01)
    pVerb = nengo.Probe(model.verb.state.output, synapse=0.01)



#Make a simulator and run it
if not spinnaker:
    sim = nengo.Simulator(model)
else:
    import nengo_spinnaker

    config = nengo_spinnaker.Config()
    for n in model.input.input_nodes.values():
        config[n].f_of_t = True

    sim = nengo_spinnaker.Simulator(model, config=config)

sim.run(6)

from matplotlib import pyplot as plt
figure = plt.figure()

N = 5

p1 = figure.add_subplot(N,1,1)
p1.plot(sim.trange(), sim.data[pActions])
p1.set_ylabel('Action')
p1.set_ylim([-0.1, 1.5])

p2 = figure.add_subplot(N,1,2)
p2.plot(sim.trange(), sim.data[pUtility])
p2.set_ylabel('Utility')
p2.set_ylim([-0.1, 1.5])

p2 = figure.add_subplot(N,1,3)
p2.plot(sim.trange(), sim.data[pMotor])
p2.set_ylabel('Motor')
p2.set_ylim([-0.1, 1.5])

p2 = figure.add_subplot(N,1,4)
p2.plot(sim.trange(), sim.data[pNoun])
p2.set_ylabel('Noun')

p2 = figure.add_subplot(N,1,5)
p2.plot(sim.trange(), sim.data[pVerb])
p2.set_ylabel('Verb')

plt.show(block=True)
