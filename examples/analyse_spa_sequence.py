import logging
import nengo
import nengo.spa as spa
import nengo_spinnaker

logging.basicConfig(level=logging.DEBUG)

# Number of dimensions for the Semantic Pointers
dimensions = 16

# Make a model object with the SPA network
model = spa.SPA(label='Routed_Sequence')

with model:
    # Specify the modules to be used
    model.cortex = spa.Buffer(dimensions=dimensions)
    model.vision = spa.Buffer(dimensions=dimensions) 
    # Specify the action mapping
    actions = spa.Actions(
        'dot(vision, START) --> cortex = vision',
        'dot(cortex, A) --> cortex = B',
        'dot(cortex, B) --> cortex = C',
        'dot(cortex, C) --> cortex = D',
        'dot(cortex, D) --> cortex = E',
        'dot(cortex, E) --> cortex = A'
    )
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg)

with model:
    #Probe things that will be plotted
    # TODO Add probing back...
    # pActions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    # pUtility = nengo.Probe(model.bg.input, synapse=None)
    pass

analyser = nengo_spinnaker.Analyser(model)
analyser.create_hypergraph_as_json("ss.json")
