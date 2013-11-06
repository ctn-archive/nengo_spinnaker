import nengo
import numpy as np

model = nengo.Network('Communication Channel')

model.make_node('input', np.sin)
model.make_ensemble('A', neurons=100, dimensions=1)
model.make_ensemble('B', neurons=90, dimensions=1)
model.connect('input', 'A', filter=0.01)
model.connect('A', 'B', filter=0.01)




