import numpy as np
import logging
import warnings
logger = logging.getLogger(__name__)

import nengo
import nengo.utils.builder

from . import fixpoint as fp
from . import vertices


class SpiNNakerProbe(object):
    """A NengoProbe encapsulates the logic required to retrieve data from a
    SpiNNaker machine.
    """
    def __init__(self, probe, dt=0.001):
        self.probe = probe
        self.dt = dt

    def get_data(self, txrx):
        raise NotImplementedError


class DecodedValueProbe(SpiNNakerProbe):
    def __init__(self, recording_vertex, probe):
        super(DecodedValueProbe, self).__init__(probe)
        self.recording_vertex = recording_vertex

    def get_data(self, txrx):
        # For only 1 subvertex, get the recorded data
        assert(len(self.recording_vertex.subvertices) == 1)
        sv = self.recording_vertex.subvertices[0]
        (x, y, p) = sv.placement.processor.get_coordinates()

        size = self.recording_vertex.regions[
            self.recording_vertex.recording_region_index-1].sizeof(
                sv.lo_atom, sv.hi_atom)

        sdata = vertices.retrieve_region_data(
            txrx, x, y, p, self.recording_vertex.recording_region_index,
            size
        )

        # Cast as a Numpy array, shape and return
        data = np.array(fp.kbits([int(i) for i in
                                  np.fromstring(sdata, dtype=np.uint32)]))
        return data.reshape((self.recording_vertex.run_ticks,
                             self.recording_vertex.width))


try:
    from bitarray import bitarray

    class SpikeProbe(SpiNNakerProbe):
        def __init__(self, target_vertex, probe):
            super(SpikeProbe, self).__init__(probe)
            self.target_vertex = target_vertex

        def get_data(self, txrx):
            # Calculate the number of frames
            n_frames = int(self.target_vertex.runtime * 1000)  # TODO Neaten!
            data = [list() for n in range(n_frames)]

            for subvertex in self.target_vertex.subvertices:
                # Get the contents of the "SPIKES" region for each subvertex
                (x, y, p) = subvertex.placement.processor.get_coordinates()

                size = self.target_vertex.regions[
                    self.target_vertex.spikes_recording_region-1].sizeof(
                        subvertex.lo_atom, subvertex.hi_atom)

                sdata = vertices.retrieve_region_data(
                    txrx, x, y, p, self.target_vertex.spikes_recording_region,
                    size)

                # Cast the spikes as a bit array
                spikes = bitarray()
                spikes.frombytes(sdata)

                # Convert each frame into a list of spiked neurons
                frame_length = ((subvertex.n_atoms >> 5) +
                                (1 if subvertex.n_atoms & 0x1f else 0))

                for f in range(n_frames-1):  # TODO Understand the -1
                    frame = spikes[32*f*frame_length + 32:
                                   32*(f + 1)*frame_length + 32]
                    data[f].extend([n + subvertex.lo_atom for n in
                                    range(subvertex.n_atoms) if frame[n]])

            # Convert into list of spike times
            spikes = [[0.] for n in range(self.probe.target.n_neurons)]
            for (i, f) in enumerate(data):
                for n in f:
                    spikes[n].append(i*self.dt)

            return spikes

except ImportError:
    # No bitarray, so no spike probing!
    warnings.warn("Cannot import module bitarray: spike probing is disabled",
                  ImportWarning)

    class SpikeProbe(object):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Spike probing requires the module "
                                      "'bitarray' to be installed.")
