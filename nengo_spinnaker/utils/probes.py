from bitarray import bitarray
import numpy as np

from . import vertices
from . import fixpoint as fp


class SpiNNakerProbe(object):
    """A NengoProbe encapsulates the logic required to retrieve data from a
    SpiNNaker machine.
    """
    def __init__(self, target_vertex, probe):
        self.target_vertex = target_vertex
        self.probe = probe

    def get_data(self, txrx):
        raise NotImplementedError


class DecodedValueProbe(SpiNNakerProbe):
    def __init__(self, target_vertex, recording_vertex, probe):
        super(DecodedValueProbe, self).__init__(target_vertex, probe)
        self.recording_vertex = recording_vertex

    def get_data(self, txrx):
        # For only 1 subvertex, get the recorded data
        assert(len(self.recording_vertex.subvertices) == 1)
        sv = self.recording_vertex.subvertices[0]
        (x, y, p) = sv.placement.processor.get_coordinates()

        sdata = vertices.retrieve_region_data(
            txrx, x, y, p, self.recording_vertex.REGIONS['VALUES'],
            self.recording_vertex.sizeof_values(sv.n_atoms)
        )

        # Cast as a Numpy array, shape and return
        data = np.array(fp.kbits([int(i) for i in
                                  np.fromstring(sdata, dtype=np.uint32)]))
        return data.reshape((self.recording_vertex.width,
                             self.recording_vertex.run_ticks))


class SpikeProbe(SpiNNakerProbe):
    def get_data(self, txrx):
        # Calculate the number of frames
        n_frames = int(self.target_vertex.runtime * 1000)  # TODO Neaten!
        data = [list() for n in range(n_frames)]

        for subvertex in self.target_vertex.subvertices:
            # Get the contents of the "SPIKES" region for each subvertex
            (x, y, p) = subvertex.placement.processor.get_coordinates()

            sdata = vertices.retrieve_region_data(
                txrx, x, y, p, self.target_vertex.REGIONS["SPIKES"],
                self.target_vertex.sizeof_region_recording(subvertex.n_atoms)
            )

            # Cast the spikes as a bit array
            spikes = bitarray()
            spikes.frombytes(sdata)

            # Convert each frame into a list of spiked neurons
            frame_length = ((subvertex.n_atoms >> 5) +
                            (1 if subvertex.n_atoms & 0x1f else 0))

            for f in range(n_frames):
                frame = spikes[32*f*frame_length + 32:
                               32*(f + 1)*frame_length + 32]
                data[f].extend([n + subvertex.lo_atom for n in
                                range(subvertex.n_atoms) if frame[n]])

        return data
