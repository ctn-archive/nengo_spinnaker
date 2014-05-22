from bitarray import bitarray

from . import vertices


class SpiNNakerProbe(object):
    """A NengoProbe encapsulates the logic required to retrieve data from a
    SpiNNaker machine.
    """
    def __init__(self, target_vertex, probe):
        self.target_vertex = target_vertex
        self.probe = probe

    def get_data(self, txrx):
        raise NotImplementedError


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
