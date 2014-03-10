from pacman103.lib import graph
from pacman103.lib import data_spec_gen
from pacman103.lib import lib_map
from pacman103.front.common import enums
import os
from pacman103.lib import parameters
import numpy as np

REGIONS = enums.enum1(
    'SYSTEM',
    'BIAS',
    'ENCODERS',
    'DECODERS',
    'DECODER_KEYS',
    'FILTERS',
    'FILTER_ROUTING'
    )


class EnsembleVertex(graph.Vertex):
    def __init__(self, data, constraints=None):
        super(EnsembleVertex, self).__init__(data.N, constraints=constraints,
                                             label=data.label)
        self.data = data

    def model_name(self):
        return 'nengo_ensemble'

    def sizeof_filters_region(self):
        # 2 words per filter
        return 4 * 2 * len(self.data.filters)

    def sizeof_filter_keys_region(self, subvertex):
        # 3 words per entry
        # 1 entry per in_subedge
        return 4 * 3 * len(in_subedges)
    
    def get_requirements_per_atom(self):
        chip_memory = 4 + self.data.D_in*4 + self.data.D_out*4
        data_memory = chip_memory
        cycles = 1

        return lib_map.Resources(cycles, data_memory, chip_memory)

    def generateDataSpec(self, processor, subvertex, dao):
        IDENTIFIER = 0xABCD
        print('generate', self.data.label)

        spec = data_spec_gen.DataSpec(processor, dao)
        spec.initialise(IDENTIFIER, dao)
        spec.comment('NEF Ensemble vertex information')

        x, y, p = processor.get_coordinates()
        executableTarget = lib_map.ExecutableTarget(
            dao.get_binaries_directory() + os.sep
            + 'nengo_ensemble.aplx', x, y, p)

        # size is measured in bytes
        spec.reserveMemRegion(REGIONS.SYSTEM, size=8*4)
        spec.reserveMemRegion(REGIONS.BIAS,
                              size=subvertex.n_neurons*4)
        spec.reserveMemRegion(REGIONS.ENCODERS,
                              size=subvertex.n_neurons*self.data.D_in*4)
        spec.reserveMemRegion(REGIONS.DECODERS,
                              size=subvertex.n_neurons*self.data.D_out*4)
        spec.reserveMemRegion(REGIONS.DECODER_KEYS,
                              size=self.data.D_out*4)
        spec.reserveMemRegion(REGIONS.FILTERS,
                              size=self.sizeof_filters_region())
        spec.reserveMemRegion(REGIONS.FILTER_KEYS,
                              size=self.sizeof_filter_keys_region(subvertex))

        #TODO: adjust time resolution of sim
        #TODO: adjust realtime rate
        dt = 0.001
        spec.switchWriteFocus(REGIONS.SYSTEM)
        spec.write(data=self.data.D_in)
        spec.write(data=self.data.D_out)
        spec.write(data=subvertex.n_neurons)
        spec.write(data=int(dt*(10**6)))   # dt in us
        spec.write(data=int(round(self.data.tau_ref/0.001)))  # t_ref in steps

        # 1/tau_rc in 1/seconds
        spec.write(data=parameters.S1615(1. / self.data.tau_rc).converted)
        
        # Number of filters, number of routing elements for filters
        # spec.write(data=...)
        # spec.write(data=...)
                
        spec.switchWriteFocus(REGIONS.BIAS)
        spec.comment("# *** Bias Currents, including any constant inputs. ***")
        # Encode any constant inputs, and add to the biases
        additional_bias = np.dot(
            self.data.encoders,
            self.data.constant_input
        )
        self.data.bias += additional_bias

        # Write the bias currents
        for i in range(subvertex.lo_atom, subvertex.hi_atom):
            spec.write(data=parameters.S1615(self.data.bias[i]).converted)

        spec.switchWriteFocus(REGIONS.ENCODERS)
        for i in range(subvertex.lo_atom, subvertex.hi_atom):
            for j in range(self.data.D_in):
                spec.write(
                    data=parameters.S1615(
                        self.data.encoders[i, j] * self.data.gain[i]
                    ).converted
                )

        decoders = self.data.get_merged_decoders()
        spec.switchWriteFocus(REGIONS.DECODERS)
        for i in range(subvertex.lo_atom, subvertex.hi_atom):
            for j in range(self.data.D_out):
                spec.write(data=parameters.S1615(decoders[i, j]/dt).converted)

        spec.switchWriteFocus(REGIONS.DECODER_KEYS)
        index = 0
        for d, t, f in self.data.decoder_list:
            for i in range(d.shape[1]):
                spec.write(
                    data=((x << 24) | (y << 16) | ((p-1) << 11)
                          | (index << 6) | (i))
                )
            index += 1

        # Write the filter parameters
        spec.switchWriteFocus(REGIONS.FILTERS)
        spec.comment("# *** Filter Parameters. ***")
        for f in self.data.filters:
            decay = np.exp(-dt/f)
            spec.write(data=parameters.S1615(decay).converted)
            spec.write(data=parameters.S1615(1 - decay).converted)

        # Write the filter routing entries
        spec.switchWriteFocus(REGIONS.FILTER_ROUTING)
        spec.comment("# *** Filter Routing Keys and Masks. ***")
        """
        For each incoming subedge we write the key, mask and index of the
        filter to which it is connected.  At some later point we can try
        to combine keys and masks to minimise the number of comparisons
        which are made in the SpiNNaker application.
        """

        # End the writing of this specification:
        spec.endSpec()
        spec.closeSpecFile()

        # No memory writes required for this Data Spec:
        memoryWriteTargets = list()
        loadTargets = list()

        # Return list of target cores, executables, files to load and
        # memory writes to perform:
        return executableTarget, loadTargets, memoryWriteTargets

    def generate_routing_info(self, subedge):
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (subedge.edge.index << 6)
        mask = 0xFFFFFFE0

        return key, mask
