import collections
import itertools
import logging
import numpy as np
import os.path

from .regions import Subregion

logger = logging.getLogger(__name__)


MemoryRegionWrite = collections.namedtuple('MemoryRegionWrite',
                                           'x y base_address size_bytes path')
MemoryWordWrite = collections.namedtuple('MemoryWordWrite',
                                         'x y address n_bytes value')


class Error(Exception):
    pass


class InsufficientMemoryError(Error):
    def __init__(self, mem_req, mem_available):
        self.mem_req = mem_req
        self.mem_available = mem_available

    def __str__(self):
        return "Attempted to request {}B of memory, only {}B are available."\
               .format(self.mem_req, self.mem_available)


def get_region_offsets(subregions):
    """Get the address offsets for the given list of subregions.

    :returns: Ordered dictionary of subregion to offset of that subregion.
    """
    offsets = reduce(
        lambda addrs, sr: addrs + [addrs[-1] + (0 if sr is None else
                                                sr.size_words * 4)],
        subregions[:-1], [0])

    return collections.OrderedDict(zip(subregions, offsets))


def create_app_pointer_table_region(subregions, magic_num=0xAD130AD6,
                                    version=0x00010000,  # REDUNDANT?
                                    timer_period=1000):
    """Create a subregion to act as a application pointer table."""
    # Create the basic application pointer table
    app_pointer = [magic_num, version, timer_period, 0x0, ]

    # Add the offset (bytes) for each of the given regions, including the
    # offset caused by the application pointer table.
    size = (len(app_pointer) + len(subregions))
    app_pointer.extend(v + size * 4 for (k, v) in
                       get_region_offsets(subregions).iteritems())

    # Convert the data to a Numpy array and return the new subregion
    return Subregion(np.array(app_pointer, dtype=np.uint32), size, False)


def get_total_bytes_used(subregions):
    return sum(sr.size_words for sr in subregions) * 4


def write_core_region_files(x, y, p, subregions, base_addr, tmpdir):
    """Create the files for the data to write to memory and return a list of
    data to write.
    """
    # Create the file for each written region
    writes = list()
    for i, subregion in enumerate(subregions):
        if subregion is None:
            continue

        if not subregion.unfilled:
            path = os.path.join(
                tmpdir, '{:03d}_{:03d}_{:02d}_{:03d}.bin'.format(x, y, p, i))

            with open(path, 'wb+') as f:
                f.write(bytearray(subregion.data))

            logger.debug("Writing %s" % path)
            writes.append(MemoryRegionWrite(x, y, base_addr,
                                            subregion.size_words * 4, path))

        # Update the base address
        base_addr += subregion.size_words * 4

    return writes


def generate_data_for_placements(placed_vertices, sdram_empty_mem, sdram_base,
                                 get_user_0_register_from_core_func, tmpdir):
    """Generates the data files and writes required to load data for the given
    placements.

    :param placed_vertices: Placed vertex objects.
    """
    # Store all the regions and words to write
    region_writes = list()
    word_writes = list()

    # Sort the placements by chip and core number
    placed_vertices = sorted(placed_vertices,
                             key=lambda p: (p.x*256 + p.y)*18 + p.p)

    # Split by core number
    grouped_ps = itertools.groupby(placed_vertices, key=lambda p: (p.x, p.y))
    for ((x, y), ps) in grouped_ps:
        # Keep track of the amount of memory we've used
        memory = sdram_empty_mem

        # Iterate through the placements for this (x, y) and generate all data
        # as required.
        for placed_vertex in ps:
            # Generate the complete list of regions for this subvertex
            regions = [create_app_pointer_table_region(
                placed_vertex.subregions,
                timer_period=placed_vertex.timer_period)
            ]
            regions.extend(list(placed_vertex.subregions))

            # Get the number of bytes used by the subregions
            mem_used = get_total_bytes_used(regions)

            # Determine where this set of regions will be written in SDRAM
            if memory - mem_used < 0:
                raise InsufficientMemoryError(mem_used, memory)
            memory -= mem_used
            base = memory + sdram_base

            # Generate the set of region writes for these regions
            region_writes.extend(
                write_core_region_files(placed_vertex.x, placed_vertex.y,
                                        placed_vertex.p, regions, base, tmpdir)
            )

            # Add a write to indicate the starting address of the app_ptr table
            ba = get_user_0_register_from_core_func(placed_vertex.x,
                                                    placed_vertex.y,
                                                    placed_vertex.p)
            word_writes.append(MemoryWordWrite(placed_vertex.x,
                                               placed_vertex.y, ba, 4, base))

    return word_writes, region_writes
