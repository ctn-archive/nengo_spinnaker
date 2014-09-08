import collections
import numpy as np
import struct

from . import vertices

MS_SCALE = (1.0 / 200032.4)

def get_profile_data(txrx, profiled_vertices):
    """Build a dictionary of object->vertices->subvertices->profile data.
    """
    profile_data = collections.defaultdict(
        lambda: collections.defaultdict(dict))

    for (obj, vertices) in profiled_vertices.iteritems():
        for v in vertices:
            for subvertex in v.subvertices:
                profile_data[obj][v][subvertex] =\
                    get_subvertex_profile_data(txrx, subvertex)

    return profile_data


def get_subvertex_profile_data(txrx, subvertex):
    """Get profiling data from the given subvertex."""
    # Get subvertex's placement
    (x, y, p) = subvertex.placement.processor.get_coordinates()

    # **HACK** this is only correct for ensemble
    profile_region_index = 14

    # Read number of words of profile data written from 1st word of region
    words_written_data = vertices.retrieve_region_data(
            txrx, x, y, p, profile_region_index, 1)
    words_written = struct.unpack_from("<I", words_written_data)[0]
   
    # Now read entire region
    profiling_data = vertices.retrieve_region_data(
            txrx, x, y, p, profile_region_index, words_written + 1)
    
    # Slice off 1st word (number of words written) and convert the 
    # Remainder of the data into a numpy array of uint64 samples
    profiling_samples = np.fromstring(profiling_data[4:], dtype=np.uint64)
    
    # Return profiling samples
    return profiling_samples

def plot_profile_data(data):
    # Create 32-bit view of data and slice this to seperate times, tags and flags
    data_view = data.view(np.uint32)
    sample_times = data_view[::2]
    sample_tags_and_flags = data_view[1::2]
    
    # Further split the tags and flags word into seperate arrays of tags and flags
    sample_tags = np.bitwise_and(sample_tags_and_flags, 0x7FFFFFFF)
    sample_flags = np.right_shift(sample_tags_and_flags, 31)
    
    # Find indices of samples relating to entries and exits
    sample_entry_indices = np.where(sample_flags == 1)
    sample_exit_indices = np.where(sample_flags == 0)
    
    # Convert count-down times to count up times from 1st sample
    sample_times = np.subtract(sample_times[0], sample_times)
    sample_times_ms = np.multiply(sample_times, MS_SCALE, dtype=np.float)

    # Slice tags and times into entry and exits
    entry_tags = sample_tags[sample_entry_indices]
    entry_times_ms = sample_times_ms[sample_entry_indices]
    exit_tags = sample_tags[sample_exit_indices]
    exit_times_ms = sample_times_ms[sample_exit_indices]

    # Loop through unique tags
    tag_dictionary = dict()
    unique_tags = np.unique(sample_tags)
    for tag in unique_tags:
        # Get indices where these tags occur
        tag_entry_indices = np.where(entry_tags == tag)
        tag_exit_indices = np.where(exit_tags == tag)
        
        # Use these to get subset for this tag
        tag_entry_times_ms = entry_times_ms[tag_entry_indices]
        tag_exit_times_ms = exit_times_ms[tag_exit_indices]
        
        # If the first exit is before the first
        # Entry, add a dummy entry at beginning
        if tag_exit_times_ms[0] < tag_entry_times_ms[0]:
            print "WARNING: profile starts mid-tag"
            tag_entry_times_ms = np.append(0.0, tag_entry_times_ms)
        
        if len(tag_entry_times_ms) > len(tag_exit_times_ms):
            print "WARNING: profile finishes mid-tag"
            tag_entry_times_ms = tag_entry_times_ms[:len(tag_exit_times_ms)-len(tag_entry_times_ms)]
 
        # Subtract entry times from exit times to get durations of each call
        tag_durations_ms = np.subtract(tag_exit_times_ms, tag_entry_times_ms)
        
        # Add entry times and durations to dictionary
        tag_dictionary[tag] = (tag_entry_times_ms, tag_durations_ms)
        
    return tag_dictionary