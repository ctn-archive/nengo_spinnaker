/*
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 */

#include "spin-nengo-ensemble.h"

/**
 * \brief Handle an incoming "spike" or dimension.
 * \param key Multicast key associated with the "spike"
 * \param payload Partial value of the dimension to be accumulated
 *
 * Each arriving multicast packet contains a part of the value for a given
 * dimension for the given timestep.  On receipt of a packet the input
 * dimension referred to is taken from the bottom nibble of the key and the
 * value of the payload is added to the accumulator for this dimension.
 */
void incoming_spike_callback( uint key, uint payload )
{
  uint dimension = key & 0x0000000f;
  ibuf_accumulator[ dimension ] += kbits( payload );
}
