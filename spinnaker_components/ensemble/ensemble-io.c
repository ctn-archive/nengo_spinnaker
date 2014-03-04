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

/**
 * \brief Transmit the value associated with one dimension.
 * \param index Dimension to transmit a value for.
 * \param arg1 Unused
 *
 * Timer2 is used to ensure that transmission of outgoing MC packets is
 * regular over time, so as to avoid overloading the network.
 *
 * At each interval the value for one dimension is transmitted, and
 * transmission of the next is scheduled.
 */
void outgoing_dimension_callback( uint index, uint arg1 ) {
  // Transmit the packet with the appropriate key
  spin1_send_mc_packet(
    output_keys[ index ],
    bitsk(output_values[ index ]),
    WITH_PAYLOAD
  );

  // Zero the output buffer and increment the output dimension counter
  output_values[ index ] = 0;
  index++;
  
  if( index >= n_output_dimensions ) {
    index = 0;
  }

  // Schedule this function to be called again
  if( !timer_schedule_proc( (event_proc) outgoing_dimension_callback, index, 0, us_per_output ) ) {
    io_printf( IO_BUF, "[Timer2] [ERROR] Failed to schedule next.\n" );
  }
}
