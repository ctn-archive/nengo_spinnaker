/*****************************************************************************

SpiNNaker and Nengo Integration

******************************************************************************

Authors:
 Andrew Mundy <mundya@cs.man.ac.uk> -- University of Manchester
 Terry Stewart			    -- University of Waterloo

Date:
 17-22 February 2014

******************************************************************************

Advanced Processors Technologies,   Computational Neuroscience Research Group,
School of Computer Science,         Centre for Theoretical Neuroscience,
University of Manchester,           University of Waterloo,
Oxford Road,                        200 University Avenue West,
Manchester, M13 9PL,                Waterloo, ON, N2L 3G1,
United Kingdom                      Canada

*****************************************************************************/

#include "spin-nengo-ensemble.h"

void incoming_spike_callback( uint key, uint payload )
{
  /*
   * - Determine the dimension this packet relates to from the key
   * - Add the value of the payload (cast to accum) to the received value for
   *   that dimension.
   */
  uint dimension = key & 0x0000000f;
  ibuf_accumulator[ dimension ] += kbits( payload );
}

/*
 * Possible TODO:
 * - On timer2 (if possible) transmit the decoded value for a dimension.
 */
