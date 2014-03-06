/*
 * Ensemble - Input
 * ----------------
 * Structures and functions to deal with arriving multicast packets (input).
 *
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

#include "ensemble-input.h"

// Globals
uint g_n_input_dimensions;
value_t *gp_ibuf_accumulator, *gp_ibuf_filtered, g_filter;

void initialise_input( uint n, value_t f ) {
  // Value preparation
  g_n_input_dimensions = n;

  // Buffer initialisation
  gp_ibuf_accumulator = spin1_malloc( n * sizeof( value_t ) );
  gp_ibuf_filtered = spin1_malloc( n * sizeof( value_t ) );

  // Set up the multicast callback
  spin1_callback_on(
    MC_PACKET_RECEIVED, incoming_dimension_value_callback, -1
  );
}

// Incoming spike callback
void incoming_dimension_value_callback( uint key, uint payload )
{
  uint dimension = key & 0x0000000f;
  gp_ibuf_accumulator[ dimension ] += kbits( payload );
}
