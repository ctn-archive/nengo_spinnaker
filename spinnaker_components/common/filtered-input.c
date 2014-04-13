/*
 * Filtered Input
 * --------------
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

#include "filtered-input.h"

filtered_input_t g_input;

value_t* initialise_input(uint n_filters, uint n_input_dimensions) {
  // Value preparation
  g_input.n_filters = n_filters;
  g_input.n_dimensions = n_input_dimensions;

  // Buffer initialisation
  g_input.filters = spin1_malloc(
    g_input.n_filters * sizeof( filtered_input_buffer_t* )
  );
  for( uint f = 0; f < g_input.n_filters; f++ ) {
    g_input.filters[f] = input_buffer_initialise( g_input.n_dimensions );
    g_input.filters[f]->filter = 0;   // Initialised later
    g_input.filters[f]->n_filter = 0; // Initialised later
    g_input.filters[f]->mask = 0;     // Initialised later
  };

  g_input.input = spin1_malloc( g_input.n_dimensions * sizeof( value_t ) );

  // Routes initialisation
  g_input.routes = spin1_malloc(
    g_input.n_routes * sizeof( input_filter_key_t )
  );
  for( uint r = 0; r < g_input.n_routes; r++ ) {
    g_input.routes[r].key    = 0x00000000;  // Initialised later
    g_input.routes[r].mask   = 0x00000000;  // Initialised later
    g_input.routes[r].filter = 0x00000000;  // Initialised later
  }

  // Set up the multicast callback
  spin1_callback_on(
    MCPL_PACKET_RECEIVED, incoming_dimension_value_callback, -1
  );

  // Return the input (to the encoders) buffer
  return g_input.input;
}

// Incoming spike callback
void incoming_dimension_value_callback( uint key, uint payload ) {
  uint dimension = key & 0x0000000f;

  /*
   * 1. Look up key in input routing table entry
   * 2. Select appropriate filter
   * 3. Add value (payload) to appropriate dimension of given filter.
   */
  input_buffer_acc(input_filter(key), dimension, payload);
}

// Input step
void input_filter_step( void ) {
  // Zero the input accumulator
  for( uint d = 0; d < g_input.n_dimensions; d++ ) {
    g_input.input[d] = 0x00000000;
  }

  // For each filter, apply filtering and accumulate the value in the global
  // input accumulator.
  for( uint f = 0; f < g_input.n_filters; f++ ) {
    input_buffer_step( g_input.filters[f] );

    for( uint d = 0; d < g_input.n_dimensions; d++ ) {
      g_input.input[d] += g_input.filters[f]->filtered[d];
    }
  }
}
