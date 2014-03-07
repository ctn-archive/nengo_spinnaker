/*
 * Ensemble - Data
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
 * 
 */

#include "ensemble.h"

bool data_system( address_t addr ) {
  initialise_ensemble(
    (region_system_t *) addr
  );
  return true;
}

bool data_get_bias(
  address_t addr,
  uint n_neurons
){
  spin1_memcpy( g_ensemble.i_bias, addr,
    n_neurons * sizeof( current_t ) );
  return true;
}

bool data_get_encoders(
  address_t addr,
  uint n_neurons,
  uint n_input_dimensions
){
  spin1_memcpy( g_ensemble.encoders, addr,
    n_neurons * n_input_dimensions * sizeof( value_t ) );
  return true;
}

bool data_get_decoders(
  address_t addr,
  uint n_neurons,
  uint n_output_dimensions
){
  spin1_memcpy( g_ensemble.decoders, addr,
    n_neurons * n_output_dimensions * sizeof( value_t ) );
  return true;
}

bool data_get_keys(
  address_t addr,
  uint n_output_dimensions
){
  spin1_memcpy( gp_output_keys, addr,
    n_output_dimensions * sizeof( uint ) );
  return true;
}
