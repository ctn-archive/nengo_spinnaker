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
    addr[2],
    addr[3],
    addr[0],
    addr[1],
    addr[4] << 28,
    kbits( addr[5] ),
    kbits( addr[6] )
  );
}

bool data_get_bias(
  address_t addr,
  uint n_neurons
){
  spin1_memcpy( gp_i_bias, addr,
    n_neurons * sizeof( accum ) );
  return true;
}

bool data_get_encoders(
  address_t addr,
  uint n_neurons,
  uint n_input_dimensions
){
  spin1_memcpy( gp_encoders, addr,
    n_neurons * n_input_dimensions * sizeof( accum ) );
  return true;
}

bool data_get_decoders(
  address_t addr,
  uint n_neurons,
  uint n_output_dimensions
){
  spin1_memcpy( gp_i_bias, addr,
    n_neurons * n_output_dimensions * sizeof( accum ) );
  return true;
}

bool data_get_keys(
  address_t addr,
  uint n_output_dimensions
){
  spin1_memcpy( gp_i_bias, addr,
    n_output_dimensions * sizeof( uint ) );
  return true;
}
