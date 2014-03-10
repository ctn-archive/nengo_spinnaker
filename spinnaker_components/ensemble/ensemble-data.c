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

bool data_get_filters( address_t addr, region_system_t *pars ) {
  // TODO: Be less hacky
  for( uint f = 0; f < pars->n_filters; f++ ){
    g_input.filters[f]->filter   = addr[2*f + 0];
    g_input.filters[f]->n_filter = addr[2*f + 1];
  }
  return true;
}

bool data_get_filter_keys( address_t addr, region_system_t *pars ) {
  spin1_memcpy( g_input.routes, addr,
    g_input.n_routes * sizeof( input_filter_key_t )
  );
  return true;
}
