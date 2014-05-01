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

  for( uint n = 0; n < n_neurons; n++ ){
    io_printf( IO_BUF, "Decoder[%d] = ", n );
    for( uint d = 0; d < n_output_dimensions; d++ ){
      io_printf( IO_BUF, "%k, ", neuron_decoder( n, d ) );
    }
    io_printf( IO_BUF, "\n" );
  }

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
    g_input.filters[f]->filter = kbits(addr[3*f + 0]);
    g_input.filters[f]->n_filter = kbits(addr[3*f + 1]);
    g_input.filters[f]->mask = addr[3*f + 2];
    g_input.filters[f]->mask_ = ~(addr[3*f + 2]);

    io_printf(IO_BUF, "Filter[%d] = %k, %k, MASK=0x%08x\n", f,
      g_input.filters[f]->filter,
      g_input.filters[f]->n_filter,
      g_input.filters[f]->mask
    );
  }
  return true;
}

bool data_get_filter_keys( address_t addr, region_system_t *pars ) {
    io_printf(IO_BUF, "initializing %d filter keys\n", g_input.n_routes);

  spin1_memcpy( g_input.routes, addr,
    g_input.n_routes * sizeof( input_filter_key_t )
  );
  for (int i = 0; i < g_input.n_routes; i++) {
    io_printf(IO_BUF, "FilterKey[%d] = 0x%08x, 0x%08x => %d\n", i,
              g_input.routes[i].key, g_input.routes[i].mask,
              g_input.routes[i].filter);
  }
  return true;
}
