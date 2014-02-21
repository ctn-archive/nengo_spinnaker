#include "spin-nengo-ensemble.h"

bool copy_in_system_region( uint *addr ){
  /* We have 7 parameters of a word each in the order:
   * 1. Number of input dimensions D_in
   * 2. Number of output dimensions D_out
   * 3. Number of neurons N
   * 4. dt in useconds
   * 5. tau_ref
   * 6. tau_rc
   * 7. Filter decay constant
   */
  addr += 20;
  spin1_memcpy( &n_input_dimensions, addr, sizeof( uint ) );
  addr++;
  spin1_memcpy( &n_output_dimensions, addr, sizeof( uint ) );
  addr++;
  spin1_memcpy( &n_neurons, addr, sizeof( uint ) );
  addr++;
  spin1_memcpy( &dt, addr, sizeof( uint ) );
  addr++;
  spin1_memcpy( &t_ref, addr, sizeof( uint ) );
  addr++;
  spin1_memcpy( &t_rc, addr, sizeof( value_t ) );
  addr++;
  spin1_memcpy( &filter, addr, sizeof( value_t ) );
  addr++;

  return true;
}

bool copy_in_bias( uint *addr_base ){
  /* Biases are 1xN, thus we just copy this area of memory. */
  addr_base += 20 + 7;
  spin1_memcpy( i_bias, addr_base, n_neurons * sizeof( accum ) );
  return true;
}

bool copy_in_encoders( uint *addr_base ){
  /* Encoders are a N x D_in matrix. */
  addr_base += 20 + 7 + n_neurons;
  spin1_memcpy( encoders, addr_base, n_neurons * n_input_dimensions * sizeof( accum ) );
  return true;
}

bool copy_in_decoders( uint *addr_base ){
  /* Decoders are a N x D_out matrix. */
  addr_base += 20 + 7 + n_neurons * (1 + n_input_dimensions);
  spin1_memcpy( decoders, addr_base, n_neurons * n_output_dimensions * sizeof( accum ) );
  return true;
}

bool copy_in_decoder_keys( uint *addr_base ){
  /* Biases are 1xN. */
  addr_base += 20 + 7 + n_neurons * (1 + n_input_dimensions + n_output_dimensions);
  spin1_memcpy( output_keys, addr_base, n_output_dimensions * sizeof( uint ) );
  return true;
}
