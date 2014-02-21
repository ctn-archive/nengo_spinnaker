#include "spin-nengo-ensemble.h"

uint n_input_dimensions, n_output_dimensions, n_neurons, dt, t_ref,
     *v_ref_voltage, *output_keys;
current_t *i_bias;
accum *encoders, *decoders;
value_t *ibuf_accumulator, *ibuf_filtered, *output_values, t_rc, filter,
        *decoded_values;

int c_main( void )
{
  /* Initialise system as in the standard harness, see JK re correctly
   * integrating this code at some point. (We don't use delay buffers,
   * have dimension buffers, etc.)
   *
   *   - Setup routing table entries
   *   - Setup timer and callbacks
   *   - Any work to move neuron parameters into the correct locations.
   */

  // Setup callbacks, etc.
  spin1_callback_on( MC_PACKET_RECEIVED, incoming_spike_callback, -1 );
  spin1_callback_on( TIMER_TICK, timer_callback, 2 );
  io_printf( IO_STD, "Testing...\n" );

  // Setup buffers, etc.
  // test_initialise( );
  uint *address = (uint *) system_load_sram();
  copy_in_system_region( address );
  initialise_buffers( );
  copy_in_bias( address );
  copy_in_encoders( address );
  copy_in_decoders( address );
  copy_in_decoder_keys( address );
  io_printf( IO_STD, "N: %d, D_in: %d, D_out: %d, dt: %d, t_rc: %f,"
             " t_ref: %d steps, filter: %f\n",
             n_neurons, n_input_dimensions, n_output_dimensions, dt,
             t_rc, t_ref, filter
  );
  
  spin1_set_mc_table_entry(0, 0, 0xFFFFFFE0, 0x00000100);

  // Setup timer tick, start
  spin1_set_timer_tick( dt );
  spin1_start( );
}

//! Initialise buffers and values
void initialise_buffers( void )
{
  // Encoders / decoders / bias
  encoders = spin1_malloc( sizeof(accum) * n_input_dimensions * n_neurons );
  decoders = spin1_malloc( sizeof(accum) * n_output_dimensions * n_neurons );
  output_keys = spin1_malloc( sizeof( uint ) * n_output_dimensions );
  i_bias = spin1_malloc( sizeof(current_t) * n_neurons );

  // Input buffers / voltages
  ibuf_accumulator = spin1_malloc( sizeof(value_t) * n_input_dimensions );
  ibuf_filtered = spin1_malloc( sizeof(value_t) * n_input_dimensions );
  v_ref_voltage = spin1_malloc( sizeof(uint) * n_neurons );

  // Output buffers
  decoded_values = spin1_malloc( sizeof(value_t) * n_output_dimensions );
}
