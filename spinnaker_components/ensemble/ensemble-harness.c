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

uint n_input_dimensions, n_output_dimensions, n_neurons, dt, t_ref,
     *v_ref_voltage, *output_keys;
current_t *i_bias;
accum *encoders, *decoders;
value_t *output_values, one_over_t_rc, filter, *decoded_values, n_filter;

filtered_input_buffer_t *in_buff;

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
  address_t address = system_load_sram();
  copy_in_system_region( region_start( 1, address ) );
  initialise_buffers( );
  copy_in_bias         ( region_start( 2, address ) );
  copy_in_encoders     ( region_start( 3, address ) );
  copy_in_decoders     ( region_start( 4, address ) );
  copy_in_decoder_keys ( region_start( 5, address ) );

  io_printf( IO_STD, "N: %d, D_in: %d, D_out: %d, dt: %d, one_over_t_rc: %f,"
             " t_ref: %d steps, filter: %f\n",
             n_neurons, n_input_dimensions, n_output_dimensions, dt,
             one_over_t_rc, t_ref >> 28, filter
  );
  
  // Set up routing tables
  if( leadAp ){
    io_printf( IO_STD, "ENS leadAp = 0x%02x\n", leadAp );
    system_lead_app_configured( );
  }

  // Load core map
  system_load_core_map( );

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
  in_buff = input_buffer_initialise( n_input_dimensions );
  in_buff->filter = filter;
  in_buff->n_filter = n_filter;
  v_ref_voltage = spin1_malloc( sizeof(uint) * n_neurons );

  // Output buffers
  decoded_values = spin1_malloc( sizeof(value_t) * n_output_dimensions );
}
