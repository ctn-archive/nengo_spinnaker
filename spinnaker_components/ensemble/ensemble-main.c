#include "ensemble.h"

void c_main( void ) {
  // Set the system up
  address_t address = system_load_sram();
  data_system       ( region_start( 1, address ) );
  data_get_bias     ( region_start( 2, address ), g_ensemble.n_neurons );
  data_get_encoders ( region_start( 3, address ), g_ensemble.n_neurons, g_n_input_dimensions );
  data_get_decoders ( region_start( 4, address ), g_ensemble.n_neurons, g_n_output_dimensions );
  data_get_keys     ( region_start( 5, address ), g_n_output_dimensions );

  // Set up routing tables
  if( leadAp ){
    system_lead_app_configured( );
  }

  // Load core map
  system_load_core_map( );

  // Setup timer tick, start
  spin1_set_timer_tick( g_ensemble.machine_timestep );
  spin1_start( );
}
