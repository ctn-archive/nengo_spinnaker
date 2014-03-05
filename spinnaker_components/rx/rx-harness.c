/*
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

#include "rx.h"

int c_main( void )
{
  // Read in values
  address_t address = system_load_sram();
  copy_in_system_region ( region_start( 1, address ) );
  copy_in_keys          ( region_start( 2, address ) );
  copy_in_initial_values( region_start( 3, address ) );

  // Routing and core map
  if( leadAp ){
    system_lead_app_configured();
  }

  system_load_core_map();

  // Enable callbacks
  spin1_set_timer_tick( ticks_per_output );
  spin1_callback_on( TIMER_TICK, timer_callback, 0 );
  spin1_callback_on( SDP_PACKET_RX, sdp_packet_received, -2 );

  // Go!
  spin1_start( );
}
