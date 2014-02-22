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

#include "dimension_source.h"

uint key;

void copy_in_system_region( address_t addr ){
  key = (uint) addr[0];
  io_printf( IO_STD, "Rx Key 0x%08x\n", key );
}

int c_main( void )
{
  // Read in values
  address_t address = system_load_sram();
  copy_in_system_region( region_start( 1, address ) );

  // Routing and core map
  if( leadAp ){
    io_printf( IO_STD, "TX leadAp = 0x%02x\n", leadAp );
    system_lead_app_configured();
  }

  system_load_core_map();

  // Enable the timer tick callback
  spin1_set_timer_tick( 1000 ); // Timer tick / us
  spin1_callback_on( TIMER_TICK, timer_callback, 0 );

  // Go!
  spin1_start( );
}

void timer_callback( uint simulation_time, uint none )
{
  // Set some predefined values per dimension
  accum val = 0.5;
  spin1_send_mc_packet( key | 0x0, bitsk( val ), WITH_PAYLOAD );
  spin1_send_mc_packet( key | 0x1, bitsk( val ), WITH_PAYLOAD );
  spin1_send_mc_packet( key | 0x2, bitsk( val ), WITH_PAYLOAD );
}
