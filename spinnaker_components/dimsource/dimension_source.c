#include "dimension_source.h"

int c_main( void )
{
  // Enable the timer tick callback
  spin1_set_timer_tick( 1000 ); // Timer tick / ms
  spin1_callback_on( TIMER_TICK, timer_callback, 0 );

  // Broadcast sent packets to every core
  spin1_set_mc_table_entry( 0, 0x00000001, 0xffffffff, 0x00000100 );

  // Go!
  spin1_start( );
}

void timer_callback( uint simulation_time, uint none )
{
  // Set some predefined values per dimension
  accum val = 0.5;
  spin1_send_mc_packet( 0x00000001, simulation_time, 1 );
}
