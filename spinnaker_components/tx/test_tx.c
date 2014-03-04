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

#include "test_tx.h"

void c_main( void )
{
  // Load core map
  system_load_core_map( );

  // Set up routing tables
  if( leadAp ){
    io_printf( IO_STD, "TX leadAp = 0x%02x\n", leadAp );
    system_lead_app_configured( );
  }

  // Setup the mc_packet_received callback
  //spin1_set_timer_tick( 10000 );
  spin1_callback_on( MC_PACKET_RECEIVED, mc_packet_received, 0 );
  spin1_start( );
}

void mc_packet_received( uint key, uint payload )
{
  io_printf( IO_STD, "MC: 0x%08x, 0x%08x\n", key, payload );

  // Construct the message
  sdp_msg_t sdp_message;

  sdp_message.dest_addr = 0x00;
  sdp_message.dest_port = 0xff;
  sdp_message.srce_addr = sv->p2p_addr;
  // sdp_message.srce_port = virt_cpu;
  sdp_message.flags = 0x07;
  sdp_message.tag = 0x01;

  sdp_message.arg1 = key;
  sdp_message.arg2 = payload;

  sdp_message.length = sizeof (sdp_hdr_t) + sizeof (cmd_hdr_t);

  spin1_send_sdp_msg( &sdp_message, 1000 );
}
