#include "test_tx.h"

void c_main( void )
{
  // Setup the mc_packet_received callback
  //spin1_set_timer_tick( 10000 );
  spin1_callback_on( MC_PACKET_RECEIVED, mc_packet_received, 0 );
  spin1_start( );
}

void mc_packet_received( uint key, uint payload )
{
  io_printf( IO_STD, "MC: 0x%08x, %d\n", key, payload );

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

  sdp_message.length = sizeof (sdp_hdr_t) + sizeof (sdp_hdr_t);

  spin1_send_sdp_msg( &sdp_message, 1000 );
}
