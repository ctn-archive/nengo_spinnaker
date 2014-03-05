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

void sdp_packet_received( uint mailbox, uint port ) {
  sdp_msg_t *msg = ( sdp_msg_t * ) mailbox;

  switch ( msg->cmd_rc ) {
    case 0x00000001:  // Replace dimensional values - TODO: Change this?
      spin1_memcpy( msg->data, values, n_dimensions * sizeof( value_t ) );
      spin1_msg_free( msg );
      break;
  };
}
