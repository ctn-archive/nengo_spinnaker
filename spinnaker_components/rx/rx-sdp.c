/*****************************************************************************

SpiNNaker and Nengo Integration

******************************************************************************

Authors:
 Andrew Mundy <mundya@cs.man.ac.uk> -- University of Manchester
 Terry Stewart			    -- University of Waterloo

Date:
 3 March 2014

******************************************************************************

Advanced Processors Technologies,   Computational Neuroscience Research Group,
School of Computer Science,         Centre for Theoretical Neuroscience,
University of Manchester,           University of Waterloo,
Oxford Road,                        200 University Avenue West,
Manchester, M13 9PL,                Waterloo, ON, N2L 3G1,
United Kingdom                      Canada

*****************************************************************************/

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
