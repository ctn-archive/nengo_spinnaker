/*****************************************************************************

SpiNNaker and Nengo Integration

******************************************************************************

Authors:
 Andrew Mundy <mundya@cs.man.ac.uk> -- University of Manchester
 Terry Stewart			    -- University of Waterloo

Date:
 17-22 February, 3 March 2014

******************************************************************************

Advanced Processors Technologies,   Computational Neuroscience Research Group,
School of Computer Science,         Centre for Theoretical Neuroscience,
University of Manchester,           University of Waterloo,
Oxford Road,                        200 University Avenue West,
Manchester, M13 9PL,                Waterloo, ON, N2L 3G1,
United Kingdom                      Canada

*****************************************************************************/

#include "rx.h"

void timer_callback( uint simulation_time, uint none ) {
  // Output the current value
  spin1_send_mc_packet( keys[ n_current_output],
                        bitsk( values[ n_current_output ] ),
                        WITH_PAYLOAD
  );

  // Increment the current output counter
  n_current_output++;
  if( n_current_output >= n_dimensions ){
    n_current_output = 0;
  }
}
