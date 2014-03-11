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
