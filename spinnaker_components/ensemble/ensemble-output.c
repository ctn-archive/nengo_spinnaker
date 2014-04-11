/*
 * Ensemble - Output
 * -----------------
 * Structures and functions to deal with arriving multicast packets (input).
 *
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 * 
 */

#include "ensemble-output.h"

uint g_n_output_dimensions, *gp_output_keys, g_us_per_output;
value_t * gp_output_values;

// Initialise everything necessary for the output system
value_t* initialise_output( region_system_t *pars ){
  io_printf( IO_BUF, "[Ensemble] INITIALISE_OUTPUT.\n" );
  // Store globals, initialise arrays
  g_n_output_dimensions = pars->n_output_dimensions;
  gp_output_values = spin1_malloc(
    pars->n_output_dimensions * sizeof( value_t )
  );
  gp_output_keys = spin1_malloc(
    pars->n_output_dimensions * sizeof( uint )
  );

  io_printf( IO_BUF, "[Ensemble] n_output_dimensions = %d\n",
    g_n_output_dimensions
  );

  // Calculate the number of microseconds between transmitting output packets
  g_us_per_output = pars->machine_timestep / pars->n_output_dimensions;

  // Setup Timer2, initialise output loop
  event_register_timer( SLOT_8 );
  timer_schedule_proc( outgoing_dimension_callback, 0, 0, g_us_per_output );

  // Return the output buffer
  return gp_output_values;
}

// Transmit a dimension
void outgoing_dimension_callback( uint index, uint arg1 ) {
  // Transmit the packet with the appropriate key
  spin1_send_mc_packet(
    gp_output_keys[ index ],
    bitsk(gp_output_values[ index ]),
    WITH_PAYLOAD
  );

  // Zero the output buffer and increment the output dimension counter
  gp_output_values[ index ] = 0;
  index++;
  
  if( index >= g_n_output_dimensions ) {
    index = 0;
  }

  // Schedule this function to be called again
  if( !timer_schedule_proc( (event_proc) outgoing_dimension_callback, index, 0, g_us_per_output ) ) {
    io_printf( IO_BUF, "[Timer2] [ERROR] Failed to schedule next.\n" );
  }
}
