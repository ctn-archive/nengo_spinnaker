/*
 * Ensemble - Harness
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
 */

#include "ensemble.h"

/* Parameters and Buffers ***************************************************/
uint g_n_neurons;        
uint g_dt;               
uint g_t_ref;            
value_t g_one_over_t_rc; 
current_t * gp_i_bias;   
accum * gp_encoders;     
accum * gp_decoders;     
uint * gp_v_ref_voltage; 

/* Initialisation ***********************************************************/
void initialise_ensemble(
  uint n_neurons,
  uint dt,
  uint n_input_dimensions,
  uint n_output_dimensions,
  uint t_ref,
  value_t one_over_t_rc,
  value_t filter
) {
  // Save constants
  g_n_neurons = n_neurons;
  g_dt = dt;
  g_t_ref = t_ref;
  g_one_over_t_rc = one_over_t_rc;

  // Holder for bias currents
  gp_i_bias = spin1_malloc(
    g_n_neurons * sizeof( current_t )
  );

  // Holder for refactory period and voltages
  gp_v_ref_voltage = spin1_malloc(
    g_n_neurons * sizeof( uint )
  );
  for( uint n = 0; n < g_n_neurons; n++ ){
    gp_v_ref_voltage[n] = 0;
  }

  // Initialise some buffers
  gp_encoders = spin1_malloc(
    g_n_neurons * n_input_dimensions * sizeof( accum )
  );
  gp_decoders = spin1_malloc(
    g_n_neurons * n_output_dimensions * sizeof( accum )
  );

  // Setup subcomponents
  initialise_input( n_input_dimensions, dt );
  initialise_output( n_output_dimensions, filter );
}
