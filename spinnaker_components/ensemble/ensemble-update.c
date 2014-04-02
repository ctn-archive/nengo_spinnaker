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

#include "ensemble.h"

uint lfsr = 1;  //!< LFSR for spike peturbation

void ensemble_update( uint arg0, uint arg1 )
{
  // Values used below
  current_t i_membrane;
  voltage_t v_delta, v_voltage;

  // Filter inputs
  input_filter_step( );

  // Perform neuron updates
  for( uint n = 0; n < g_ensemble.n_neurons; n++ ) {
    // If this neuron is refractory then skip any further processing
    if( neuron_refractory( n ) != 0 ) {
      decrement_neuron_refractory( n );
      continue;
    }

    // Include neuron bias
    i_membrane = g_ensemble.i_bias[n];

    // Encode the input and add to the membrane current
    for( uchar d = 0; d < g_n_input_dimensions; d++ ) {
      i_membrane += neuron_encoder(n, d) * g_ensemble.input[d];
    }

    v_voltage = neuron_voltage(n);
    v_delta = ( i_membrane - v_voltage ) * g_ensemble.one_over_t_rc;
    /* io_printf( IO_STD, "dt = %k, J = %k, V = %k, dV = %k\n",
                  dt, i_membrane, v_voltage, v_delta );
    */
    v_voltage += v_delta;

    // Voltages can't go below 0.0
    if( v_voltage < 0.0k ) {
      v_voltage = 0.0k;
    }

    // Save state
    set_neuron_voltage( n, v_voltage );

    // If this neuron has fired then process
    if( v_voltage > 1.0k ) {
      // Zero the voltage, set the refractory time
      set_neuron_refractory( n );
      //io_printf( IO_STD, "%d SPIKED.  %d ticks till active. v_ = 0x%08x\n",
      //           n, neuron_refractory( n ), v_ref_voltage[n] );

      /* Randomly peturb the refractory period to account for inter-tick
         spiking.*/
      if( bitsk(lfsr) * v_delta < v_voltage - 1.0k ) {
        decrement_neuron_refractory( n );
      }
      lfsr = ((lfsr >> 1) ^ (~lfsr & 0xB400)) & 0x00007fff;

      // Update the output values
      for( uint d = 0; d < g_n_output_dimensions; d++ ) {
        //io_printf( IO_STD, "%d spike: %d: %08x\n", n, d, neuron_decoder(n, d));
        g_ensemble.output[d] += neuron_decoder( n, d );
      }
    }
  }
}
