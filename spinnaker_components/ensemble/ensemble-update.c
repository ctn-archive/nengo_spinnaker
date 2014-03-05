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

#include "spin-nengo-ensemble.h"

/**
 * \brief Filter input values, perform neuron update and transmit any output
 *        packets.
 * \param arg0 Unused parameter
 * \param arg1 Unused parameter
 *
 * The following steps are performed by this code:
 * 1. Decay the input values
 * 2. Simulate neurons
 * 3. Simultaneously accumulate output values and transmit
 *
 * Decay Input Values
 * ------------------
 *  The previous input values stored in ::ibuf_filtered are multiplied by
 *  ::filter to provide a decay of stored values.  Following this the values
 *  accumulated in ::ibuf_accumulator are multiplied by \f$1 - filter\f$ and
 *  added to the filtered values.  The values in ::ibuf_accumulator are then
 *  zeroed.
 *
 * Simulate Neurons
 * ----------------
 *  Neurons are then simulated using Euler's Method as in most implementations
 *  of the NEF.  When a neuron spikes it is immediately decoded and its
 *  contribution to the output of the Ensemble added to ::output_values.
 */
void timer_callback( uint arg0, uint arg1 )
{
  /*
   * - Decay input values (L.P.F) - switch off packet receipt interrupt here?
   * - Perform neural updates, interleave with decoding and transmitting
   *   dimension data.
   */
  // Values used below
  current_t i_membrane;
  voltage_t v_delta, v_voltage;

  // For every input dimension, decay the input value and zero the accumulator.
  for( uint d = 0; d < n_input_dimensions; d++ ) {
    /* START CRITICAL SECTION */
    ibuf_filtered[d] = ibuf_filtered[d] * filter
                     + ibuf_accumulator[d] * (1 - filter);
    ibuf_accumulator[d] = 0;
    /* END CRITICAL SECTION */
  }

  // Perform neuron updates
  for( uint n = 0; n < n_neurons; n++ ) {
    // If this neuron is refractory then skip any further processing
    if( neuron_refractory( n ) != 0 ) {
      decrement_neuron_refractory( n );
      continue;
    }

    // Include neuron bias
    i_membrane = i_bias[n];

    // Encode the input and add to the membrane current
    for( uchar d = 0; d < n_input_dimensions; d++ ) {
      i_membrane += neuron_encoder(n, d) * ibuf_filtered[d];
    }

    v_voltage = neuron_voltage(n);
    v_delta = ( i_membrane - v_voltage ) * one_over_t_rc;
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

      // Update the output values
      for( uint d = 0; d < n_output_dimensions; d++ ) {
        //io_printf( IO_STD, "%d spike: %d: %08x\n", n, d, neuron_decoder(n, d));
        output_values[d] += neuron_decoder( n, d );
      }
    }
  }
}
