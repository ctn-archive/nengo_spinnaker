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

#include "spin-nengo-ensemble.h"

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

  // Consider changing this so that we:
  // 1. Use a power of 2 for the gap between transmitting
  // 2. Compute this elsewhere
  uint neurons_per_packet = n_neurons / n_output_dimensions;
  uint n_counter = neurons_per_packet;
  uint n_current_output_dimension = 0;
  
  // For every input dimension, decay the input value and zero the accumulator.
  for( uint d = 0; d < n_input_dimensions; d++ ) {
    /* START CRITICAL SECTION */
    ibuf_filtered[d] = ibuf_filtered[d] * filter
                     + ibuf_accumulator[d] * (1 - filter);
    ibuf_accumulator[d] = 0;
    /* END CRITICAL SECTION */
  }

  // Perform neuron updates, interspersed with decoding and transmitting
  for( uint n = 0; n < n_neurons; n++ ) {
    // If this neuron is a multiple of neurons_per_packet then transmit a
    // dimension packet.
    n_counter--;
    if( n_counter == 0 ){
      // Transmit the packet with the appropriate key
      spin1_send_mc_packet(
        output_keys[ n_current_output_dimension ],
        bitsk(output_values[ n_current_output_dimension ]),
        WITH_PAYLOAD
      );

      // Zero the output buffer and increment the output dimension counter
      output_values[ n_current_output_dimension ] = 0;
      n_current_output_dimension++;
      n_counter = neurons_per_packet;
    }

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
