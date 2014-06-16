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
#include "ensemble-output.h"
#include "ensemble-pes.h"

uint lfsr = 1;                   //!< LFSR for spike perturbation

void ensemble_update(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
  }

  // Prepare the recorder
  record_buffer_prepare(&g_ensemble.recd);

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
    for( uchar d = 0; d < g_input.n_dimensions; d++ ) {
      i_membrane += neuron_encoder(n, d) * g_ensemble.input[d];
    }

    v_voltage = neuron_voltage(n);
    v_delta = ( i_membrane - v_voltage ) * g_ensemble.dt_over_t_rc;
    /* io_printf( IO_STD, "n = %d, J = %k, V = %k, dV = %k\n",
                  n, i_membrane, v_voltage, v_delta );
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
      //io_printf( IO_STD, "[Ensemble] Neuron %d spiked.", n );

      // Zero the voltage, set the refractory time
      set_neuron_refractory( n );
      set_neuron_voltage(n, 0.0k);

      /* Randomly perturb the refractory period to account for inter-tick
         spiking.*/
      if(kbits(lfsr & 0x00007fff) * v_delta < v_voltage - 1.0k) {
        decrement_neuron_refractory( n );
      }
      lfsr = ((lfsr >> 1) ^ (~lfsr & 0xB400));

      // Update the output values
      for( uint d = 0; d < g_n_output_dimensions; d++ ) {
        /* io_printf( IO_STD, "[%d] = %.3k (0x%08x)",
          d, neuron_decoder(n,d), neuron_decoder(n,d) ); */
        g_ensemble.output[d] += neuron_decoder( n, d );
      }
      //io_printf( IO_STD, "\n" );

      // Record that the spike occurred
      record_spike(&g_ensemble.recd, n);
      
      // Update learning rule's filtered activity
      //pes_update_neuron_activity(n, true);
      pes_neuron_spiked(n);
    }
    // Otherwise, just update learning rule's filtered activity
    /*else
    {
      pes_update_neuron_activity(n, false);
    }*/
  }
  
  // Update PES decoder learning
  //pes_update();

  // Transmit decoded Ensemble representation
  for (uint output_index = 0; output_index < g_n_output_dimensions;
       output_index++) {
    spin1_send_mc_packet(
      gp_output_keys[output_index],
      bitsk(gp_output_values[output_index]),
      WITH_PAYLOAD
    );
    gp_output_values[output_index] = 0;

    spin1_delay_us(1);
  }

  // Flush the recording buffer
  record_buffer_flush(&g_ensemble.recd);
}
