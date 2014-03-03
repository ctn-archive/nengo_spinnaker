/**
 * \addtogroup Ensemble
 * \brief An implementation of the Nengo LIF neuron with multidimensional
 *        input capabilities.
 *
 * The Ensemble component implements a LIF neuron model which accepts and
 * transmits multidimensional values.  As in the NEF each neuron in the
 * Ensemble has an *Encoder* which is provided by the Nengo framework running
 * on the host. On each time step the encoders are used to convert the real
 * value presented to the ensemble into currents applied to input of each
 * simulated neuron. Spikes are accumulated and converted into real values
 * using *decoders* (again provided by the host). Decoded values are output
 * in an interleaved fashion during the neuron update loop.
 *
 * Number | Region | Description | Handling Function
 * ------ | ------ | ----------- | -----------------
 * 1 | System | Global parameters | ::copy_in_system_region
 * 2 | Bias   | Bias currents | ::copy_in_bias
 * 3 | Encoders | Neuron encoder matrix | ::copy_in_encoders
 * 4 | Decoders | Decoder matrix | ::copy_in_decoders
 * 5 | Decoder Keys | Routing keys for decoded values | ::copy_in_decoder_keys
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 * \author Terry Stewart
 * 
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 * @{
 */

#include "spin1_api.h"
#include "stdfix-full-iso.h"
#include "ensemble-data.h"
#include "common-typedefs.h"

typedef accum value_t;
typedef accum current_t;
typedef accum voltage_t;

/* Main and callbacks ********************************************************/
int c_main( void );
void timer_callback( uint arg0, uint arg1 );
void incoming_spike_callback( uint key, uint payload );

/* Initialisation functions **************************************************/
void initialise_buffers( void );

/* Parameters ****************************************************************/
extern uint n_input_dimensions;  //!< Number of input dimensions \f$D_{in}\f$
extern uint n_output_dimensions; //!< Number of output dimensions \f$D_{out}\f$
extern uint * output_keys;       //!< Output dimension keys \f$1 \times D_{out}\f$
extern uint n_neurons;           //!< Number of neurons \f$N\f$

extern uint dt;                  //!< Machine time step  / useconds
extern uint t_ref;               //!< Refractory period \f$\tau_{ref} - 1\f$ / steps
extern value_t one_over_t_rc;    //!< \f$\tau_{rc}^{-1}\f$
extern value_t filter;           //!< Input decay factor

extern current_t * i_bias;       //!< Population biases \f$1 \times N\f$

extern accum * encoders; //!< Encoder values \f$N \times D_{in}\f$ (including gains)
extern accum * decoders; //!< Decoder values \f$N \times \sum D_{outs}\f$

/* Buffers *******************************************************************/
extern value_t * ibuf_accumulator; //!< Input buffers \f$1 \times D_{in}\f$
extern value_t * ibuf_filtered;    //!< Filtered input buffers \f$1 \times D_{in}\f$
extern uint * v_ref_voltage;       //!< 4b refractory state, remainder voltages
extern value_t * output_values;    //!< Output buffers \f$1 \times D_{out}\f$

/* Static inline access functions ********************************************/
// -- Encoder(s) and decoder(s)
//! Get the encoder value for the given neuron and dimension
static inline accum neuron_encoder( uint n, uint d )
  { return encoders[ n * n_input_dimensions + d ]; };

static inline accum neuron_decoder( uint n, uint d )
  { return decoders[ n * n_output_dimensions + d ]; };

// -- Voltages and refractory periods
//! Get the membrane voltage for the given neuron
static inline voltage_t neuron_voltage( uint n )
  { return kbits( v_ref_voltage[n] & 0x0fffffff ); };

//! Set the membrane voltage for the given neuron
static inline void set_neuron_voltage( uint n, voltage_t v )
  { v_ref_voltage[n] = (
      ( v_ref_voltage[n] & 0xf0000000 )
    | ( bitsk( v ) & 0x0fffffff ) );
  };

//! Get the refractory status of a given neuron
static inline uint neuron_refractory( uint n )
  { return ( v_ref_voltage[n] & 0xf0000000 ) >> 28; };

//! Put the given neuron in a refractory state (zero voltage, set timer)
static inline void set_neuron_refractory( uint n )
  { v_ref_voltage[n] = t_ref; };

//! Decrement the refractory time for the given neuron
static inline void decrement_neuron_refractory( uint n )
  { v_ref_voltage[n] -= 0x10000000; };

/** @}*/
