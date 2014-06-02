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

#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

#include "spin1_api.h"
#include "stdfix-full-iso.h"
#include "common-typedefs.h"

#include "nengo_typedefs.h"
#include "nengo-common.h"

#include "recording.h"

#include "ensemble_data.h"
#include "ensemble_output.h"

#include "input_filter.h"

/* Structs ******************************************************************/
/** \brief Persistent neuron variables.
  */
typedef struct neuron_status {
  unsigned char refractory_time : 4;  //!< 4 bits of refractory state
  unsigned int  voltage : 28;           //!< 28 bits stored voltage
} neuron_status_t;

/** \brief Shared ensemble parameters.
  */
typedef struct ensemble_parameters {
  uint n_neurons;          //!< Number of neurons \f$N\f$
  uint machine_timestep;   //!< Machine time step  / useconds

  uint t_ref;              //!< Refractory period \f$\tau_{ref} - 1\f$ / steps
  value_t dt_over_t_rc;    //!< \f$\frac{dt}{\tau_{rc}}\$

  current_t *i_bias;        //!< Population biases \f$1 \times N\f$
  neuron_status_t *status;  //!< Neuron status

  value_t *encoders;        //!< Encoder values \f$N \times D_{in}\f$ (including gains)
  value_t *decoders;        //!< Decoder values \f$N \times\sum D_{outs}\f$

  value_t *input;           //!< Input buffer
  value_t *output;          //!< Output buffer

  recording_buffer_t recd;  //!< Spike recording
} ensemble_parameters_t;

/* Parameters and Buffers ***************************************************/
extern ensemble_parameters_t g_ensemble;  //!< Global parameters
extern uint g_output_period;       //!< Delay in transmitting decoded output
extern input_filter_t g_input;     //!< Input filters and buffers

/* Functions ****************************************************************/
/**
 * \brief Initialise the ensemble.
 */
bool initialise_ensemble(
  region_system_t *pars  //!< Pointer to formatted system region
);

/**
 * \brief Filter input values, perform neuron update and transmit any output
 *        packets.
 * \param arg0 Unused parameter
 * \param arg1 Unused parameter
 *
 * Neurons are then simulated using Euler's Method as in most implementations
 * of the NEF.  When a neuron spikes it is immediately decoded and its
 * contribution to the output of the Ensemble added to ::output_values.
 */
void ensemble_update( uint arg0, uint arg1 );

/* Static inline access functions ********************************************/
// -- Encoder(s) and decoder(s)
//! Get the encoder value for the given neuron and dimension
static inline value_t neuron_encoder( uint n, uint d )
  { return g_ensemble.encoders[ n * g_input.n_dimensions + d ]; };

static inline value_t neuron_decoder( uint n, uint d )
  { return g_ensemble.decoders[ n * g_n_output_dimensions + d ]; };

// -- Voltages and refractory periods
//! Get the membrane voltage for the given neuron
static inline voltage_t neuron_voltage( uint n )
  { return kbits( g_ensemble.status[n].voltage ); };

//! Set the membrane voltage for the given neuron
static inline void set_neuron_voltage( uint n, voltage_t v )
  { g_ensemble.status[n].voltage = ( bitsk( v ) & 0x0fffffff ); };

//! Get the refractory status of a given neuron
static inline uint neuron_refractory( uint n )
  { return g_ensemble.status[n].refractory_time; };

//! Put the given neuron in a refractory state (zero voltage, set timer)
static inline void set_neuron_refractory( uint n )
  { g_ensemble.status[n].refractory_time = g_ensemble.t_ref; };

//! Decrement the refractory time for the given neuron
static inline void decrement_neuron_refractory( uint n )
  { g_ensemble.status[n].refractory_time--; };

#endif

/** @} */
