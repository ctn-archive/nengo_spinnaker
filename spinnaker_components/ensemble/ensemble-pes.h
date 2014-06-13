#ifndef __ENSEMBLE_PES_H__
#define __ENSEMBLE_PES_H__

/**
 * Ensemble - PES
 * -----------------
 * Functions to perform PES decoder learning
 * 
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 * 
 * \addtogroup ensemble
 * @{
 */

//----------------------------------
// Structs
//----------------------------------
// Structure defining structure of PES region
// **TODO** this could be an opaque type here
typedef struct region_pes_t
{
  // Scalar learning rate used in PES decoder delta calculation
  value_t learning_rate;

  // Values defining the decay of the exponential filter applied 
  // To the neuron activity used in PES decoder delta calculation
  value_t activity_decay;
  
  // Index of the input signal filter that contains error signal
  uint error_signal_filter_index;
} region_pes_t;


// Structure containing parameters and state required for PES learning
typedef struct pes_parameters_t
{
  // Number of neuron sized array used to hold filtered neuron activity
  value_t *filtered_activity;

  // Scalar learning rate used in PES decoder delta calculation
  value_t learning_rate;

  // Values defining the decay (and one-minus the decay of the exponential filter 
  // Applied to the neuron activity used in PES decoder delta calculation
  value_t activity_decay;
  value_t one_minus_activity_decay;
  
  // Index of the input signal filter that contains error signal
  uint error_signal_filter_index;
} pes_parameters_t;

//----------------------------------
// External variables
//----------------------------------
extern pes_parameters_t g_pes;

//----------------------------------
// Inline functions
//----------------------------------
static inline void pes_update_neuron_activity(uint n, bool spiked)
{
  // If learning is enabled
  if(g_pes.learning_rate > 0.0k)
  {
    // Decay neuron's filtered activity
    g_pes.filtered_activity[n] *= g_pes.activity_decay;
    
    // If neuron's spiked, add extra energy into trace
    // **NOTE** spike causes 1.0 to be added
    if(spiked)
    {
      g_pes.filtered_activity[n] += g_pes.one_minus_activity_decay;
    }
  }
}

//----------------------------------
// Functions
//----------------------------------
/**
* \brief Copy in data controlling the PES learning 
* rule to the PES region of the Ensemble.
*/
void get_pes(region_pes_t *pars);

/**
* \brief Copy in data controlling the PES learning 
* rule to the PES region of the Ensemble.
*/
bool initialise_pes(uint n_neurons);

/**
* \brief Update PES learning rule
*/
void pes_update();

/** @} */

#endif  // __ENSEMBLE_PES_H__