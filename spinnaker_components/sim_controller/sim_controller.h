/*!
 * \brief Application which issues timing corrections and control signals to the
 *        simulation, thus controlling execution.
 *
 * \author Jonathan Heathcote <jdh@cs.man.ac.uk>
 * \copyright Advanced Processor Technologies, School of Computer Science, 
 *            University of Manchester
 * @{
 */

#ifndef __SIM_CONTROLLER_H__
#define __SIM_CONTROLLER_H__

// Commands to control the master, from the host
typedef enum _sim_control_cmd_t {
	// Get the current timer value (in ticks) from the sim controller
	SIM_CTL_GET_TIME  = 0,
	
	// Get the mean absolute drift in clock ticks of the remote clocks according
	// to the most recent round of corrections.
	SIM_CTL_GET_DRIFT = 1,
	
	// Cause interrupts to start on all cores at the specified time
	SIM_CTL_START_AT  = 2,
	
	// Cause interrupts to stop on all cores at the specified time
	SIM_CTL_STOP_AT   = 3,
} sim_control_cmd_t;

#endif

/*! @} */
