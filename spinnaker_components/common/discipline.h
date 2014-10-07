/*!
 * \brief The discipline system keeps clocks (and thus timer interrupts) in the
 *        system locked on a single global reference time provided by the
 *        simulation controller. Also responsible for starting/stopping
 *        interrupts in order to implement pausing.
 *
 * \author Jonathan Heathcote <jdh@cs.man.ac.uk>
 * \copyright Advanced Processor Technologies, School of Computer Science, 
 *            University of Manchester
 * @{
 */

#include "spin1_api.h"

#include "common-typedefs.h"

#include "disciplined_clock.h"
#include "disciplined_timer.h"

#ifndef __DISCIPLINE_H__
#define __DISCIPLINE_H__

/*! \brief Region containing MC keys used for discipline communications.
 */
typedef struct _region_discipline_keys_t {
  uint ping; //!< Key for pings from sim controller
  uint pong; //!< Key for ping responses to sim controller
  uint correction; //!< Key for corrections from sim controller
  uint start_at; //!< Key for timer interrupt start command from sim controller
  uint stop_at;  //!< Key for timer interrupt stop command from sim controller
} region_discipline_keys_t;


/*! \brief Initialise the clock/timer discipline system and specify the timestep
 *         (usec) to be used.
 */
void discipline_initialise(address_t pars, uint timestep);

/*! \brief Process an MC packet used for clock discipline.
 *  \returns Whether the MC packet was handled.
 */
bool discipline_process_mc_packet(uint key, uint payload);

/*! \brief To be called once during every timer interrupt.
 *  \returns Time (in timer 2 ticks) at which the interrupt was intended to
 *           happen.
 */
inline dclk_time_t discipline_on_interrupt(void) {
  return dtimer_schedule_next_interrupt();
}

#endif

/*! @} */
