#include <spinnaker.h>
#include <spin1_api.h>

#include "disciplined_clock.h"
#include "disciplined_timer.h"

/**
 * The disciplined timer state variable. Just one since this library can only
 * control a single timer.
 */
static volatile dtimer_state_t dtimer;


void
dtimer_start_interrupts( dclk_time_t next_interrupt_time
                       , dclk_time_t interrupt_period
                       )
{
	// Set up the data structure
	dtimer.next_interrupt_time = next_interrupt_time;
	dtimer.interrupt_period    = interrupt_period;
	dtimer.stop                = FALSE;
	
	// Set up the timer (but do not enable and don't set the prescaler)
	uint timer_control = DTIMER_TC[TC_CONTROL];
	timer_control &= ~( (1 << 0) // One shot
	                  | (1 << 1) // Timer Size
	                  | (1 << 5) // Interrupt Enable
	                  | (1 << 6) // Mode
	                  | (1 << 7) // Enable
	                  );
	timer_control |=  ( (1 << 0) // One-shot counter
	                  | (1 << 1) // 32-bit counter
	                  | (1 << 5) // Interrupt
	                  | (1 << 6) // Periodic
	                  | (0 << 7) // Not Enabled
	                  );
	DTIMER_TC[TC_CONTROL] = timer_control;
	
	// Schedule the next interrupt. Note that the next_interrupt_time is
	// decremented since dtimer_schedule_next_interrupt will set the interrupt for
	// whatever time next_interrupt_time+interrupt_period is.
	dtimer.next_interrupt_time -= dtimer.interrupt_period;
	dtimer_schedule_next_interrupt();
	
	// Enable the timer (and thus its interrupts)
	DTIMER_TC[TC_CONTROL] |= (1<<7);
}


void
dtimer_stop_interrupts(dclk_time_t stop_time)
{
	// Note: Assignment ordering is significant
	dtimer.stop_time = stop_time;
	dtimer.stop = TRUE;
}


dclk_time_t
dtimer_schedule_next_interrupt(void)
{
	// Note the time that this interrupt was supposed to occur (since it will be
	// returned to the user)
	dclk_time_t nominal_time_now = dtimer.next_interrupt_time;
	
	// Stop the timer if required
	dclk_time_t new_next_interrupt_time = nominal_time_now
	                                    + dtimer.interrupt_period;
	if ( dtimer.stop
	     && (dclk_offset_t)(dtimer.stop_time - new_next_interrupt_time) < 0
	   ) {
		// Disable the timer (not required useful as an indicator that interrupts
		// intentionally stopped).
		DTIMER_TC[TC_CONTROL] &= ~(1<<7);
	} else {
		// Reload the timer with the next interrupt time (make sure that the number of
		// ticks is at least one to ensure the interrupt does happen).
		dclk_time_t ticks_til_next_interrupt
			= dclk_get_ticks_until_time(dtimer.next_interrupt_time);
		
		dtimer.next_interrupt_time = new_next_interrupt_time;
		
		if (ticks_til_next_interrupt > 0)
			DTIMER_TC[TC_LOAD] = ticks_til_next_interrupt;
		else
			DTIMER_TC[TC_LOAD] = 1;
	}
	
	return nominal_time_now;
}


