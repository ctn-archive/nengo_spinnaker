/**
 * A utility to synchronise the Timer 1 interrupt within a large SpiNNaker
 * machine.
 */

#ifndef DISCIPLINED_TIMER_H
#define DISCIPLINED_TIMER_H

#include "disciplined_clock.h"

/**
 * A structure which stores all persistent timer discipline state. Not intended
 * for public access.
 */
typedef struct {
	// The dclk_state_t of the disciplined clock to use as a reference
	volatile dclk_state_t *dclk;
	
	// The time at which the next timer interrupt is scheduled (corrected time)
	dclk_time_t next_interrupt_time;
	
	// The period between interrupts (in corrected timer ticks)
	dclk_time_t interrupt_period;
	
	// A boolean which indicates whether the timer should stop when the time
	// reaches (or passes) stop_time.
	uint stop;
	
	// If stop is TRUE, further interrupts will not be scheduled after this time.
	dclk_time_t stop_time;
} dtimer_state_t;


/**
 * Set up the timer such that it will interrupt starting at next_interrupt_time
 * and from then onwards at intervals of interrupt_period. Times are given in
 * timer ticks. Returns the value loaded into the timer.
 *
 * This function will set the timer and enable interrupts. Note that it will not
 * change the clock divider setting: this must be chosen to be the same as the
 * timer used by the disciplined clock.
 *
 * Note that the timer ISR must call dtimer_schedule_next_interrupt on every
 * interrupt. The ISR must also (on average) take less time to complete than the
 * interrupt_period. Assuming these are met and that the disciplined clock
 * remains locked, the ISR is guarunteed to be called the correct number of
 * times given the interrupt period.
 *
 * See the documentation for dclk_get_ticks_until_time for further timing
 * guaruntees.
 */
void dtimer_start_interrupts( volatile dclk_state_t *dclk
                            , dclk_time_t next_interrupt_time
                            , dclk_time_t interrupt_period
                            );

/**
 * Stop timer interrupts at the required time. Note that This will only prevent
 * new interrupts being scheduled; it will not stop the next interrupt
 * occurring. As a result, to ensure reliable global, synchronised stopping the
 * stop time should be at least one interrupt into the future.
 *
 * Note that calling dtimer_start_interrupts will clear this request. As a
 * result, if a stop time is being set immediately after dtimer_start_interrupts
 * interrupts may occurr before this function is called and so the time should
 * be sufficiently in the future to allow for this if deterministic behaviour is
 * desired.
 */
void dtimer_stop_interrupts(dclk_time_t stop_time);

/**
 * To be called from the ISR after the interrupt flag has been cleared. If this
 * function is not called, no further interrupts will occur. Returns the time at
 * which the interrupt was *supposed* to occur.
 *
 * This function must only be used after dtimer_start_interrupts has been
 * called.
 *
 * Calling this function outside of the ISR after interrupts have stopped will
 * resume interrupts from the last time they occurred. This will essentially
 * result in a rapid burst of interrupts until they catch up with their intended
 * time. This use case is probably not recommended.
 */
dclk_time_t dtimer_schedule_next_interrupt(void);

/**
 * Pointer to the specific timer to control.
 */
#define DTIMER_TC (tc1)

#endif

