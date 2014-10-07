#include "disciplined_clock.h"

#ifndef MIN
#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) (((a)<(b)) ? (b) : (a))
#endif

static volatile dclk_state_t state;

void
dclk_initialise_state(void)
{
	state.last_update_time = dclk_read_raw_time();
	state.last_corrected_time = dclk_read_raw_time();
	state.offset = 0;
	state.correction_freq = 0;
	state.correction_phase_accumulator = 0;
	state.freq_correction_weight = DCLK_FREQ_CORRECTION_WEIGHT_START;
	state.phase_correction_weight = DCLK_PHASE_CORRECTION_WEIGHT_START;
}


dclk_time_t
dclk_read_raw_time(void)
{
	return -tc2[TC_COUNT];
}


dclk_time_t
dclk_get_time(void)
{
	dclk_time_t raw_time = dclk_read_raw_time();
	
	// Work out the correction for frequency since the last frequency correction
	dclk_time_t delta_raw_ticks = raw_time - state.last_update_time;
	dclk_offset_t freq_correction = (dclk_offset_t)( ( ((dclk_dfp_freq_t)delta_raw_ticks)
	                                                 * ((dclk_dfp_freq_t)state.correction_freq)
	                                                 )
	                                               >> DCLK_FP_FREQ_FBITS
	                                               );
	
	// Apply any accumulated phase error while ensuring time monotonicity.
	if (state.correction_phase_accumulator > 0) {
		// Positive, integral phase errors can be incorporated immediately without
		// any impact on monotonicity.
		dclk_offset_t phase_correction = state.correction_phase_accumulator
		                                 >> DCLK_FP_PHASE_FBITS;
		
		state.correction_phase_accumulator -= phase_correction << DCLK_FP_PHASE_FBITS;
		state.offset += phase_correction;
	} else if (state.correction_phase_accumulator < 0) {
		// Negative, integral phase errors can be incorporated only at the rate at
		// which corrected time elapses.
		dclk_time_t new_corrected_time = raw_time + state.offset + freq_correction;
		dclk_time_t corrected_time_since_last_read = new_corrected_time - state.last_corrected_time;
		
		dclk_offset_t phase_correction = -((state.correction_phase_accumulator >> DCLK_FP_PHASE_FBITS) + 1);
		phase_correction = MIN(corrected_time_since_last_read, phase_correction);
		
		state.correction_phase_accumulator += phase_correction << DCLK_FP_PHASE_FBITS;
		state.offset -= phase_correction;
	}
	
	state.last_corrected_time = raw_time + state.offset + freq_correction;
	return state.last_corrected_time;
}


dclk_time_t
dclk_get_ticks_until_time(dclk_time_t target_time)
{
	// If the clock was perfect, how many ticks would we need to wait?
	dclk_time_t cur_time = dclk_get_time();
	dclk_offset_t delta_ticks = target_time - cur_time;
	
	// If the time was in the past don't wait (note that "in the past" here means
	// more than half the timer range away since the timers can wrap).
	if (delta_ticks <= 0)
		return 0;
	
	// Given that the clock is really running at a different frequency, find out
	// how many ticks will be added/removed due to frequency corrections (assuming
	// the frequency correction rate remains constant over the period) and update
	// our estimate.
	//
	// TODO: Could take advantage of knowing *when* frequency corrections are
	// actually due to be applied (assuming freq doesn't change).
	dclk_offset_t freq_correction = (dclk_offset_t)( ( ((dclk_dfp_freq_t)delta_ticks)
	                                                 * ((dclk_dfp_freq_t)state.correction_freq)
	                                                 )
	                                               >> DCLK_FP_FREQ_FBITS
	                                               );
	delta_ticks += freq_correction;
	
	// How much phase error correction is (currently) due to be applied? Assume
	// the phase error won't change and simply apply as much as possible in the
	// given period.
	dclk_offset_t phase_correction = 0;
	if (state.correction_phase_accumulator > 0) {
		// Positive, integral phase errors can be incorporated immediately without
		// any impact on monotonicity. (Note that code should never be called since
		// calling dclk_get_time will have already incorporated this.)
		phase_correction = state.correction_phase_accumulator
		                   >> DCLK_FP_PHASE_FBITS;
	} else if (state.correction_phase_accumulator < 0) {
		// Negative, integral phase errors can be incorporated only at the rate at
		// which corrected time is expected to elapse (thus the phase correction can
		// only at worst cancel out the difference in time).
		phase_correction = -((state.correction_phase_accumulator >> DCLK_FP_PHASE_FBITS) + 1);
		phase_correction = MIN(delta_ticks, phase_correction);
	}
	delta_ticks += phase_correction;
	
	return delta_ticks;
}


void
dclk_correct_phase_now(dclk_offset_t correction)
{
	// Apply the desired phase correction immediately
	state.offset += correction;
	
	// Reset the phase accumulator
	state.correction_phase_accumulator = 0;
	
	// Attempt to read the clock which will set state.last_corrected_time but
	// since correction_phase_accumulator is zero, will only account for the
	// current frequency correction meaning future calls will be monotonic.
	dclk_get_time();
	
	// Mark now as the last update time such that the frequency estimate in the
	// next true update is usable.
	state.last_update_time = dclk_read_raw_time();
}


void
dclk_add_correction(dclk_offset_t correction)
{
	dclk_time_t raw_time = dclk_read_raw_time();
	
	dclk_time_t time_since_last_poll = raw_time - state.last_update_time;
	state.last_update_time = raw_time;
	
	if (time_since_last_poll == 0)
		return;
	
	// Update the offset to incorporate the frequency corrections since the last
	// correction was added since the frequency correction may change after this
	// update.
	dclk_dfp_freq_t freq_correction = ( ((dclk_dfp_freq_t)time_since_last_poll)
	                                  * ((dclk_dfp_freq_t)state.correction_freq)
	                                  );
	state.offset += (dclk_offset_t)(freq_correction >> DCLK_FP_FREQ_FBITS);
	
	// Add the remaining fractional frequency correction to the phase correction
	// accumulator.
	state.correction_phase_accumulator
		+= (dclk_fp_phase_t)(  (freq_correction & ((1ll<<DCLK_FP_FREQ_FBITS)-1))
		                    >> (DCLK_FP_FREQ_FBITS-DCLK_FP_PHASE_FBITS)
		                    );
	
	// Making the assumptions that neither oscillator has shifted and there is no
	// jitter in the correction measurement, what is the frequency of single-count
	// errors since the last poll? Note that since only one fixed point number is
	// involved, no shifting is required.
	dclk_fp_freq_t correction_freq_adjustment = (dclk_fp_freq_t)
	                                            ( ( ((dclk_dfp_freq_t)correction)
	                                              * ((dclk_dfp_freq_t)state.freq_correction_weight)
	                                              )
	                                            / ((dclk_dfp_freq_t)time_since_last_poll)
	                                            );
	state.correction_freq += correction_freq_adjustment;
	
	// Accumulate phase corrections of a fraction of the correction. By only
	// applying the corrections fractionally the effect of jitter is reduced at
	// the expense of a longer time constant before the time is corrected. Note
	// converted into fixed point by multiplication with a fixed point number.
	state.correction_phase_accumulator += correction * state.phase_correction_weight;
	
	// Move frequency/phase correction weights down from their starting values
	// towards their target values each time a correction is added. This allows
	// early corrections to be applied more harshly ensuring a quick initial lock.
	if (state.freq_correction_weight > DCLK_FREQ_CORRECTION_WEIGHT_STEP)
		state.freq_correction_weight -= DCLK_FREQ_CORRECTION_WEIGHT_STEP;
	state.freq_correction_weight = MAX( state.freq_correction_weight
	                                  , DCLK_FREQ_CORRECTION_WEIGHT_TARGET
	                                  );
	
	if (state.phase_correction_weight > DCLK_PHASE_CORRECTION_WEIGHT_STEP)
		state.phase_correction_weight -= DCLK_PHASE_CORRECTION_WEIGHT_STEP;
	state.phase_correction_weight = MAX( state.phase_correction_weight
	                                   , DCLK_PHASE_CORRECTION_WEIGHT_TARGET
	                                   );
}
