/**
 * A disciplined clock which can attempt to track a reference clock. This
 * version of the library is a fork from the original and is hard-coded for
 * SpiNNaker's TC2.
 *
 * Authors:
 *   - Jonathan Heathcote <jdh@cs.man.ac.uk>
 */

#ifndef DISCIPLINED_CLOCK_H
#define DISCIPLINED_CLOCK_H

#include <stdint.h>


////////////////////////////////////////////////////////////////////////////////
// Type and function prototype definitions
////////////////////////////////////////////////////////////////////////////////

// Integer types used to define time and time offsets.
typedef uint32_t dclk_time_t;
typedef  int32_t dclk_offset_t;

// Fixed point types use for fractional values. Note that while the types do not
// enforce fixed point in any way, the use of these types should inform the
// reader when fixed point numbers are in use.
typedef  int32_t dclk_fp_freq_t;
typedef  int32_t dclk_fp_phase_t;

// Double length fixed point numbers. These are used internally within
// certain calculations to ensure appropriate range is available.
typedef  int64_t dclk_dfp_freq_t;
typedef  int64_t dclk_dfp_phase_t;

/**
 * A structure which stores all persistent clock state. Not intended for public
 * access.
 */
typedef struct {
	// The last raw value of the clock source when an update was performed.
	dclk_time_t last_update_time;
	
	// The last corrected value of the clock source when the time was read. Used to
	// ensure monotonic time when incorporating phase corrections.
	dclk_time_t last_corrected_time;
	
	// The current time offset from the raw clock source (correct at the time of
	// last reading).
	dclk_offset_t offset;
	
	// The frequency at which corrections should be accumulated to compensate for
	// frequency mismatch. If 0, no corrections are required, if +ve then 1 should
	// be added to the offset at this frequency. If -ve, -1 should be added at
	// this frequency.
	dclk_fp_freq_t correction_freq;
	
	// An accumulator for phase (i.e. time) corrections. These are added to the
	// offset in integral steps at such a rate that the time remains monotonic.
	dclk_fp_phase_t correction_phase_accumulator;
	
	// Weights which are applied to updates to correction frequency and phase
	// respectively.
	dclk_fp_freq_t  freq_correction_weight;
	dclk_fp_phase_t phase_correction_weight;
} dclk_state_t;


// The number of fractional bits in a fixed point value representing a frequency
#define DCLK_FP_FREQ_FBITS 30

// The number of fractional bits in a fixed point value representing a phase
#define DCLK_FP_PHASE_FBITS 16

// Conversions from doubles to fixed-point types (intended to be done in the
// compiler).
#define DCLK_DOUBLE_TO_FP_FREQ(d)    ((dclk_fp_freq_t)((d) * ((double)(1<<DCLK_FP_FREQ_FBITS))))
#define DCLK_DOUBLE_TO_FP_PHASE(d)   ((dclk_fp_phase_t)((d) * ((double)(1<<DCLK_FP_PHASE_FBITS))))


/**
 * Initialise the state structure.
 */
void dclk_initialise_state(void);


/**
 * Reads the current raw timer value.
 */
dclk_time_t dclk_read_raw_time(void);


/**
 * Get the current corrected time (in timer ticks). Note that sequential calls
 * to this command are not guaranteed to observe every time value however values
 * are guaranteed to be monotonic within the period of the timer (i.e. the
 * period of dclk_time_t). This guarantee is broken if dclk_correct_phase_now is
 * called.
 */
dclk_time_t dclk_get_time(void);


/**
 * Get the estimated number of raw ticks until a specified (corrected) timer
 * value will be reached.
 *
 * This function can only produce estimates since it cannot account for future
 * corrections which may slow down/speed up the clock. The estimate is based on
 * the accumulated phase error and estimated frequency error.
 *
 * Note that if the specified time is in the past or in (unusual) cases where
 * the clock has become badly out of phase, the function will return zero.
 */
dclk_time_t dclk_get_ticks_until_time(dclk_time_t time);


/**
 * When the clock first starts it may suffer from a large phase offset from the
 * master clock and so this function can be used to violate clock monotonicity
 * and also bypass the clock discipline algorithm to initially set the clock.
 * This can avoid inducing large oscillations due to the initial error and allow
 * the clock to settle quicker.
 */
void dclk_correct_phase_now(dclk_offset_t correction);


/**
 * Given a noisy correction from a remote clock, attempt to discipline the
 * clock.
 */
void dclk_add_correction(dclk_offset_t correction);


////////////////////////////////////////////////////////////////////////////////
// Discipline Parameters
////////////////////////////////////////////////////////////////////////////////

// The weight with which the correction frequency updates are applied. This is
// ramped down from DCLK_FREQ_CORRECTION_WEIGHT_START to
// DCLK_FREQ_CORRECTION_WEIGHT_TARGET in steps of
// DCLK_FREQ_CORRECTION_WEIGHT_STEP. This allows the system to quickly lock on
// to roughly the right frequency and then later filter out noise.
#define DCLK_FREQ_CORRECTION_WEIGHT_TARGET DCLK_DOUBLE_TO_FP_FREQ(0.05)
#define DCLK_FREQ_CORRECTION_WEIGHT_START  DCLK_DOUBLE_TO_FP_FREQ(1.0)
#define DCLK_FREQ_CORRECTION_WEIGHT_STEP   DCLK_DOUBLE_TO_FP_FREQ(0.10)

// The weight with which phase updates are applied. This is ramped down from
// DCLK_PHASE_CORRECTION_WEIGHT_START to DCLK_PHASE_CORRECTION_WEIGHT_TARGET in
// steps of DCLK_PHASE_CORRECTION_WEIGHT_STEP. This allows the system to quickly
// lock on to roughly the right time and then later filter out noise.
#define DCLK_PHASE_CORRECTION_WEIGHT_TARGET DCLK_DOUBLE_TO_FP_PHASE(0.1)
#define DCLK_PHASE_CORRECTION_WEIGHT_START  DCLK_DOUBLE_TO_FP_PHASE(1.0)
#define DCLK_PHASE_CORRECTION_WEIGHT_STEP   DCLK_DOUBLE_TO_FP_PHASE(0.2)

#endif
