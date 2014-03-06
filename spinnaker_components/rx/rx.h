/*****************************************************************************

SpiNNaker and Nengo Integration

******************************************************************************

Authors:
 Andrew Mundy <mundya@cs.man.ac.uk> -- University of Manchester
 Terry Stewart			    -- University of Waterloo

Date:
 17-22 February, 3 March 2014

******************************************************************************

Advanced Processors Technologies,   Computational Neuroscience Research Group,
School of Computer Science,         Centre for Theoretical Neuroscience,
University of Manchester,           University of Waterloo,
Oxford Road,                        200 University Avenue West,
Manchester, M13 9PL,                Waterloo, ON, N2L 3G1,
United Kingdom                      Canada

*****************************************************************************/

#include "spin1_api.h"
#include "stdfix-full-iso.h"
#include "common-impl.h"
#include "common-typedefs.h"

// Typedefs
typedef accum value_t;

// Function prototypes
int c_main( void );

void copy_in_system_region( address_t addr );   //! System region
void copy_in_keys( address_t addr );            //! Keys
void copy_in_initial_values( address_t addr );  //! Copy initial values

void timer_callback( uint simulation_time, uint none ); //! Timer tick
void sdp_packet_received( uint mailbox, uint port );    //! Handle SDP Rx

// System variables
extern uint n_dimensions; //!< Number of dimensions associated with this Rx
extern uint dt;           //!< Time step in us

extern uint ticks_per_output; //!< Number of ticks to wait between each output
extern uint n_current_output; //!< Index of current output

extern uint *keys;        //!< Keys to associate with outgoing values
extern value_t *values;   //!< Most recently cached output value
