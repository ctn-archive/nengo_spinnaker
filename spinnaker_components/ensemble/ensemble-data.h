/*****************************************************************************

SpiNNaker and Nengo Integration

******************************************************************************

Authors:
 Andrew Mundy <mundya@cs.man.ac.uk> -- University of Manchester
 Terry Stewart			    -- University of Waterloo

Date:
 17-22 February 2014

******************************************************************************

Advanced Processors Technologies,   Computational Neuroscience Research Group,
School of Computer Science,         Centre for Theoretical Neuroscience,
University of Manchester,           University of Waterloo,
Oxford Road,                        200 University Avenue West,
Manchester, M13 9PL,                Waterloo, ON, N2L 3G1,
United Kingdom                      Canada

*****************************************************************************/

#include "common-impl.h"
#include "common-typedefs.h"

#ifndef __ENSEMBLE_DATA_H__
#define __ENSEMBLE_DATA_H__

bool copy_in_system_region( address_t addr ); // Copy in global data
bool copy_in_bias( address_t addr );          // Copy in bias data
bool copy_in_encoders( address_t addr );      // Copy in encoders
bool copy_in_decoders( address_t addr );      // Copy in decoders
bool copy_in_decoder_keys( address_t addr );  // Copy in decoder keys

#endif
