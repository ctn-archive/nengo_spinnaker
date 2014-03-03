/** \addtogroup Ensemble
 * @{ */
/*
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 */
#include "common-impl.h"
#include "common-typedefs.h"

#ifndef __ENSEMBLE_DATA_H__
#define __ENSEMBLE_DATA_H__

bool copy_in_system_region( address_t addr );
bool copy_in_bias( address_t addr );
bool copy_in_encoders( address_t addr );
bool copy_in_decoders( address_t addr );
bool copy_in_decoder_keys( address_t addr );

#endif
/** @}*/
