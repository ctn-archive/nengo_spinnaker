# SpiNNaker Nengo Integration
# ----------------------------------------------------------------------------
# Authors:
#  Andrew Mundy <mundya@cs.man.ac.uk>
#  Terry Stewart
#
# February 2014
# ----------------------------------------------------------------------------
# Code derived from Andrew Rowley, University of Manchester

APPS = ensemble dimsource tx

all :
	for a in $(APPS); do ( cd $$a; "$(MAKE)"  ); done

clean :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" clean  ); done
