# SpiNNaker Nengo Integration
# ----------------------------------------------------------------------------
# Authors:
#  Andrew Mundy <mundya@cs.man.ac.uk>
#  Terry Stewart
#
# February 2014
# ----------------------------------------------------------------------------
# Code derived from Andrew Rowley, University of Manchester

APPS = ensemble tx rx
COMPLETE_SOURCE = ./*/*.c ./*/*.h

all :
	for a in $(APPS); do ( cd $$a; "$(MAKE)"  ) || exit $$?; done

docs : ${COMPLETE_SOURCE}
	doxygen

clean :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" clean  ) || exit $$?; done
	rm -rf ./docs/
