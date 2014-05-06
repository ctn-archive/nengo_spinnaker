# SpiNNaker Nengo Integration
# ----------------------------------------------------------------------------
# Authors:
#  Andrew Mundy <mundya@cs.man.ac.uk>
#  Terry Stewart
#
# February - April 2014
# ----------------------------------------------------------------------------
# Code derived from Andrew Rowley, University of Manchester

APPS = ensemble tx rx filter
APP_OUTPUT_DIR = $(PWD)/binaries
COMPLETE_SOURCE = ./*/*.c ./*/*.h

DEVELOPER = $(filter nengo_spinnaker, $(notdir $(wildcard ../*)))

all :
	mkdir -p $(APP_OUTPUT_DIR)
	for a in $(APPS); do ( cd $$a; "$(MAKE)"  ) || exit $$?; done

docs : ${COMPLETE_SOURCE}
	doxygen

clean :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" clean  ) || exit $$?; done
	rm -rf ./docs/
