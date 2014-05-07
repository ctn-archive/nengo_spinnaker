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

all :
	for a in $(APPS); do ( cd $$a; "$(MAKE)"  ) || exit $$?; done
	for a in $(APPS); do ( cp $(APP_OUTPUT_DIR)/nengo_$$a.aplx ../nengo_spinnaker/binaries/ ); done

docs : ${COMPLETE_SOURCE}
	doxygen

clean :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" clean  ) || exit $$?; done
	rm -rf ./docs/
	rm ../nengo_spinnaker/binaries/*.aplx
