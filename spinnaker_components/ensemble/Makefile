# Project Settings
APP = nengo-ensemble
SOURCES = ensemble-harness.c ensemble-io.c ensemble-update.c ensemble-data.c
OBJECTS = $(SOURCES:.c=.o)
OBJECTS += ../common/bit_field.o ../common/spikes.o ../common/out_spikes.o ../common/configuration.o ../common/recording.o ../common/spin1-api-configuration.o
CFLAGS += -I ../common/

include ../Makefile.depend
