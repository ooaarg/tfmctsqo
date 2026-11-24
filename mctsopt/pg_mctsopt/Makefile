EXTENSION = pg_mctsopt
MODULE_big = pg_mctsopt
OBJS = pg_mctsopt.o backend.o trace.o search.o
DATA = pg_mctsopt--1.0.sql
REGRESS = basic threshold
REGRESS_OPTS = --dlpath=$(CURDIR)
PG_CPPFLAGS = -I../mctsopt_core/include
SHLIB_LINK = -lm
PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
# The portable engine and adapter use C11, including mixed declarations.
override CFLAGS += -std=c11 -Wno-declaration-after-statement -ffp-contract=off
backend.o: native.inc controls.inc mcts_internal.h mcts_trace.h ../mctsopt_core/include/mctsopt.h
search.o: ../mctsopt_core/src/search.c ../mctsopt_core/include/mctsopt.h
	$(CC) $(CFLAGS) $(CFLAGS_SL) $(CPPFLAGS) -c $< -o $@
