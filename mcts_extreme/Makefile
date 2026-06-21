EXTENSION = mcts_extreme
EXTVERSION = 1.0

MODULE_big = mcts_extreme

OBJS = \
	core/mcts_extreme.o \
	mcts/mcts.o \
	mcts/local_search.o \
	mcts/trace.o

DATA = mcts_extreme--1.0.sql
PGFILEDESC = "mcts_extreme - MCTS join-order search (UCT-Extreme + Luby restarts)"

REGRESS = basic

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
