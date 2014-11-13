## * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
##* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
##=======================================================================
##Copyright (C) 2010-2014 Leonardo A. BAUTISTA GOMEZ
##This program is free software; you can redistribute it and/or modify
##it under the terms of the GNU General Public License (GPL) as published
##of the License, or (at your option) any later version.
##
##This program is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU General Public License for more details.
##
##To read the license please visit http://www.gnu.org/copyleft/gpl.html
##=======================================================================

##=======================================================================
##   PLEASE SET THESE VARIABLES BEFORE COMPILING
##=======================================================================

FTIPATH		?= /path/to/fti/install/directory/FTI
BUILDPATH	?= .

##=======================================================================
##   DIRECTORY TREE
##=======================================================================

LIB 		= $(BUILDPATH)/lib
OBJ		= $(BUILDPATH)/obj
SRC		= src
DOC		= doc
INC		= include

##=======================================================================
##   COMPILERS
##=======================================================================

CC 		?= cc
F90 		?= f95
MPICC 		?= mpicc
MPIF90		?= mpif90

##=======================================================================
##   FLAGS
##=======================================================================

FTIFLAGS	= -fPIC -g -Iinclude/

##=======================================================================
##   TARGETS
##=======================================================================


OBJS		= $(OBJ)/galois.o $(OBJ)/jerasure.o \
		  $(OBJ)/dictionary.o $(OBJ)/iniparser.o \
		  $(OBJ)/checkpoint.o $(OBJ)/postckpt.o\
		  $(OBJ)/recover.o $(OBJ)/postreco.o\
		  $(OBJ)/topo.o $(OBJ)/conf.o $(OBJ)/meta.o \
		  $(OBJ)/tools.o $(OBJ)/api.o

.PRECIOUS: $(OBJ)/interface.F90

OBJS_F90	= $(OBJ)/interface.o $(OBJ)/ftif.o

SHARED		= libfti.so
STATIC		= libfti.a
SHARED_F90	= libfti_f90.so
STATIC_F90	= libfti_f90.a

all: $(LIB)/$(SHARED) $(LIB)/$(STATIC) $(LIB)/$(SHARED_F90) $(LIB)/$(STATIC_F90)

doc:
		@mkdir -p $(DOC)
		doxygen $(DOC)/Doxyfile

$(OBJ)/%.o: $(SRC)/%.c
		@mkdir -p $(OBJ)
		$(MPICC) $(FTIFLAGS) -c $< -o $@

$(OBJ)/%.F90: $(SRC)/%.F90.bpp
		@mkdir -p $(OBJ)
		./bpp $< $@

$(OBJ)/%.o: $(OBJ)/%.F90
		@mkdir -p $(OBJ)
		@mkdir -p $(LIB)
		$(MPIF90) $(FTIFLAGS) -c $< -o $@
		mv *.mod $(INC)/

$(LIB)/$(SHARED): $(OBJS)
		@mkdir -p $(LIB)
		$(CC) -shared -o $@ $(OBJS) -lc

$(LIB)/$(SHARED_F90): $(OBJS_F90) $(LIB)/$(SHARED)
		@mkdir -p $(LIB)
		$(F90) -shared -o $@ -L$(LIB) -lfti $(OBJS_F90)

$(LIB)/$(STATIC): $(OBJS)
		@mkdir -p $(LIB)
		$(RM) $@
		$(AR) -cvq $@ $(OBJS)

$(LIB)/$(STATIC_F90): $(OBJS_F90)
		@mkdir -p $(LIB)
		$(RM) $@
		$(AR) -cvq $@ $(OBJS_F90)

install: $(LIB)/$(SHARED) $(LIB)/$(STATIC) $(LIB)/$(SHARED_F90) $(LIB)/$(STATIC_F90)
		install -d $(FTIPATH)/lib  $(FTIPATH)/include
		install $(INC)/* $(FTIPATH)/include/
		install $(LIB)/* $(FTIPATH)/lib/

uninstall:
		$(RM) $(FTIPATH)/$(LIB)/* $(FTIPATH)/$(INC)/*
		if [ -d "$(FTIPATH)/$(LIB)" ]; then rmdir $(FTIPATH)/$(LIB); fi
		if [ -d "$(FTIPATH)/$(INC)" ]; then rmdir $(FTIPATH)/$(INC); fi
		if [ -d "$(FTIPATH)" ]; then rmdir $(FTIPATH); fi

clean:
		$(RM) $(OBJ)/* $(LIB)/*

.PHONY:		doc install uninstall clean
