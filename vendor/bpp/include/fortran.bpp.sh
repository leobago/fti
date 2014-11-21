################################################################################
# Copyright (c) 2013-2014, Julien Bigot - CEA (julien.bigot@cea.fr)
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

# default definition for FORTTYPES if unable to find config
FORTTYPES="CHARACTER1 COMPLEX4 COMPLEX8 COMPLEX16 INTEGER1 INTEGER2 INTEGER4 INTEGER8 LOGICAL1 LOGICAL2 LOGICAL4 LOGICAL8 REAL4 REAL8 REAL16"

source base.bpp.sh
CONFIG_FILE="${BPP_CONFIG:-config}.bpp.sh"
if [ -r "${CONFIG_FILE}" ]
then
	source "${CONFIG_FILE}"
fi

# Returns the Fortran kind associated to the type descriptor $1
function fort_kind {
	echo -n "$1" | sed 's/[^0-9]*//g'
}

# Returns the Fortran type associated to the type descriptor $1
function fort_ptype {
	echo -n "$1" | sed 's/[^a-zA-Z_]*//g'
}

# Returns the Fortran type associated to the type descriptor $1
function fort_type {
	echo -n "$(fort_ptype $1)"
	if [ -n "$(fort_kind $1)$2" ]
	then
		echo -n "("
	fi
	if [ -n "$(fort_kind $1)" ]
	then
		echo -n "KIND=$(fort_kind $1)"
	fi
	if [ -n "$(fort_kind $1)" -a -n "$2" ]
	then
		echo -n ","
	fi
	echo -n "$2"
	if [ -n "$(fort_kind $1)$2" ]
	then
		echo -n ")"
	fi
}

# Returns the size in bits of the Fortran type associated to the type descriptor $1
function fort_sizeof {
	KIND="$(fort_kind $1)"
	case "$1" in
	CHARACTER*|INTEGER*|LOGICAL*|REAL*)
		echo -n "$((KIND*8))"
		;;
	COMPLEX*)
		echo -n "$((2*KIND*8))"
		;;
	esac
}

# Returns an array descriptor for an assumed shape array of dimension $1
function array_desc() {
	str_repeat ':' 1 "$1" ',' '(' ')'
}

# Returns a format descriptor for I/O associated to the one letter type descriptor $1
function io_format {
	case "$1" in
	CHARACTER*)
		echo -n "A30"
		;;
	REAL*)
		echo -n "E30.5"
		;;
	INTEGER*)
		echo -n "I30"
		;;
	LOGICAL*)
		echo -n "L30"
		;;
	COMPLEX*)
		echo -n "(E15.5,E15.5)"
		;;
	esac
}
