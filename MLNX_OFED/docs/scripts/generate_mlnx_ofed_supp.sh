# !/bin/bash
#
# Copyright (c) 2014 Mellanox Technologies. All rights reserved.
#
# This Software is licensed under one of the following licenses:
#
# 1) under the terms of the "Common Public License 1.0" a copy of which is
#    available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/cpl.php.
#
# 2) under the terms of the "The BSD License" a copy of which is
#    available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/bsd-license.php.
#
# 3) under the terms of the "GNU General Public License (GPL) Version 2" a
#    copy of which is available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/gpl-license.php.
#
# Licensee has the right to choose one of the above licenses.
#
# Redistributions of source code must retain the above copyright
# notice and one of the license notices.
#
# Redistributions in binary form must reproduce both the above copyright
# notice, one of the license notices in the documentation
# and/or other materials provided with the distribution.
#
# Description:
# in order to generate suppression file run the following from the
# machine we want to run valgrind from:

# $./generate_mlnx_ofed_supp.sh > mlnx_ofed.supp 
#
# mlnx_ofed.supp is the suppression file we used when running valgrind


INSTALL_PREFIX="/usr/lib64"
MLNX_OBJ="obj:$INSTALL_PREFIX/lib*"
DIST="RedHat"

function get_dist()
{
	export distro_name=$(python -c 'import platform ; print platform.dist()[0]' | tr '[:upper:]' '[:lower:]')
	export distro_ver=$(python  -c 'import platform ; print platform.dist()[1]' | tr '[:upper:]' '[:lower:]')
	if [ "$distro_name" == "suse" ]; then
		patch_level=$(egrep PATCHLEVEL /etc/SuSE-release|cut -f2 -d=|sed -e "s/ //g")
		if [ -n "$patch_level" ]; then
			export distro_ver="${distro_ver}.${patch_level}"
		fi
	fi
	DIST=$distro_name
}

function get_prefix
{
	if [ -f /etc/debian_version ]; then
		INSTALL_PREFIX="/usr/lib"
	fi
}

# $1=installation_prifix
function print_supp_cond()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Cond"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=obj_or_func_to_remove $2=check
function print_supp_param()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Param"
	echo "   $2"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_free()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Free"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_leak()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Leak"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_value1()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Value1"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_value2()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Value2"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_value4()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Value4"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_value8()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Value8"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_addr1()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Addr1"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_addr2()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Addr2"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_addr4()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Addr4"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

# $1=installation_prifix
function print_supp_addr8()
{
	echo "{"
	echo "   <insert_a_suppression_name_here>"
	echo "   Memcheck:Addr8"
	echo "   ..."
	echo "   $1"
	echo "   ..."
	echo "}"
	echo
}

get_prefix
print_supp_cond    $MLNX_OBJ
print_supp_param  $MLNX_OBJ "mmap(length)"
print_supp_param  $MLNX_OBJ "munmap(start)"
print_supp_param  $MLNX_OBJ "munmap(length)"
print_supp_param  $MLNX_OBJ "write(buf)"
print_supp_free    $MLNX_OBJ
print_supp_leak    $MLNX_OBJ
print_supp_value1  $MLNX_OBJ
print_supp_value2  $MLNX_OBJ
print_supp_value4  $MLNX_OBJ
print_supp_value8  $MLNX_OBJ
print_supp_addr8   $MLNX_OBJ

