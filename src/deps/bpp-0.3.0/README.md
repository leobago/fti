# BPP

This is BPP, the Bash Pre-Processor.

BPP is useful in order to build clean Fortran90 interfaces. It allows to
generate Fortran code for all types, kinds, and array ranks supported by the
compiler.

## Usage

Support is provided for using BPP from CMake based projects, but you can use
it from plain Makefiles too.

There are two ways you can use BPP from your CMake project:
* with `add_subdirectory`: include BPP in your project and use it directly from
  there,
* with `find_package`: install BPP and use it as an external dependency of your
  project.

### CMake subdirectory usage

Using BPP with `add_subdirectory` is very simple.
Just copy the `bpp` directory in your source and point to it with
`add_subdirectory(bpp)`.
The `bpp_preprocess` then becomes available to process bpp files.

This is demonstrated in `example/cmake_subdirectory`.

### CMake find usage

Using BPP with `find_package` is no more complex.
If BPP is installed, just add a `find_package(Bpp REQUIRED)`.
The `bpp_preprocess` then becomes available to process bpp files.

This is demonstrated in `example/cmake_find`.

### GMake usage

Using BPP from a GNU Makefile is slightly less powerful than from CMake.
The types and kinds supported by the Fortran compiler will not be automatically
detected.
Predefined lists of supported types for well known compilers are provided
instead.

To use BPP from a Makefile, include the `share/bpp/bpp.mk` file (either from an
installed location or from a subdirectory in your project).
You can then set the `BPP_COMPILER_ID` variable to the compiler you use and
`.F90` files will be automatically generated from their `.F90.bpp` equivalent.
The `BPPFLAGS` variable is automatically passed to BPP similarly to `CFLAGS` or
`CXXFLAGS` for `cc` and `cxx`.

This is demonstrated in `example/cmake_makefile`.

## Installation

Installing BPP is very simple.
Inside the bpp directory, execute the following commands:
```
cmake -DCMAKE_INSTALL_PREFIX=/usr .
make install
```

Where the installation path is specified by the CMAKE_INSTALL_PREFIX parameter.

## Syntax

Lines starting with `!$SH` are interpreted as bash commands.

Any line not starting by this is written as-is.
If inside a bash control block (`if`, `for`, etc.), the output generation obeys
the control statement.

Bash-style variables `${VAR}` and escape sequences can be used in normal lines.

BPP provides a few standard headers that can be included with
`#!SH source <header.bpp.sh>`:
* `base.bpp.sh` provides the `str_repeat` and `str_repeat_reverse` bash
  functions to generate sequence of strings,
* `fortran.bpp.sh` provides
  - the `BPP_FORTTYPES` variable containing identifiers for all types supported
    by the compiler,
  - the function `fort_ptype` returns the type associated to an identifier,
  - the function `fort_kind` returns the kind associated to an identifier,
  - the function `fort_type` returns the full type (with kind included)
    associated to an identifier,
  - the function `fort_sizeof` returns the size in bytes associated to an
    identifier,
  - the function `io_format` returns an IO descriptor suitable for an
    identifier,
  - the function `array_desc` returns an assumed shaped array descriptor of the
    provided size,
* `hdf5_fortran.bpp.sh` provides the `HDF5TYPES` variable containing a list of 
  types identifiers suppored by HDF5 and the `hdf5_constant` function returning
  the associated HDF5 type constant associated.


## FAQ

Q. Isn't BPP redundant with assumed type parameters?

A.
The assumed type parameters functionality allows to implement part of what can
be done with BPP (support for all kinds of a type). However as of 2013 it was
not correctly supported on most compilers installed on the supercomputers.

In addition, many things can be done with BPP but not with assumed type
parameters, such as support for variable array rank or small variations of the
code depending on the kind.
