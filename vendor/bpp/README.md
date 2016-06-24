# BPP

This is BPP, the Bash Pre-Processor.

BPP is useful in order to build clean Fortran90 interfaces. It allows to
generate Fortran code for all types, kinds, and array ranks supported by the
compiler.

## Usage

There are two ways you can use BPP from your project:
*  **In-place**: you can include Bpp in your project and use it directly from
   there,
*  **Dependancy**: or you can use Bpp as an external dependancy of your
   project.

Support is provided for using Bpp from Cmake based projects, but you can use
it from plain Makefiles too.

One feature that is available to Cmake projects only is the ability to
automatically detect the types supported by the Fortran compiler. For plain
Makefiles projects, predefined lists of supported types for well known
compilers are provided. The user has to manually choose one.

### In-place usage

Using BPP in-place is very simple, just use add_subdirectory from cmake or use
the in source script from a Makefile.

The directories `cmake_local` and `makefile_local` in the example directory
demonstrate how to do that.

### Dependancy usage

Using BPP as a dependancy is very simple too, just use find_package from cmake
or use the installed script from a Makefile. You can use a system-wide
installed BPP or you can just configure BPP as a user, without installation.
It should be found anyway.

The directories `cmake_global` and `makefile_global` in the example directory
demonstrate how to do that.

## Installation

Installing Bpp is very simple.
Inside the bpp directory, execute the following commands:
```
cmake .
make install
```

In order to change the installation path for your project, set the
CMAKE_INSTALL_PREFIX cmake parameter:
```
cmake -DCMAKE_INSTALL_PREFIX=/usr .
```

## FAQ

Q. Isn't BPP redundant with assumed type parameters?

A.
The assumed type parameters functionality allows to implement part of what can
be done with BPP (support for all kinds of a type). However as of 2013 it was
not correctly supported on most compilers installed on the supercomputers.

In addition, many things can be done with BPP but not with assumed type
parameters, such as support for variable array rank or small variations of the
code depending on the kind.