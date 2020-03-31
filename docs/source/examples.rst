.. Fault Tolerance Library documentation Examples file
.. _examples:

FTI Examples
==================


Using FTI_Snapshot
------------------

.. code-block:: C

   #include <stdlib.h>
   #include <fti.h>

   int main(int argc, char** argv){
       MPI_Init(&argc, &argv);
       char* path = "config.fti"; //config file path
       FTI_Init(path, MPI_COMM_WORLD);
       int world_rank, world_size; //FTI_COMM rank & size
       MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
       MPI_Comm_size(FTI_COMM_WORLD, &world_size);

       int *array = malloc(sizeof(int) * world_size);
       int number = world_rank;
       int i = 0;
       //adding variables to protect
       FTI_Protect(1, &i, 1, FTI_INTG);
       FTI_Protect(2, &number, 1, FTI_INTG);
       for (; i < 100; i++) {
           FTI_Snapshot();
           MPI_Allgather(&number, 1, MPI_INT, array,
                 1, MPI_INT, FTI_COMM_WORLD);
       number += 1;
       }
       free(array);
       FTI_Finalize();
       MPI_Finalize();
       return 0;
   }

**DESCRIPTION**  

..

   ``FTI_Snapshot()`` makes a checkpoint by given time and also recovers data after a failure, thus makes the code shorter. Checkpoints intervals can be set in configuration file (see: `ckpt_L1 <Configuration#ckpt_l1>`_ - `ckpt_L4 <Configuration#ckpt_l4>`_\ ).  


Using FTI_Checkpoint
--------------------

.. code-block:: C

   #include <stdlib.h>
   #include <fti.h>
   #define ITER_CHECK 10

   int main(int argc, char** argv){
       MPI_Init(&argc, &argv);
       char* path = "config.fti"; //config file path
       FTI_Init(path, MPI_COMM_WORLD);
       int world_rank, world_size; //FTI_COMM rank & size
       MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
       MPI_Comm_size(FTI_COMM_WORLD, &world_size);

       int *array = malloc(sizeof(int) * world_size);
       int number = world_rank;
       int i = 0;
       //adding variables to protect
       FTI_Protect(1, &i, 1, FTI_INTG);
       FTI_Protect(2, &number, 1, FTI_INTG);
       if (FTI_Status() != 0) {
           FTI_Recover();
       }
       for (; i < 100; i++) {
           if (i % ITER_CHECK == 0) {
               FTI_Checkpoint(i / ITER_CHECK + 1, 2);
           }
           MPI_Allgather(&number, 1, MPI_INT, array,
                 1, MPI_INT, FTI_COMM_WORLD);
       number += 1;
       }
       free(array);
       FTI_Finalize();
       MPI_Finalize();
       return 0;
   }

**DESCRIPTION**  

..

   ``FTI_Checkpoint()`` allows to checkpoint at precise application intervals. Note that when using ``FTI_Checkpoint()``\ , ``ckpt_L1``\ , ``ckpt_L2``\ , ``ckpt_L3`` and ``ckpt_L4`` are not taken into account.


Using FTI_Realloc with Fortran and Intrinsic Types
--------------------------------------------------

.. code-block:: Fortran

   program test_fti_realloc
       use fti
       use iso_c_binding
       implicit none
       include 'mpif.h'

       integer, parameter          :: dp=kind(1.0d0)
       integer, parameter          :: N1=128*1024*25  !> 25 MB / Process
       integer, parameter          :: N2=128*1024*50  !> 50 MB / Process
       integer, parameter          :: N11 = 128       
       integer, parameter          :: N12 = 1024
       integer, parameter          :: N13 = 25
       integer, parameter          :: N21 = 128       
       integer, parameter          :: N22 = 1024
       integer, parameter          :: N23 = 50
       integer, target             :: FTI_COMM_WORLD
       integer                     :: ierr, status

       real(dp), dimension(:,:,:), pointer :: arr
       type(c_ptr)             :: arr_c_ptr
       real(dp), dimension(:,:,:), pointer :: tmp
       integer(4), dimension(:), pointer   :: shape

       allocate(arr(N11,N12,N13))
       allocate(shape(3))

       !> INITIALIZE MPI AND FTI    
       call MPI_Init(ierr)
       FTI_COMM_WORLD = MPI_COMM_WORLD
       call FTI_Init('config.fti', FTI_COMM_WORLD, ierr)

       !> PROTECT DATA AND ITS SHAPE
       call FTI_Protect(0, arr, ierr)
       call FTI_Protect(1, shape, ierr)

       call FTI_Status(status)

       !> EXECUTE ON RESTART
       if ( status .eq. 1 ) then
           !> REALLOCATE TO SIZE AT CHECKPOINT
           arr_c_ptr = c_loc(arr(1,1,1))
           call FTI_Realloc(0, arr_c_ptr, ierr)
           call FTI_recover(ierr)
           !> RESHAPE ARRAY
           call c_f_pointer(arr_c_ptr, arr, shape)
           call FTI_Finalize(ierr)
           call MPI_Finalize(ierr)
           STOP
       end if

       !> FIRST CHECKPOINT
       call FTI_Checkpoint(1, 1, ierr)

       !> CHANGE ARRAY DIMENSION
       !> AND STORE IN SHAPE ARRAY
       shape = [N21,N22,N23]
       allocate(tmp(N21,N22,N23))
       tmp(1:N11,1:N12,1:N13) = arr
       deallocate(arr)
       arr => tmp

       !> TELL FTI ABOUT THE NEW DIMENSION
       call FTI_Protect(0, arr, ierr)

       !> SECOND CHECKPOINT
       call FTI_Checkpoint(2,1, ierr)

       !> SIMULATE CRASH
       call MPI_Abort(MPI_COMM_WORLD,-1,ierr)
   end program

Using FTI_Realloc with Fortran and Derived Types
------------------------------------------------

.. code-block:: Fortran

   program test_fti_realloc
       use fti
       use iso_c_binding
       implicit none
       include 'mpif.h'

       !> DEFINE DERIVED TYPE
       type :: polar 
           real :: radius
           real :: phi
       end type

       integer, parameter          :: dp=kind(1.0d0)
       integer, parameter          :: N1=128*1024*25  !> 25 MB / Process
       integer, parameter          :: N2=128*1024*50  !> 50 MB / Process
       integer, parameter          :: N11 = 128       
       integer, parameter          :: N12 = 1024
       integer, parameter          :: N13 = 25
       integer, parameter          :: N21 = 128       
       integer, parameter          :: N22 = 1024
       integer, parameter          :: N23 = 50
       integer, target             :: FTI_COMM_WORLD
       integer                     :: ierr, status
       type(FTI_type)              :: FTI_Polar

       type(c_ptr)                            :: cPtr
       type(polar), dimension(:,:,:), pointer :: arr
       type(polar), dimension(:,:,:), pointer :: tmp
       integer(4), dimension(:), pointer      :: shape

       !> INITIALIZE FTI TYPE 'FTI_POLAR'
       call FTI_InitType(FTI_Polar, 2*4, ierr)

       allocate(arr(N11,N12,N13))
       allocate(shape(3))

       !> INITIALIZE MPI AND FTI
       call MPI_Init(ierr)
       FTI_COMM_WORLD = MPI_COMM_WORLD
       call FTI_Init('config.fti', FTI_COMM_WORLD, ierr)

       !> PROTECT DATA AND ITS SHAPE
       call FTI_Protect(0, c_loc(arr), size(arr),FTI_Polar, ierr)
       call FTI_Protect(1, shape, ierr)

       call FTI_Status(status)

       !> EXECUTE ON RESTART
       if ( status .eq. 1 ) then
           !> REALLOCATE TO DIMENSION AT LAST CHECKPOINT
           cPtr = c_loc(arr)
           call FTI_Realloc(0, cPtr, ierr) !> PASS DATA AS C-POINTER
           call FTI_recover(ierr)
           call c_f_pointer(cPtr, arr, shape) !> CAST BACK TO F-POINTER
           call FTI_Finalize(ierr)
           call MPI_Finalize(ierr)
           STOP
       end if

       !> FIRST CHECKPOINT
       call FTI_Checkpoint(1, 1, ierr)

       !> CHANGE ARRAY DIMENSION
       !> AND STORE IN SHAPE ARRAY
       shape = [N21,N22,N23]
       allocate(tmp(N21,N22,N23))
       tmp(1:N11,1:N12,1:N13) = arr
       deallocate(arr)
       arr => tmp

       !> TELL FTI ABOUT THE NEW DIMENSION
       call FTI_Protect(0, c_loc(arr), size(arr), FTI_Polar, ierr)

       !> SECOND CHECKPOINT
       call FTI_Checkpoint(2,1, ierr)

       !> SIMULATE CRASH
       call MPI_Abort(MPI_COMM_WORLD,-1,ierr)
   end program
