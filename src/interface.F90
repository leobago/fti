


module FTI

  use ISO_C_BINDING

  private


  type, public :: FTI_type

    private
    type(c_ptr) :: raw_type

  endtype FTI_type



  !> Token returned if a FTI function succeeds.
  integer, parameter :: FTI_SCES = 0
  !> Token returned if a FTI function fails.
  integer, parameter :: FTI_NSCS = -1



  type(FTI_type) :: FTI_CHARACTER1
  type(FTI_type) :: FTI_CHARACTER4
  type(FTI_type) :: FTI_COMPLEX4
  type(FTI_type) :: FTI_COMPLEX8
  type(FTI_type) :: FTI_COMPLEX16
  type(FTI_type) :: FTI_INTEGER1
  type(FTI_type) :: FTI_INTEGER2
  type(FTI_type) :: FTI_INTEGER4
  type(FTI_type) :: FTI_INTEGER8
  type(FTI_type) :: FTI_INTEGER16
  type(FTI_type) :: FTI_LOGICAL1
  type(FTI_type) :: FTI_LOGICAL2
  type(FTI_type) :: FTI_LOGICAL4
  type(FTI_type) :: FTI_LOGICAL8
  type(FTI_type) :: FTI_LOGICAL16
  type(FTI_type) :: FTI_REAL4
  type(FTI_type) :: FTI_REAL8
  type(FTI_type) :: FTI_REAL16



  public :: FTI_SCES, FTI_NSCS, &
      FTI_CHARACTER1, &
      FTI_CHARACTER4, &
      FTI_COMPLEX4, &
      FTI_COMPLEX8, &
      FTI_COMPLEX16, &
      FTI_INTEGER1, &
      FTI_INTEGER2, &
      FTI_INTEGER4, &
      FTI_INTEGER8, &
      FTI_INTEGER16, &
      FTI_LOGICAL1, &
      FTI_LOGICAL2, &
      FTI_LOGICAL4, &
      FTI_LOGICAL8, &
      FTI_LOGICAL16, &
      FTI_REAL4, &
      FTI_REAL8, &
      FTI_REAL16, &
      FTI_Init, FTI_Status, FTI_InitType, FTI_Protect,  &
      FTI_Checkpoint, FTI_Recover, FTI_Snapshot, FTI_Finalize



  interface

    function FTI_Init_impl(config_file, global_comm) &
            bind(c, name='FTI_Init_fort_wrapper')

      use ISO_C_BINDING

      integer(c_int) :: FTI_Init_impl
      character(c_char), intent(IN) :: config_file(*)
      integer(c_int), intent(INOUT) :: global_comm

    endfunction FTI_Init_impl

  endinterface



  interface

    function FTI_Status_impl() &
            bind(c, name='FTI_Status')

      use ISO_C_BINDING

      integer(c_int) :: FTI_Status_impl

    endfunction FTI_Status_impl

  endinterface



  interface

    function FTI_InitType_impl(type_F, size_F) &
            bind(c, name='FTI_InitType_wrapper')

      use ISO_C_BINDING

      integer(c_int) :: FTI_InitType_impl
      type(c_ptr), intent(OUT) :: type_F
      integer(c_int), value :: size_F

    endfunction FTI_InitType_impl

  endinterface



  interface

    function FTI_Protect_impl(id_F, ptr, count_F, type_F) &
            bind(c, name='FTI_Protect_wrapper')

      use ISO_C_BINDING

      integer(c_int) :: FTI_Protect_impl
      integer(c_int), value :: id_F
      type(c_ptr), value :: ptr
      integer(c_long), value :: count_F
      type(c_ptr), value :: type_F

    endfunction FTI_Protect_impl

  endinterface



  interface

    function FTI_Checkpoint_impl(id_F, level) &
            bind(c, name='FTI_Checkpoint')

      use ISO_C_BINDING

      integer(c_int) :: FTI_Checkpoint_impl
      integer(c_int), value :: id_F
      integer(c_int), value :: level

    endfunction FTI_Checkpoint_impl

  endinterface



  interface

    function FTI_Recover_impl() &
            bind(c, name='FTI_Recover')

      use ISO_C_BINDING

      integer(c_int) :: FTI_Recover_impl

    endfunction FTI_Recover_impl

  endinterface



  interface

    function FTI_Snapshot_impl() &
            bind(c, name='FTI_Snapshot')

      use ISO_C_BINDING

      integer(c_int) :: FTI_Snapshot_impl

    endfunction FTI_Snapshot_impl

  endinterface



  interface

    function FTI_Finalize_impl() &
            bind(c, name='FTI_Finalize')

      use ISO_C_BINDING

      integer(c_int) :: FTI_Finalize_impl

    endfunction FTI_Finalize_impl

  endinterface



  interface FTI_Protect

    module procedure FTI_Protect_Ptr
    module procedure FTI_Protect_CHARACTER10
    module procedure FTI_Protect_CHARACTER11
    module procedure FTI_Protect_CHARACTER12
    module procedure FTI_Protect_CHARACTER13
    module procedure FTI_Protect_CHARACTER14
    module procedure FTI_Protect_CHARACTER15
    module procedure FTI_Protect_CHARACTER16
    module procedure FTI_Protect_CHARACTER17
    module procedure FTI_Protect_CHARACTER40
    module procedure FTI_Protect_CHARACTER41
    module procedure FTI_Protect_CHARACTER42
    module procedure FTI_Protect_CHARACTER43
    module procedure FTI_Protect_CHARACTER44
    module procedure FTI_Protect_CHARACTER45
    module procedure FTI_Protect_CHARACTER46
    module procedure FTI_Protect_CHARACTER47
    module procedure FTI_Protect_COMPLEX40
    module procedure FTI_Protect_COMPLEX41
    module procedure FTI_Protect_COMPLEX42
    module procedure FTI_Protect_COMPLEX43
    module procedure FTI_Protect_COMPLEX44
    module procedure FTI_Protect_COMPLEX45
    module procedure FTI_Protect_COMPLEX46
    module procedure FTI_Protect_COMPLEX47
    module procedure FTI_Protect_COMPLEX80
    module procedure FTI_Protect_COMPLEX81
    module procedure FTI_Protect_COMPLEX82
    module procedure FTI_Protect_COMPLEX83
    module procedure FTI_Protect_COMPLEX84
    module procedure FTI_Protect_COMPLEX85
    module procedure FTI_Protect_COMPLEX86
    module procedure FTI_Protect_COMPLEX87
    module procedure FTI_Protect_COMPLEX160
    module procedure FTI_Protect_COMPLEX161
    module procedure FTI_Protect_COMPLEX162
    module procedure FTI_Protect_COMPLEX163
    module procedure FTI_Protect_COMPLEX164
    module procedure FTI_Protect_COMPLEX165
    module procedure FTI_Protect_COMPLEX166
    module procedure FTI_Protect_COMPLEX167
    module procedure FTI_Protect_INTEGER10
    module procedure FTI_Protect_INTEGER11
    module procedure FTI_Protect_INTEGER12
    module procedure FTI_Protect_INTEGER13
    module procedure FTI_Protect_INTEGER14
    module procedure FTI_Protect_INTEGER15
    module procedure FTI_Protect_INTEGER16
    module procedure FTI_Protect_INTEGER17
    module procedure FTI_Protect_INTEGER20
    module procedure FTI_Protect_INTEGER21
    module procedure FTI_Protect_INTEGER22
    module procedure FTI_Protect_INTEGER23
    module procedure FTI_Protect_INTEGER24
    module procedure FTI_Protect_INTEGER25
    module procedure FTI_Protect_INTEGER26
    module procedure FTI_Protect_INTEGER27
    module procedure FTI_Protect_INTEGER40
    module procedure FTI_Protect_INTEGER41
    module procedure FTI_Protect_INTEGER42
    module procedure FTI_Protect_INTEGER43
    module procedure FTI_Protect_INTEGER44
    module procedure FTI_Protect_INTEGER45
    module procedure FTI_Protect_INTEGER46
    module procedure FTI_Protect_INTEGER47
    module procedure FTI_Protect_INTEGER80
    module procedure FTI_Protect_INTEGER81
    module procedure FTI_Protect_INTEGER82
    module procedure FTI_Protect_INTEGER83
    module procedure FTI_Protect_INTEGER84
    module procedure FTI_Protect_INTEGER85
    module procedure FTI_Protect_INTEGER86
    module procedure FTI_Protect_INTEGER87
    module procedure FTI_Protect_INTEGER160
    module procedure FTI_Protect_INTEGER161
    module procedure FTI_Protect_INTEGER162
    module procedure FTI_Protect_INTEGER163
    module procedure FTI_Protect_INTEGER164
    module procedure FTI_Protect_INTEGER165
    module procedure FTI_Protect_INTEGER166
    module procedure FTI_Protect_INTEGER167
    module procedure FTI_Protect_LOGICAL10
    module procedure FTI_Protect_LOGICAL11
    module procedure FTI_Protect_LOGICAL12
    module procedure FTI_Protect_LOGICAL13
    module procedure FTI_Protect_LOGICAL14
    module procedure FTI_Protect_LOGICAL15
    module procedure FTI_Protect_LOGICAL16
    module procedure FTI_Protect_LOGICAL17
    module procedure FTI_Protect_LOGICAL20
    module procedure FTI_Protect_LOGICAL21
    module procedure FTI_Protect_LOGICAL22
    module procedure FTI_Protect_LOGICAL23
    module procedure FTI_Protect_LOGICAL24
    module procedure FTI_Protect_LOGICAL25
    module procedure FTI_Protect_LOGICAL26
    module procedure FTI_Protect_LOGICAL27
    module procedure FTI_Protect_LOGICAL40
    module procedure FTI_Protect_LOGICAL41
    module procedure FTI_Protect_LOGICAL42
    module procedure FTI_Protect_LOGICAL43
    module procedure FTI_Protect_LOGICAL44
    module procedure FTI_Protect_LOGICAL45
    module procedure FTI_Protect_LOGICAL46
    module procedure FTI_Protect_LOGICAL47
    module procedure FTI_Protect_LOGICAL80
    module procedure FTI_Protect_LOGICAL81
    module procedure FTI_Protect_LOGICAL82
    module procedure FTI_Protect_LOGICAL83
    module procedure FTI_Protect_LOGICAL84
    module procedure FTI_Protect_LOGICAL85
    module procedure FTI_Protect_LOGICAL86
    module procedure FTI_Protect_LOGICAL87
    module procedure FTI_Protect_LOGICAL160
    module procedure FTI_Protect_LOGICAL161
    module procedure FTI_Protect_LOGICAL162
    module procedure FTI_Protect_LOGICAL163
    module procedure FTI_Protect_LOGICAL164
    module procedure FTI_Protect_LOGICAL165
    module procedure FTI_Protect_LOGICAL166
    module procedure FTI_Protect_LOGICAL167
    module procedure FTI_Protect_REAL40
    module procedure FTI_Protect_REAL41
    module procedure FTI_Protect_REAL42
    module procedure FTI_Protect_REAL43
    module procedure FTI_Protect_REAL44
    module procedure FTI_Protect_REAL45
    module procedure FTI_Protect_REAL46
    module procedure FTI_Protect_REAL47
    module procedure FTI_Protect_REAL80
    module procedure FTI_Protect_REAL81
    module procedure FTI_Protect_REAL82
    module procedure FTI_Protect_REAL83
    module procedure FTI_Protect_REAL84
    module procedure FTI_Protect_REAL85
    module procedure FTI_Protect_REAL86
    module procedure FTI_Protect_REAL87
    module procedure FTI_Protect_REAL160
    module procedure FTI_Protect_REAL161
    module procedure FTI_Protect_REAL162
    module procedure FTI_Protect_REAL163
    module procedure FTI_Protect_REAL164
    module procedure FTI_Protect_REAL165
    module procedure FTI_Protect_REAL166
    module procedure FTI_Protect_REAL167

  endinterface FTI_Protect


contains

  subroutine FTI_Init(config_file, global_comm, err)

    include 'mpif.h'

    character(len=*), intent(IN) :: config_file
    integer, intent(INOUT) :: global_comm
    integer, intent(OUT) :: err

    character, target, dimension(1:len_trim(config_file)+1) :: config_file_c
    integer :: ii, ll
    integer(c_int) :: global_comm_c

    ll = len_trim(config_file)
    do ii = 1, ll
      config_file_c(ii) = config_file(ii:ii)
    enddo
    config_file_c(ll+1) = c_null_char
    global_comm_c = int(global_comm, c_int)
    err = int(FTI_Init_impl(config_file_c, global_comm_c))
    global_comm = int(global_comm_c)
    if (err /= FTI_SCES ) then
      return
    endif

    call FTI_InitType(FTI_CHARACTER1, int(8_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_CHARACTER4, int(32_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_COMPLEX4, int(64_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_COMPLEX8, int(128_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_COMPLEX16, int(256_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_INTEGER1, int(8_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_INTEGER2, int(16_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_INTEGER4, int(32_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_INTEGER8, int(64_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_INTEGER16, int(128_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_LOGICAL1, int(8_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_LOGICAL2, int(16_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_LOGICAL4, int(32_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_LOGICAL8, int(64_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_LOGICAL16, int(128_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_REAL4, int(32_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_REAL8, int(64_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif
    call FTI_InitType(FTI_REAL16, int(128_C_int/8_c_int, C_int), err)
    if (err /= FTI_SCES ) then
      return
    endif

  endsubroutine FTI_Init



  subroutine FTI_Status(status)

    integer, intent(OUT) :: status

    status = int(FTI_Status_impl())

  endsubroutine FTI_Status



  subroutine FTI_InitType(type_F, size_F, err)

    type(FTI_type), intent(OUT) :: type_F
    integer, intent(IN) :: size_F
    integer, intent(OUT) :: err

    err = int(FTI_InitType_impl(type_F%raw_type, int(size_F, c_int)))

  endsubroutine FTI_InitType



  subroutine FTI_Protect_Ptr(id_F, ptr, count_F, type_F, err)

    integer, intent(IN) :: id_F
    type(c_ptr), value :: ptr
    integer, intent(IN) :: count_F
    type(FTI_type), intent(IN) :: type_F
    integer, intent(OUT) :: err

    err = int(FTI_Protect_impl(int(id_F, c_int), ptr, int(count_F, c_long), &
            type_F%raw_type))

  endsubroutine FTI_Protect_Ptr



  subroutine FTI_Protect_CHARACTER10(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER10



  subroutine FTI_Protect_CHARACTER11(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER11



  subroutine FTI_Protect_CHARACTER12(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER12



  subroutine FTI_Protect_CHARACTER13(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER13



  subroutine FTI_Protect_CHARACTER14(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER14



  subroutine FTI_Protect_CHARACTER15(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER15



  subroutine FTI_Protect_CHARACTER16(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER16



  subroutine FTI_Protect_CHARACTER17(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=1), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_CHARACTER1, err)

  endsubroutine FTI_Protect_CHARACTER17



  subroutine FTI_Protect_CHARACTER40(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER40



  subroutine FTI_Protect_CHARACTER41(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER41



  subroutine FTI_Protect_CHARACTER42(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER42



  subroutine FTI_Protect_CHARACTER43(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER43



  subroutine FTI_Protect_CHARACTER44(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER44



  subroutine FTI_Protect_CHARACTER45(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER45



  subroutine FTI_Protect_CHARACTER46(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER46



  subroutine FTI_Protect_CHARACTER47(id_F, data, err)

    integer, intent(IN) :: id_F
    CHARACTER(KIND=4), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_CHARACTER4, err)

  endsubroutine FTI_Protect_CHARACTER47



  subroutine FTI_Protect_COMPLEX40(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX40



  subroutine FTI_Protect_COMPLEX41(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX41



  subroutine FTI_Protect_COMPLEX42(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX42



  subroutine FTI_Protect_COMPLEX43(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX43



  subroutine FTI_Protect_COMPLEX44(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX44



  subroutine FTI_Protect_COMPLEX45(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX45



  subroutine FTI_Protect_COMPLEX46(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX46



  subroutine FTI_Protect_COMPLEX47(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=4), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_COMPLEX4, err)

  endsubroutine FTI_Protect_COMPLEX47



  subroutine FTI_Protect_COMPLEX80(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX80



  subroutine FTI_Protect_COMPLEX81(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX81



  subroutine FTI_Protect_COMPLEX82(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX82



  subroutine FTI_Protect_COMPLEX83(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX83



  subroutine FTI_Protect_COMPLEX84(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX84



  subroutine FTI_Protect_COMPLEX85(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX85



  subroutine FTI_Protect_COMPLEX86(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX86



  subroutine FTI_Protect_COMPLEX87(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=8), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_COMPLEX8, err)

  endsubroutine FTI_Protect_COMPLEX87



  subroutine FTI_Protect_COMPLEX160(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX160



  subroutine FTI_Protect_COMPLEX161(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX161



  subroutine FTI_Protect_COMPLEX162(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX162



  subroutine FTI_Protect_COMPLEX163(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX163



  subroutine FTI_Protect_COMPLEX164(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX164



  subroutine FTI_Protect_COMPLEX165(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX165



  subroutine FTI_Protect_COMPLEX166(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX166



  subroutine FTI_Protect_COMPLEX167(id_F, data, err)

    integer, intent(IN) :: id_F
    COMPLEX(KIND=16), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_COMPLEX16, err)

  endsubroutine FTI_Protect_COMPLEX167



  subroutine FTI_Protect_INTEGER10(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER10



  subroutine FTI_Protect_INTEGER11(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER11



  subroutine FTI_Protect_INTEGER12(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER12



  subroutine FTI_Protect_INTEGER13(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER13



  subroutine FTI_Protect_INTEGER14(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER14



  subroutine FTI_Protect_INTEGER15(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER15



  subroutine FTI_Protect_INTEGER16(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER16



  subroutine FTI_Protect_INTEGER17(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=1), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_INTEGER1, err)

  endsubroutine FTI_Protect_INTEGER17



  subroutine FTI_Protect_INTEGER20(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER20



  subroutine FTI_Protect_INTEGER21(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER21



  subroutine FTI_Protect_INTEGER22(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER22



  subroutine FTI_Protect_INTEGER23(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER23



  subroutine FTI_Protect_INTEGER24(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER24



  subroutine FTI_Protect_INTEGER25(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER25



  subroutine FTI_Protect_INTEGER26(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER26



  subroutine FTI_Protect_INTEGER27(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=2), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_INTEGER2, err)

  endsubroutine FTI_Protect_INTEGER27



  subroutine FTI_Protect_INTEGER40(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER40



  subroutine FTI_Protect_INTEGER41(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER41



  subroutine FTI_Protect_INTEGER42(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER42



  subroutine FTI_Protect_INTEGER43(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER43



  subroutine FTI_Protect_INTEGER44(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER44



  subroutine FTI_Protect_INTEGER45(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER45



  subroutine FTI_Protect_INTEGER46(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER46



  subroutine FTI_Protect_INTEGER47(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=4), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_INTEGER4, err)

  endsubroutine FTI_Protect_INTEGER47



  subroutine FTI_Protect_INTEGER80(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER80



  subroutine FTI_Protect_INTEGER81(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER81



  subroutine FTI_Protect_INTEGER82(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER82



  subroutine FTI_Protect_INTEGER83(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER83



  subroutine FTI_Protect_INTEGER84(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER84



  subroutine FTI_Protect_INTEGER85(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER85



  subroutine FTI_Protect_INTEGER86(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER86



  subroutine FTI_Protect_INTEGER87(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=8), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_INTEGER8, err)

  endsubroutine FTI_Protect_INTEGER87



  subroutine FTI_Protect_INTEGER160(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER160



  subroutine FTI_Protect_INTEGER161(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER161



  subroutine FTI_Protect_INTEGER162(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER162



  subroutine FTI_Protect_INTEGER163(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER163



  subroutine FTI_Protect_INTEGER164(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER164



  subroutine FTI_Protect_INTEGER165(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER165



  subroutine FTI_Protect_INTEGER166(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER166



  subroutine FTI_Protect_INTEGER167(id_F, data, err)

    integer, intent(IN) :: id_F
    INTEGER(KIND=16), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_INTEGER16, err)

  endsubroutine FTI_Protect_INTEGER167



  subroutine FTI_Protect_LOGICAL10(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL10



  subroutine FTI_Protect_LOGICAL11(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL11



  subroutine FTI_Protect_LOGICAL12(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL12



  subroutine FTI_Protect_LOGICAL13(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL13



  subroutine FTI_Protect_LOGICAL14(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL14



  subroutine FTI_Protect_LOGICAL15(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL15



  subroutine FTI_Protect_LOGICAL16(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL16



  subroutine FTI_Protect_LOGICAL17(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=1), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_LOGICAL1, err)

  endsubroutine FTI_Protect_LOGICAL17



  subroutine FTI_Protect_LOGICAL20(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL20



  subroutine FTI_Protect_LOGICAL21(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL21



  subroutine FTI_Protect_LOGICAL22(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL22



  subroutine FTI_Protect_LOGICAL23(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL23



  subroutine FTI_Protect_LOGICAL24(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL24



  subroutine FTI_Protect_LOGICAL25(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL25



  subroutine FTI_Protect_LOGICAL26(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL26



  subroutine FTI_Protect_LOGICAL27(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=2), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_LOGICAL2, err)

  endsubroutine FTI_Protect_LOGICAL27



  subroutine FTI_Protect_LOGICAL40(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL40



  subroutine FTI_Protect_LOGICAL41(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL41



  subroutine FTI_Protect_LOGICAL42(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL42



  subroutine FTI_Protect_LOGICAL43(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL43



  subroutine FTI_Protect_LOGICAL44(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL44



  subroutine FTI_Protect_LOGICAL45(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL45



  subroutine FTI_Protect_LOGICAL46(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL46



  subroutine FTI_Protect_LOGICAL47(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=4), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_LOGICAL4, err)

  endsubroutine FTI_Protect_LOGICAL47



  subroutine FTI_Protect_LOGICAL80(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL80



  subroutine FTI_Protect_LOGICAL81(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL81



  subroutine FTI_Protect_LOGICAL82(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL82



  subroutine FTI_Protect_LOGICAL83(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL83



  subroutine FTI_Protect_LOGICAL84(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL84



  subroutine FTI_Protect_LOGICAL85(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL85



  subroutine FTI_Protect_LOGICAL86(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL86



  subroutine FTI_Protect_LOGICAL87(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=8), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_LOGICAL8, err)

  endsubroutine FTI_Protect_LOGICAL87



  subroutine FTI_Protect_LOGICAL160(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL160



  subroutine FTI_Protect_LOGICAL161(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL161



  subroutine FTI_Protect_LOGICAL162(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL162



  subroutine FTI_Protect_LOGICAL163(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL163



  subroutine FTI_Protect_LOGICAL164(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL164



  subroutine FTI_Protect_LOGICAL165(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL165



  subroutine FTI_Protect_LOGICAL166(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL166



  subroutine FTI_Protect_LOGICAL167(id_F, data, err)

    integer, intent(IN) :: id_F
    LOGICAL(KIND=16), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_LOGICAL16, err)

  endsubroutine FTI_Protect_LOGICAL167



  subroutine FTI_Protect_REAL40(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL40



  subroutine FTI_Protect_REAL41(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL41



  subroutine FTI_Protect_REAL42(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL42



  subroutine FTI_Protect_REAL43(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL43



  subroutine FTI_Protect_REAL44(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL44



  subroutine FTI_Protect_REAL45(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL45



  subroutine FTI_Protect_REAL46(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL46



  subroutine FTI_Protect_REAL47(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=4), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_REAL4, err)

  endsubroutine FTI_Protect_REAL47



  subroutine FTI_Protect_REAL80(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL80



  subroutine FTI_Protect_REAL81(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL81



  subroutine FTI_Protect_REAL82(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL82



  subroutine FTI_Protect_REAL83(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL83



  subroutine FTI_Protect_REAL84(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL84



  subroutine FTI_Protect_REAL85(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL85



  subroutine FTI_Protect_REAL86(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL86



  subroutine FTI_Protect_REAL87(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=8), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_REAL8, err)

  endsubroutine FTI_Protect_REAL87



  subroutine FTI_Protect_REAL160(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data
    integer, intent(OUT) :: err

    call FTI_Protect_Ptr(id_F, c_loc(data), 1, FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL160



  subroutine FTI_Protect_REAL161(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data(:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1))), &
            size(data), FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL161



  subroutine FTI_Protect_REAL162(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data(:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2))), &
            size(data), FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL162



  subroutine FTI_Protect_REAL163(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data(:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3))), &
            size(data), FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL163



  subroutine FTI_Protect_REAL164(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data(:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4))), &
            size(data), FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL164



  subroutine FTI_Protect_REAL165(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data(:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5))), &
            size(data), FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL165



  subroutine FTI_Protect_REAL166(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data(:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6))), &
            size(data), FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL166



  subroutine FTI_Protect_REAL167(id_F, data, err)

    integer, intent(IN) :: id_F
    REAL(KIND=16), pointer :: data(:,:,:,:,:,:,:)
    integer, intent(OUT) :: err

    ! workaround, we take the address of the first array element and hope for
    ! the best since not much better can be done
    call FTI_Protect_Ptr(id_F, &
            c_loc(data(lbound(data, 1),&
lbound(data, 2),&
lbound(data, 3),&
lbound(data, 4),&
lbound(data, 5),&
lbound(data, 6),&
lbound(data, 7))), &
            size(data), FTI_REAL16, err)

  endsubroutine FTI_Protect_REAL167



  subroutine FTI_Checkpoint(id_F, level, err)

    integer, intent(IN) :: id_F
    integer, intent(IN) :: level
    integer, intent(OUT) :: err

    err = int(FTI_Checkpoint_impl(int(id_F, c_int), int(level, c_int)))

  endsubroutine FTI_Checkpoint



  subroutine FTI_Recover(err)

    integer, intent(OUT) :: err

    err = int(FTI_Recover_impl())

  endsubroutine FTI_Recover



  subroutine FTI_Snapshot(err)

    integer, intent(OUT) :: err

    err = int(FTI_Snapshot_impl())

  endsubroutine FTI_Snapshot



  subroutine FTI_Finalize(err)

    integer, intent(OUT) :: err

    err = int(FTI_Finalize_impl())

  endsubroutine FTI_Finalize

endmodule FTI
