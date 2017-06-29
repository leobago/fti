!> @author Kai Keller (kellekai@gmx.de)
!> @date   June, 2017
!> @brief  Unitary test program for FTI Fortran
!! 
!! The purpose of this code is to test the checkpoint
!! and restart mechanism in FTI for any case.
!!
!! [ USAGE ]
!! 
!! ./PROGRAM ARG1=config ARG2=crash ARG3=level ARG4=diff
!!
!> @param ARG1 relative/absolute path to FTI config file 
!> @param ARG2 1/0 (on/off) - simulate failure  
!> @param ARG3 checkpoint level (1 - 4) 
!> @param ARG4 1/0 (on/off) - enable/disable different ckpt sizes  

MODULE CHECK_FUNCTIONS
   IMPLICIT NONE
   PRIVATE

   PUBLIC INIT_ARRAYS, VALIDIFY, WRITE_DATA, READ_DATA, &
      BUFFER, IERROR

   INTEGER, PARAMETER      :: BUFFER = 256
   INTEGER                 :: IERROR

   CONTAINS
   
   SUBROUTINE INIT_ARRAYS ( A, B )
      REAL, INTENT(OUT)    :: A(:), B(:)
      INTEGER              :: VALUES(1:8), K, I=1
      INTEGER, DIMENSION(:), ALLOCATABLE :: SEED
   
      CALL DATE_AND_TIME(VALUES=VALUES)
      CALL RANDOM_SEED(SIZE=K)
      ALLOCATE(SEED(1:K))
      SEED(:) = VALUES(8)
      CALL RANDOM_SEED(PUT=SEED)
   
      A = 1.0
   
      DO WHILE (I <= SIZE(B))
      CALL RANDOM_NUMBER(B(I))
      I = I+1
      END DO
   END SUBROUTINE 
   
   SUBROUTINE VALIDIFY ( A, B_CHK, ASIZE, RES )
      REAL, INTENT(IN)           :: A(:), B_CHK(:)
      INTEGER (8), INTENT(IN)    :: ASIZE
      INTEGER, INTENT(OUT)       :: RES
      INTEGER                    :: I = 1
   
      DO
      IF ( A(I) /= B_CHK(I) ) THEN
         RES = -1
         EXIT
      END IF
      IF ( I == ASIZE ) EXIT
      I = I+1
      END DO
   END SUBROUTINE
   
   SUBROUTINE WRITE_DATA (B, ASIZE, RANK)
      INTEGER, INTENT(IN)           :: RANK
      INTEGER (8), INTENT(IN)       :: ASIZE
      REAL, INTENT(IN)              :: B(:)
      CHARACTER (LEN=BUFFER)        :: STR
      INTEGER                       :: RECLEN1, RECLEN2, RECLEN, I
   
      WRITE ( STR, '(a,I5.5,a)') 'chk/check-', RANK, ".tst"
   
      INQUIRE ( IOLENGTH=RECLEN1 ) ( B(I), I=1,ASIZE )
      INQUIRE ( IOLENGTH=RECLEN2 ) ASIZE
      RECLEN = RECLEN1 + RECLEN2
   
      OPEN ( 55, FILE=STR, STATUS='REPLACE', IOSTAT=IERROR, FORM='UNFORMATTED', &
         ACCESS='DIRECT', RECL=RECLEN, ACTION='WRITE' )
   
      WRITE ( 55, REC = 1 ) ASIZE
      WRITE ( 55, REC = 2 ) B
   
      CLOSE ( 55 )
   END SUBROUTINE
   
   SUBROUTINE READ_DATA ( B_CHK, ASIZE_CHK, RANK, ASIZE, RES )
      INTEGER, INTENT(IN)         :: RANK
      INTEGER (8), INTENT(IN)     :: ASIZE
      INTEGER (8)                 :: ASIZE_CHK
      REAL, INTENT(OUT)           :: B_CHK(:)
      CHARACTER (LEN=BUFFER)      :: STR
      INTEGER                     :: RECLEN1, RECLEN2, RECLEN, I
      INTEGER, INTENT(OUT)        :: RES
   
      WRITE ( STR, '(a,I5.5,a)') 'chk/check-', RANK, ".tst"
   
      INQUIRE ( IOLENGTH=RECLEN1 ) ASIZE_CHK
      INQUIRE ( IOLENGTH=RECLEN2 ) ( B_CHK(I), I=1,ASIZE )
      RECLEN = RECLEN1+RECLEN2
   
      OPEN ( 55, FILE=STR, STATUS='UNKNOWN', IOSTAT=IERROR, FORM='UNFORMATTED', &
         ACCESS='DIRECT', RECL=RECLEN, ACTION='READ' )
   
      READ ( 55, REC = 1 ) ASIZE_CHK
   
      IF ( ASIZE_CHK /= ASIZE ) THEN
         WRITE(*,'(A, I5.5, A)') "[ERROR -", RANK, "] : wrong dimension"
         RES = -1
         CLOSE ( 55 )
      ELSE 
         READ ( 55, REC = 2 ) B_CHK
         CLOSE ( 55 )
      END IF
   END SUBROUTINE

END MODULE

PROGRAM CHECK
   USE MPI
   USE FTI
   USE CHECK_FUNCTIONS

   IMPLICIT NONE

   !**** CONSTANTS
   INTEGER, PARAMETER      :: N = 10
   INTEGER, PARAMETER      :: CNTRLD_EXIT = 10
   INTEGER, PARAMETER      :: RECOVERY_FAILED = 20
   INTEGER, PARAMETER      :: DATA_CORRUPT = 30
   INTEGER, PARAMETER      :: KEEP = 2
   INTEGER, PARAMETER      :: RESTART = 1
   INTEGER, PARAMETER      :: INIT = 0

   !**** VARIABLE DEFINITIONS
   INTEGER                 :: CRASH, LEVEL, DIFF_SIZES
   INTEGER                 :: RES, TMP, PAR, STATE, SCES = 1
   INTEGER                 :: FTI_APP_RANK, INVALID
   INTEGER (8)             :: ASIZE_CHK
   INTEGER, TARGET         :: FTI_COMM_WORLD
   CHARACTER (BUFFER)      :: CF
   CHARACTER (1)           :: CRASH_CHAR, LEVEL_CHAR, DIFF_SIZES_CHAR

   !**** DYNAMIC ARRAYS
   REAL, DIMENSION(:), ALLOCATABLE            :: B_CHK


   !**** FTI PROTECTED VARIABLES
   REAL, TARGET, DIMENSION(:), ALLOCATABLE    :: A, B 
   INTEGER (8), TARGET     :: ASIZE

   !**** POINTER FOR FTI PROTECTED VARS
   REAL, POINTER           :: A_PTR(:), B_PTR(:) 
   INTEGER (8), POINTER    :: ASIZE_PTR

   !**** GET ARGUMENTS FROM COMMAND LINE 
   CALL GETARG(1, CF)
   CALL GETARG(2, CRASH_CHAR)
   CALL GETARG(3, LEVEL_CHAR)
   CALL GETARG(4, DIFF_SIZES_CHAR)
   READ (CRASH_CHAR(1:1),'(I4)') CRASH
   READ (LEVEL_CHAR(1:1),'(I4)') LEVEL
   READ (DIFF_SIZES_CHAR(1:1),'(I4)') DIFF_SIZES

   !**** MAIN PROGRAM STARTS HERE
   CALL MPI_INIT(IERROR)
   FTI_COMM_WORLD = MPI_COMM_WORLD
   CALL FTI_INIT(CF, FTI_COMM_WORLD, IERROR)

   CALL MPI_COMM_RANK(FTI_COMM_WORLD, FTI_APP_RANK, IERROR)

   ASIZE = N 

   IF ( DIFF_SIZES == 1 ) THEN

      PAR = MOD( FTI_APP_RANK, 7 )
      
      SELECT CASE (PAR) 
      
         CASE ( 0 )
            ASIZE = N;
         CASE ( 1 )
            ASIZE = 2*N;
         CASE ( 2 )
            ASIZE = 3*N;
         CASE ( 3 )
            ASIZE = 4*N;
         CASE ( 4 )
            ASIZE = 5*N;
         CASE ( 5 )
            ASIZE = 6*N;
         CASE ( 6 )
            ASIZE = 7*N;
      
      END SELECT

   END IF

   ALLOCATE( A(ASIZE), STAT=IERROR )
   ALLOCATE( B(ASIZE), STAT=IERROR )

   A_PTR => A
   B_PTR => B
   ASIZE_PTR => ASIZE

   CALL FTI_STATUS( STATE );

   CALL FTI_PROTECT(0, A_PTR, IERROR);
   CALL FTI_PROTECT(1, B_PTR, IERROR);
   CALL FTI_PROTECT(2, ASIZE_PTR, IERROR);

   IF (STATE == INIT) THEN
      CALL INIT_ARRAYS ( A, B )
      CALL WRITE_DATA (B, ASIZE, FTI_APP_RANK);
      CALL MPI_BARRIER ( FTI_COMM_WORLD, IERROR )
      CALL FTI_CHECKPOINT ( 1, LEVEL, IERROR )
      CALL SLEEP(2)
      IF ( CRASH == 1 .AND. FTI_APP_RANK == 0 ) THEN 
         CALL EXIT ( CNTRLD_EXIT )
      END IF
   END IF

   IF ( STATE == RESTART .OR. STATE == KEEP ) THEN
      CALL FTI_RECOVER ( IERROR )
      IF (IERROR /= 0) THEN
         CALL EXIT ( RECOVERY_FAILED )
      END IF
      ALLOCATE ( B_CHK(ASIZE), STAT=IERROR )
      CALL READ_DATA ( B_CHK, ASIZE_CHK, FTI_APP_RANK, ASIZE, IERROR )
      CALL MPI_BARRIER ( FTI_COMM_WORLD, IERROR )
      IF ( IERROR /= 0 ) THEN
         CALL EXIT ( DATA_CORRUPT )
      END IF
   END IF

   !**** CALL VECMULT

   A = A*B

   IF (STATE == RESTART .OR. STATE == KEEP) THEN
      CALL VALIDIFY ( A, B_CHK, ASIZE, INVALID )
      CALL MPI_Allreduce( INVALID, TMP, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD, IERROR);
      INVALID = TMP;
   END IF    

   IF ( FTI_APP_RANK == 0 .AND. (STATE == RESTART .OR. STATE == KEEP) ) THEN
      IF ( INVALID == 0 ) THEN
         PRINT *, "[SUCCESSFULL]"
      ELSE
         PRINT *, "[NOT SUCCESSFULL]"
         SCES = 0
      END IF
   END IF

   CALL MPI_BARRIER ( FTI_COMM_WORLD, IERROR );

   CALL FTI_FINALIZE(IERROR)
   CALL MPI_FINALIZE(IERROR)

   IF ( SCES == 1 ) THEN 
      CALL EXIT ( 0 )
   ELSE
      CALL EXIT ( DATA_CORRUPT )
   END IF
END PROGRAM

