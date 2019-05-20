#include "interface.h"
#include "utility.h"

int FTI_MPIOOpen(char *fn, void *fileDesc){
	WriteMPIInfo_t *fd = (WriteMPIInfo_t*) fileDesc;
	int res;
	char str[FTI_BUFS], mpi_err[FTI_BUFS];
	MPI_Info_create(&(fd->info));

	if ( fd->flag == 'r' )
		MPI_Info_set(fd->info, "romio_cb_read", "enable");
	else if (fd->flag == 'w' )		
		MPI_Info_set(fd->info, "romio_cb_write", "enable");

	MPI_Info_set(fd->info, "stripping_unit", "4194304");
#ifdef LUSTRE
	if (fd->FTI_Topo->splitRank == 0) {
		res = llapi_file_create(gfn, fd->FTI_Conf->stripeUnit, fd->FTI_Conf->stripeOffset, fd->FTI_Conf->stripeFactor, 0);
		if (res) {
			char error_msg[FTI_BUFS];
			error_msg[0] = 0;
			strerror_r(-res, error_msg, FTI_BUFS);
			snprintf(str, FTI_BUFS, "[Lustre] %s.", error_msg);
			FTI_Print(str, FTI_WARN);
		} else {
			snprintf(str, FTI_BUFS, "[LUSTRE] file:%s striping_unit:%i striping_factor:%i striping_offset:%i",
					ckptFile, FTI_Conf->stripeUnit, FTI_Conf->stripeFactor, FTI_Conf->stripeOffset);
			FTI_Print(str, FTI_DBUG);
		}
	}
#endif

	if ( fd->flag == 'r' )
		res = MPI_File_open(FTI_COMM_WORLD, fn, MPI_MODE_RDWR, fd->info, &(fd->pfh));
	else if (fd->flag == 'w' )		
		res = MPI_File_open(FTI_COMM_WORLD, fn, MPI_MODE_WRONLY|MPI_MODE_CREATE, fd->info, &(fd->pfh));

	if (res != 0) {
		errno = 0;
		int reslen;
		MPI_Error_string(res, mpi_err, &reslen);
		snprintf(str, FTI_BUFS, "unable to create file [MPI ERROR - %i] %s", res, mpi_err);
		FTI_Print(str, FTI_EROR);
		return FTI_NSCS;
	}
}

int FTI_MPIOClose(void *fileDesc){
	WriteMPIInfo_t *fd = (WriteMPIInfo_t*) fileDesc;
	MPI_Info_free(&(fd->info));
	MPI_File_close(&(fd->pfh));
}

int FTI_MPIOWrite(void *src, size_t size, void *fileDesc)
{
	WriteMPIInfo_t *fd= (WriteMPIInfo_t *)fileDesc;
	size_t pos = 0;
	size_t bSize = fd->FTI_Conf->transferSize;
	while (pos < size) {
		if ((size - pos) < fd->FTI_Conf->transferSize) {
			bSize = size - pos;
		}

		MPI_Datatype dType;
		MPI_Type_contiguous(bSize, MPI_BYTE, &dType);
		MPI_Type_commit(&dType);

		fd->err = MPI_File_write_at(fd->pfh, fd->offset, src, 1, dType, MPI_STATUS_IGNORE);
		// check if successful
		if (fd->err != 0) {
			errno = 0;
			return FTI_NSCS;
		}
		MPI_Type_free(&dType);
		src += bSize;
		fd->offset += bSize;
		pos = pos + bSize;
	}
	return FTI_SCES;
}


int FTI_MPIORead(void *dest, size_t size, void *fileDesc){
	WriteMPIInfo_t *fd = (WriteMPIInfo_t *)fileDesc;
	return MPI_File_read_at(fd->pfh, fd->offset, dest, size, MPI_BYTE, MPI_STATUS_IGNORE);
}

