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
	return FTI_SCES;
}

int FTI_MPIOClose(void *fileDesc){
	WriteMPIInfo_t *fd = (WriteMPIInfo_t*) fileDesc;
	MPI_Info_free(&(fd->info));
	MPI_File_close(&(fd->pfh));
	return FTI_SCES;
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

size_t FTI_GetMPIOFilePos(void *fileDesc){
	WriteMPIInfo_t *fd = (WriteMPIInfo_t *)fileDesc;
	return fd->offset;
}

int FTI_MPIORead(void *dest, size_t size, void *fileDesc){
	WriteMPIInfo_t *fd = (WriteMPIInfo_t *)fileDesc;
	return MPI_File_read_at(fd->pfh, fd->offset, dest, size, MPI_BYTE, MPI_STATUS_IGNORE);
}


void *FTI_InitMPIO(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data){
	char gfn[FTI_BUFS], ckptFile[FTI_BUFS];
	int i;

	MPI_Offset offset = 0;
	MPI_Offset chunkSize = FTI_Exec->ckptSize;
	WriteMPIInfo_t *write_info = (WriteMPIInfo_t*) malloc (sizeof(WriteMPIInfo_t));

	write_info->FTI_Conf = FTI_Conf;
	write_info->FTI_Topo= FTI_Topo;
	write_info->flag = 'w';

	FTI_Print("I/O mode: MPI-IO.", FTI_DBUG);
	snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
	snprintf(ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
	snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, ckptFile);
	FTI_MPIOOpen(gfn, write_info);


	// collect chunksizes of other ranks
	MPI_Offset* chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
	MPI_Allgather(&chunkSize, 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);

	// set file offset
	for (i = 0; i < FTI_Topo->splitRank; i++) {
		offset += chunkSizes[i];
	}
	free(chunkSizes);
	write_info->offset = offset;
	return (void *)write_info;
}

int FTI_WriteMPIOData(FTIT_dataset * FTI_DataVar, void *fd){
	WriteMPIInfo_t *write_info = (WriteMPIInfo_t *) fd;

	char str[FTI_BUFS];
	int res;
	if ( !(FTI_DataVar->isDevicePtr) ){
		FTI_Print(str,FTI_INFO);
		if (( res = FTI_MPIOWrite(FTI_DataVar->ptr, FTI_DataVar->size, write_info), "Storing Data to checkpoint file")!=FTI_SCES){
			snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
			FTI_Print(str, FTI_EROR);
			FTI_MPIOClose(write_info);
			return res;
		}
	}
#ifdef GPUSUPPORT
	// dowload data from the GPU if necessary
	// Data are stored in the GPU side.
	else {
		snprintf(str, FTI_BUFS, "Dataset #%d Writing GPU Data.", FTI_DataVar->id);
		FTI_Print(str,FTI_INFO);
		if ((res = FTI_Try(
						FTI_TransferDeviceMemToFileAsync(FTI_DataVar, FTI_MPIOWrite, write_info),
						"moving data from GPU to storage")) != FTI_SCES) {
			snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_Data[i].id);
			FTI_Print(str, FTI_EROR);
			FTI_MPIOClose(write_info);
			return res;
		}
	}
#endif
	write_info->offset += FTI_DataVar->size;
	return FTI_SCES;
}
