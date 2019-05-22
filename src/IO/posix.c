#include "interface.h"
#include "utility.h"

int FTI_PosixOpen(char *fn, void *fileDesc){
	char str[FTI_BUFS];
	WritePosixInfo_t *fd = (WritePosixInfo_t *) fileDesc;
	if ( fd->flag == 'w' )
		fd->f = fopen(fn,"wb");
	else if ( fd -> flag == 'r')
		fd->f = fopen(fn,"rb");
	else if ( fd -> flag == 'e' )
		fd->f = fopen(fn, "r+" );
	else{
		FTI_Print("Posix Open Should always indicated flag",FTI_WARN);
	}

	snprintf(str, FTI_BUFS, "Opening File %s with flags %c", fn, fd->flag);
	FTI_Print(str,FTI_WARN);	

	if ( fd->f == NULL ){
		snprintf(str, FTI_BUFS, "unable to create file [POSIX ERROR - %d] %s", errno, strerror(errno));
		FTI_Print(str,FTI_EROR);
		return FTI_NSCS;
	}
	return FTI_SCES;
}

int FTI_PosixClose(void *fileDesc){
	WritePosixInfo_t *fd = (WritePosixInfo_t *) fileDesc;
	fclose(fd->f);
	return FTI_SCES;
}

int FTI_PosixWrite(void *src, size_t size, void *fileDesc){
	WritePosixInfo_t *fd = (WritePosixInfo_t *)fileDesc;
	size_t written = 0;
	int fwrite_errno;
	char str[FTI_BUFS];

	while (written < size && !ferror(fd->f)) {
		errno = 0;
		written += fwrite(((char *)src) + written, 1, size - written, fd->f);
		fwrite_errno = errno;
	}

	if (ferror(fd->f)){
		char error_msg[FTI_BUFS];
		error_msg[0] = 0;
		strerror_r(fwrite_errno, error_msg, FTI_BUFS);
		snprintf(str, FTI_BUFS, "Unable to write : [POSIX ERROR - %s.]", error_msg);
		FTI_Print(str, FTI_EROR);
		fclose(fd->f);
		return FTI_NSCS;
	}
	else
		return FTI_SCES;

}

int FTI_PosixSeek(size_t pos, void *fileDesc){
	WritePosixInfo_t *fd = (WritePosixInfo_t *) fileDesc;
	if ( fseek( fd->f, pos, SEEK_SET ) == -1 ) {
		char error_msg[FTI_BUFS];
		sprintf(error_msg, "Unable to Seek : [POSIX ERROR -%s.]", strerror(errno));
		FTI_Print(error_msg, FTI_EROR );
		return FTI_NSCS;
	}
	return FTI_SCES;
}

int FTI_PosixRead(void *dest, size_t size, void *fileDesc){
	return FTI_SCES;
}

int FTI_PosixSync(void *fileDesc){
	fsync(fileno(((WritePosixInfo_t *) fileDesc)->f));
	return FTI_SCES;
}


WritePosixInfo_t *FTI_InitPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data){
	char fn[FTI_BUFS];
	int level = FTI_Exec->ckptLvel;

	WritePosixInfo_t *write_info = (WritePosixInfo_t *) malloc (sizeof(WritePosixInfo_t));

	snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

	if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
		snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
	}
	else {
		snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
	}

	write_info->flag = 'w';
	write_info->offset = 0;
	FTI_PosixOpen(fn,write_info);
	return write_info;
}

int FTI_WritePosixData(FTIT_dataset * FTI_DataVar, WritePosixInfo_t *write_info){
	char str[FTI_BUFS];
	int res;

	if ( !(FTI_DataVar->isDevicePtr) ){
		FTI_Print(str,FTI_INFO);
		if (( res = FTI_Try(FTI_PosixWrite(FTI_DataVar->ptr, FTI_DataVar->size, write_info),"Storing Data to Checkpoint file")) != FTI_SCES){
			snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
			FTI_Print(str, FTI_EROR);
			FTI_PosixClose(write_info);
			return FTI_NSCS;
		}
	}
#ifdef GPUSUPPORT            
	// if data are stored to the GPU move them from device
	// memory to cpu memory and store them.
	else {
		FTI_Print(str,FTI_INFO);
		if ((res = FTI_Try(
						FTI_TransferDeviceMemToFileAsync(FTI_DataVar,  FTI_PosixWrite, write_info),
						"moving data from GPU to storage")) != FTI_SCES) {
			snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_Data[i].id);
			FTI_Print(str, FTI_EROR);
			FTI_PosixClose(write_info);
			return FTI_NSCS;
		}
	}
#endif  
	return FTI_SCES;
}
