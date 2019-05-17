/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  FTI - A multi-level checkpointing library for C/C++/Fortran applications
 *
 *  Revision 1.0 : Fault Tolerance Interface (FTI)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its contributors
 *  may be used to endorse or promote products derived from this software without
 *  specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  @file   postckpt.c
 *  @date   October, 2017
 *  @brief  Post-checkpointing functions for the FTI library.
 */

#include "interface.h"
#include "macros.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      It returns FTI_SCES.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES.

  This function just returns FTI_SCES to have homogeneous code.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Local(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
		FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
	FTI_Print("Starting checkpoint post-processing L1", FTI_DBUG);
	return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It sends Ckpt file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      destination     destination group rank
  @param      postFlag        0 if postckpt done by approc, > 0 if by head
  @return     integer         FTI_SCES if successful.

  This function sends ckpt file to partner process. Partner should call
  FTI_RecvPtner to receive this file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SendCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
		int destination, int postFlag)
{
	char lfn[FTI_BUFS], str[FTI_BUFS];
	snprintf(lfn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[postFlag * FTI_BUFS]);

	//PostFlag is set to 0 if Post-processing is inline and set to processes nodeID if Post-processing done by head
	if (postFlag) {
		snprintf(str, FTI_BUFS, "L2 trying to access process's %d ckpt. file (%s).", postFlag, lfn);
	}
	else {
		snprintf(str, FTI_BUFS, "L2 trying to access local ckpt. file (%s).", lfn);
	}
	FTI_Print(str, FTI_DBUG);

	FILE* lfd = fopen(lfn, "rb");
	if (lfd == NULL) {
		FTI_Print("FTI failed to open L2 Ckpt. file.", FTI_DBUG);
		return FTI_NSCS;
	}

	char* buffer = talloc(char, FTI_Conf->blockSize);
	long toSend = FTI_Exec->meta[0].fs[postFlag]; //remaining data to send
	while (toSend > 0) {
		int sendSize = (toSend > FTI_Conf->blockSize) ? FTI_Conf->blockSize : toSend;
		int bytes;
		FREAD(bytes,buffer, sizeof(char), sendSize, lfd,"p",buffer);
		MPI_Send(buffer, bytes, MPI_CHAR, destination, FTI_Conf->generalTag, FTI_Exec->groupComm);
		toSend -= bytes;
	}

	free(buffer);
	fclose(lfd);

	return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It receives Ptner file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      source          souce group rank
  @param      postFlag        0 if postckpt done by approc, > 0 if by head
  @return     integer         FTI_SCES if successful.

  This function receives ckpt file from partner process and saves it as
  Ptner file. Partner should call FTI_SendCkpt to send file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecvPtner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
		int source, int postFlag)
{
	//heads need to use ckptFile to get ckptID and rank
	int ckptID, rank;
	sscanf(&FTI_Exec->meta[0].ckptFile[postFlag * FTI_BUFS], "Ckpt%d-Rank%d.fti", &ckptID, &rank);

	char pfn[FTI_BUFS], str[FTI_BUFS];
	snprintf(pfn, FTI_BUFS, "%s/Ckpt%d-Pcof%d.fti", FTI_Conf->lTmpDir, ckptID, rank);
	snprintf(str, FTI_BUFS, "L2 trying to access Ptner file (%s).", pfn);
	FTI_Print(str, FTI_DBUG);

	FILE* pfd = fopen(pfn, "wb");
	if (pfd == NULL) {
		FTI_Print("FTI failed to open L2 ptner file.", FTI_DBUG);
		return FTI_NSCS;
	}

	char* buffer = talloc(char, FTI_Conf->blockSize);
	unsigned long toRecv = FTI_Exec->meta[0].pfs[postFlag]; //remaining data to receive
	while (toRecv > 0) {
		int recvSize = (toRecv > FTI_Conf->blockSize) ? FTI_Conf->blockSize : toRecv;
		MPI_Recv(buffer, recvSize, MPI_CHAR, source, FTI_Conf->generalTag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
		size_t wbytes;
		FWRITE(wbytes,buffer, sizeof(char), recvSize, pfd,"p",buffer);
		toRecv -= recvSize;
	}

	free(buffer);
	fclose(pfd);

	return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It copies ckpt. files in to the partner node.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function copies the checkpoint files into the partner node. It
  follows a ring, where the ring size is the group size given in the FTI
  configuration file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Ptner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
		FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
	FTI_Print("Starting checkpoint post-processing L2", FTI_DBUG);
	if (FTI_Topo->amIaHead) {
		int res = FTI_Try(FTI_LoadTmpMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load temporary metadata.");
		if (res != FTI_SCES) {
			return FTI_NSCS;
		}
	}
	int startProc, endProc;
	if (FTI_Topo->amIaHead) { //post-processing for every process in the node
		startProc = 1;
		endProc = FTI_Topo->nodeSize;
	}
	else { //post-processing only for itself
		startProc = 0;
		endProc = 1;
	}

	int source = FTI_Topo->left; //receive Ckpt file from this process
	int destination = FTI_Topo->right; //send Ckpt file to this process
	int i;
	for (i = startProc; i < endProc; i++) {
		if (FTI_Topo->groupRank % 2) { //first send, then receive
			int res = FTI_SendCkpt(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, i);
			if (res != FTI_SCES) {
				return FTI_NSCS;
			}
			res = FTI_RecvPtner(FTI_Conf, FTI_Exec, FTI_Ckpt, source, i);
			if (res != FTI_SCES) {
				return FTI_NSCS;
			}
		} else { //first receive, then send
			int res = FTI_RecvPtner(FTI_Conf, FTI_Exec, FTI_Ckpt, source, i);
			if (res != FTI_SCES) {
				return FTI_NSCS;
			}
			res = FTI_SendCkpt(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, i);
			if (res != FTI_SCES) {
				return FTI_NSCS;
			}
		}
	}
	return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It performs RS encoding with the ckpt. files in to the group.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function performs the Reed-Solomon encoding for a given group. The
  checkpoint files are padded to the maximum size of the largest checkpoint
  file in the group +- the extra space to be a multiple of block size.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RSenc(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
		FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
	FTI_Print("Starting checkpoint post-processing L3", FTI_DBUG);
	if (FTI_Topo->amIaHead) {
		int res = FTI_Try(FTI_LoadTmpMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load temporary metadata.");
		if (res != FTI_SCES) {
			return FTI_NSCS;
		}
	}
	int startProc, endProc;
	if (FTI_Topo->amIaHead) {
		startProc = 1;
		endProc = FTI_Topo->nodeSize;
	}
	else {
		startProc = 0;
		endProc = 1;
	}

	int proc;
	for (proc = startProc; proc < endProc; proc++) {
		int ckptID, rank;
		sscanf(&FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS], "Ckpt%d-Rank%d.fti", &ckptID, &rank);
		char lfn[FTI_BUFS], efn[FTI_BUFS];

		snprintf(lfn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
		snprintf(efn, FTI_BUFS, "%s/Ckpt%d-RSed%d.fti", FTI_Conf->lTmpDir, ckptID, rank);

		char str[FTI_BUFS];
		snprintf(str, FTI_BUFS, "L3 trying to access local ckpt. file (%s).", lfn);
		FTI_Print(str, FTI_DBUG);

		//all files in group must have the same size
		long maxFs = FTI_Exec->meta[0].maxFs[proc]; //max file size in group

		// determine file size in order to write at the end of the elongated file
		// (i.e. write at the end of file after 'truncate(..., maxFs)'.
		struct stat st_;
		if( FTI_Conf->ioMode == FTI_IO_FTIFF ) {
			stat( lfn, &st_ );
		}

		if (truncate(lfn, maxFs) == -1) {
			FTI_Print("Error with truncate on checkpoint file", FTI_WARN);
			return FTI_NSCS;
		}

		// write file size at the end of elongated file to recover original size
		// during restart. The file size, thus,  will be included in the encoded data
		// and will be available at recovery before the re truncation to the original
		// file size. [Depends on the correct value assigned to maxFs inside
		// 'FTIFF_CreateMetadata'. The value has to be the maximum file size of the 
		// group PLUS 'sizeof(off_t)']
		if( FTI_Conf->ioMode == FTI_IO_FTIFF ) {
			int lftmp_ = open( lfn, O_WRONLY );
			if( lftmp_ == -1 ) {
				FTI_Print("FTI_RSenc: (FTIFF) Unable to open file!", FTI_EROR);
				return FTI_NSCS;
			} 
			if( lseek( lftmp_, -sizeof(off_t), SEEK_END ) == -1 ) {
				FTI_Print("FTI_RSenc: (FTIFF) Unable to seek in file!", FTI_EROR);
				return FTI_NSCS;
			}
			if( write( lftmp_, &st_.st_size, sizeof(off_t) ) == -1 ) {
				FTI_Print("FTI_RSenc: (FTIFF) Unable to write meta data in file!", FTI_EROR);
				return FTI_NSCS;
			}
			close( lftmp_ );
		}

		FILE* lfd = fopen(lfn, "rb");
		if (lfd == NULL) {
			FTI_Print("FTI failed to open L3 checkpoint file.", FTI_EROR);
			return FTI_NSCS;
		}

		FILE* efd = fopen(efn, "wb");
		if (efd == NULL) {
			FTI_Print("FTI failed to open encoded ckpt. file.", FTI_EROR);

			fclose(lfd);

			return FTI_NSCS;
		}

		int bs = FTI_Conf->blockSize;
		char* myData = talloc(char, bs);
		char* coding = talloc(char, bs);
		char* data = talloc(char, 2 * bs);
		int* matrix = talloc(int, FTI_Topo->groupSize* FTI_Topo->groupSize);

		int i;
		for (i = 0; i < FTI_Topo->groupSize; i++) {
			int j;
			for (j = 0; j < FTI_Topo->groupSize; j++) {
				matrix[i * FTI_Topo->groupSize + j] = galois_single_divide(1, i ^ (FTI_Topo->groupSize + j), FTI_Conf->l3WordSize);
			}
		}



		int remBsize = bs;
		long ps = ((maxFs / bs)) * bs;
		if (ps < maxFs) {
			ps = ps + bs;
		}

		//for MD5 checksum
		MD5_CTX mdContext;
		MD5_Init (&mdContext);

		// For each block
		long pos = 0;
		while (pos < ps) {
			if ((maxFs - pos) < bs) {
				remBsize = maxFs - pos;
			}

			// Reading checkpoint files
			bzero(coding, bs);
			bzero(myData, bs);
			bzero(data, 2*bs);
			size_t bytes;
			FREAD(bytes,myData, sizeof(char), remBsize, lfd,"ppppf",data,matrix,coding,myData,efd);
			int dest = FTI_Topo->groupRank;
			i = FTI_Topo->groupRank;
			int offset = 0;
			int init = 0;
			int cnt = 0;

			// For each encoding
			MPI_Request reqSend, reqRecv; //used between iterations in while loop
			while (cnt < FTI_Topo->groupSize) {
				if (cnt == 0) {
					memcpy(&(data[offset * bs]), myData, sizeof(char) * bytes);
				}
				else {
					MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
					MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);
				}

				// At every loop *but* the last one we send the data
				if (cnt != FTI_Topo->groupSize - 1) {
					dest = (dest + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
					int src = (i + 1) % FTI_Topo->groupSize;
					MPI_Isend(myData, bytes, MPI_CHAR, dest, FTI_Conf->generalTag, FTI_Exec->groupComm, &reqSend);
					MPI_Irecv(&(data[(1 - offset) * bs]), bs, MPI_CHAR, src, FTI_Conf->generalTag, FTI_Exec->groupComm, &reqRecv);
				}

				int matVal = matrix[FTI_Topo->groupRank * FTI_Topo->groupSize + i];
				// First copy or xor any data that does not need to be multiplied by a factor
				if (matVal == 1) {
					if (init == 0) {
						memcpy(coding, &(data[offset * bs]), bs);
						init = 1;
					}
					else {
						galois_region_xor(&(data[offset * bs]), coding, bs);
					}
				}

				// Then the data that needs to be multiplied by a factor
				if (matVal != 0 && matVal != 1) {
					galois_w16_region_multiply(&(data[offset * bs]), matVal, bs, coding, init);
					init = 1;
				}

				i = (i + 1) % FTI_Topo->groupSize;
				offset = 1 - offset;
				cnt++;
			}

			// Writting encoded checkpoints
			size_t wBytes = 0;
			FWRITE(wBytes,coding, sizeof(char), remBsize, efd,"",NULL);
			MD5_Update (&mdContext, coding, remBsize);

			// Next block
			pos = pos + bs;
		}

		// create checksum hex-string
		unsigned char hash[MD5_DIGEST_LENGTH];
		MD5_Final (hash, &mdContext);

		char checksum[MD5_DIGEST_STRING_LENGTH];
		int ii = 0;
		for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
			sprintf(&checksum[ii], "%02x", hash[i]);
			ii+=2;
		}

		// FTI-FF append meta data to RS file
		if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) {

			FTIFF_metaInfo *FTIFFMeta = malloc( sizeof( FTIFF_metaInfo) );

			// get timestamp
			struct timespec ntime;
			clock_gettime(CLOCK_REALTIME, &ntime);
			FTIFFMeta->timestamp = ntime.tv_sec*1000000000 + ntime.tv_nsec;

			FTIFFMeta->fs = maxFs;
			// although not needed, we have to assign value for unique hash.
			FTIFFMeta->ptFs = -1;
			FTIFFMeta->maxFs = maxFs;
			FTIFFMeta->ckptSize = FTI_Exec->meta[0].fs[proc];
			strncpy(FTIFFMeta->checksum, checksum, MD5_DIGEST_STRING_LENGTH);

			// get hash of meta data
			FTIFF_GetHashMetaInfo( FTIFFMeta->myHash, FTIFFMeta );

			// serialize data block variable meta data and append to encoded file
			char* buffer_ser = (char*) malloc ( FTI_filemetastructsize );
			if( buffer_ser == NULL ) {
				snprintf( str, FTI_BUFS, "FTI_RSenc - failed to allocate %d bytes for 'buffer_ser'", FTI_dbvarstructsize );
				FTI_Print(str, FTI_EROR);
				free(data);
				free(matrix);
				free(coding);
				free(myData);
				fclose(lfd);
				fclose(efd);
				errno = 0;
				return FTI_NSCS;
			}
			if( FTIFF_SerializeFileMeta( FTIFFMeta, buffer_ser ) != FTI_SCES ) {
				FTI_Print("FTI_RSenc - failed to serialize 'currentdbvar'", FTI_EROR);
				free(buffer_ser);
				free(data);
				free(matrix);
				free(coding);
				free(myData);
				fclose(lfd);
				fclose(efd);
				errno = 0;
				return FTI_NSCS;
			}
			size_t wBytes = 0;
			FWRITE(wBytes,buffer_ser, FTI_filemetastructsize, 1, efd,"ppppf",data,matrix,coding,myData,lfd);
			free( buffer_ser );

		}

		free(data);
		free(matrix);
		free(coding);
		free(myData);
		fclose(lfd);
		fclose(efd);

		long fs = FTI_Exec->meta[0].fs[proc]; //ckpt file size

		if (truncate(lfn, fs) == -1) {
			FTI_Print("Error with re-truncate on checkpoint file", FTI_WARN);
			return FTI_NSCS;
		}

		int res = FTI_WriteRSedChecksum(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, rank, checksum);
		if (res != FTI_SCES) {
			return FTI_NSCS;
		}
	}

	return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      It flushes the local ckpt. files in to the PFS.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      level           The level from which ckpt. files are flushed.
  @return     integer         FTI_SCES if successful.

  This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Flush(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
		FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
	if (!FTI_Topo->amIaHead && level == 0) {
		return FTI_SCES; //inline L4 saves directly to PFS (nothing to flush)
	}

	/**
	 *  FTI_Flush is either executed by application processes during
	 *  FTI_Finalize or by the heads during FTI_PostCkpt.
	 **/

	char str[FTI_BUFS];
	snprintf(str, FTI_BUFS, "Starting checkpoint post-processing L4 for level %d", level);
	FTI_Print(str, FTI_DBUG);

	if ( !(FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp) ) {
		FTI_Print("Saving to temporary global directory", FTI_DBUG);

		//Create global temp directory
		if (mkdir(FTI_Conf->gTmpDir, 0777) == -1) {
			if (errno != EEXIST) {
				FTI_Print("Cannot create global directory", FTI_EROR);
				return FTI_NSCS;
			}
		}
	} else {
		if ( !FTI_Ckpt[4].hasDcp ) {
			if (mkdir(FTI_Ckpt[4].dcpDir, 0777) == -1) {
				if (errno != EEXIST) {
					FTI_Print("Cannot create global dCP directory", FTI_EROR);
					return FTI_NSCS;
				}
			}
		}
	}
	int res = FTI_Try(FTI_LoadMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load metadata.");
	if (res != FTI_SCES) {
		return FTI_NSCS;
	}

	switch(FTI_Conf->ioMode) {
		case FTI_IO_FTIFF:
		case FTI_IO_HDF5:
		case FTI_IO_POSIX:
			FTI_FlushPosix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
			break;
		case FTI_IO_MPI:
			FTI_FlushMPI(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
			break;
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
		case FTI_IO_SIONLIB:
			FTI_FlushSionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
			break;
#endif
	}
	//}
return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It moves the level 4 ckpt. to the archive folder.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function is called if keepL4Ckpt is enabled in the configuration file.
  It moves the old level 4 ckpt file to the archive folder before the l4 
  folder in the global directory is deleted. 

 **/
/*-------------------------------------------------------------------------*/
int FTI_ArchiveL4Ckpt( FTIT_configuration* FTI_Conf, FTIT_execution *FTI_Exec, FTIT_checkpoint *FTI_Ckpt,
		FTIT_topology *FTI_Topo ) 
{
	char strerr[FTI_BUFS];
	char fn_from[FTI_BUFS];
	char fn_to[FTI_BUFS];
	struct stat st;
	errno = 0;
	stat( FTI_Ckpt[4].archDir, &st );
	switch ( errno ) {
		case 0:
			if ( !(S_ISDIR( st.st_mode )) ) {
				snprintf(strerr, FTI_BUFS, "'%s' is not a directory, cannot keep L4 checkpoint.", FTI_Ckpt[4].archDir);
				FTI_Print(strerr, FTI_WARN);
				errno = 0;
				return FTI_NSCS;
			} 
			break;
		case ENOENT:
			snprintf(strerr, FTI_BUFS, "directory '%s' does not exist, cannot keep L4 checkpoint.", FTI_Ckpt[4].archDir);
			FTI_Print(strerr, FTI_WARN);
			errno = 0;
			return FTI_NSCS;
		default:
			snprintf(strerr, FTI_BUFS, "error with stats on '%s', cannot keep L4 checkpoint.", FTI_Ckpt[4].archDir);
			FTI_Print(strerr, FTI_EROR);
			errno = 0;
			return FTI_NSCS;
	}
	if ( (FTI_Conf->ioMode == FTI_IO_POSIX) || (FTI_Conf->ioMode == FTI_IO_FTIFF) || (FTI_Conf->ioMode == FTI_IO_HDF5) ) {
		if ( (FTI_Topo->nbHeads == 0) || (FTI_Ckpt[4].isInline && (FTI_Topo->nbHeads > 0)) ) {
			snprintf(fn_from, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->meta[0].currentL4CkptFile ); 
			snprintf(fn_to, FTI_BUFS, "%s/%s", FTI_Ckpt[4].archDir, FTI_Exec->meta[0].currentL4CkptFile ); 
			if ( rename(fn_from,fn_to) != 0 ) {
				snprintf(strerr, FTI_BUFS, "could not move '%s' to '%s', cannot keep L4 checkpoint.", fn_from, fn_to);
				FTI_Print( strerr, FTI_EROR );
				errno = 0;
				return FTI_NSCS;
			}
		} else {
			int i;
			for ( i=1; i<FTI_Topo->nodeSize; ++i ) {
				snprintf(fn_from, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir, &FTI_Exec->meta[0].currentL4CkptFile[i * FTI_BUFS] ); 
				snprintf(fn_to, FTI_BUFS, "%s/%s", FTI_Ckpt[4].archDir, &FTI_Exec->meta[0].currentL4CkptFile[i * FTI_BUFS] ); 
				if ( rename(fn_from,fn_to) != 0 ) {
					snprintf(strerr, FTI_BUFS, "could not move '%s' to '%s', cannot keep L4 checkpoint.", fn_from, fn_to);
					FTI_Print( strerr, FTI_EROR );
					errno = 0;
					return FTI_NSCS;
				}
			}
		}
	} else {
		if ( FTI_Topo->splitRank == 0 ) {
			snprintf(fn_from, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].currentL4CkptFile ); 
			snprintf(fn_to, FTI_BUFS, "%s/%s", FTI_Ckpt[4].archDir, FTI_Exec->meta[FTI_Exec->ckptLvel].currentL4CkptFile ); 
			if ( rename(fn_from,fn_to) != 0 ) {
				snprintf(strerr, FTI_BUFS, "could not move '%s' to '%s', cannot keep L4 checkpoint.", fn_from, fn_to);
				FTI_Print( strerr, FTI_EROR );
				errno = 0;
				return FTI_NSCS;
			}
		}
	}

	// needed to avoid that the files get deleted before we can move them
	MPI_Barrier(FTI_COMM_WORLD);

	return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      It flushes the local ckpt. files in to the PFS using POSIX.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      level           The level from which ckpt. files are flushed.
  @return     integer         FTI_SCES if successful.

  This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FlushPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
		FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
	FTI_Print("Starting checkpoint post-processing L4 using Posix IO.", FTI_DBUG);
	int startProc, endProc, proc;
	if (FTI_Topo->amIaHead) {
		startProc = 1;
		endProc = FTI_Topo->nodeSize;
	}
	else {
		startProc = 0;
		endProc = 1;
	}

	for (proc = startProc; proc < endProc; proc++) {
		char str[FTI_BUFS];
		snprintf(str, FTI_BUFS, "Post-processing for proc %d started.", proc);
		FTI_Print(str, FTI_DBUG);
		char lfn[FTI_BUFS], gfn[FTI_BUFS];
		if ( FTI_Ckpt[4].isDcp ) {
			snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
		} else {
			snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
		}
		snprintf(str, FTI_BUFS, "Global temporary file name for proc %d: %s", proc, gfn);
		FTI_Print(str, FTI_DBUG);
		FILE* gfd = fopen(gfn, "wb");

		if (gfd == NULL) {
			FTI_Print("L4 cannot open ckpt. file in the PFS.", FTI_EROR);
			return FTI_NSCS;
		}

		if (level == 0) {
			if ( FTI_Ckpt[4].isDcp ) {
				snprintf(lfn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
			} else {
				snprintf(lfn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
			}
		}
		else {
			snprintf(lfn, FTI_BUFS, "%s/%s", FTI_Ckpt[level].dir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
		}
		snprintf(str, FTI_BUFS, "Local file name for proc %d: %s", proc, lfn);
		FTI_Print(str, FTI_DBUG);
		// Open local file
		FILE* lfd = fopen(lfn, "rb");
		if (lfd == NULL) {
			FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
			fclose(gfd);
			return FTI_NSCS;
		}

		char *readData = talloc(char, FTI_Conf->transferSize);
		long bSize = FTI_Conf->transferSize;
		long fs = FTI_Exec->meta[level].fs[proc];
		snprintf(str, FTI_BUFS, "Local file size for proc %d: %ld", proc, fs);
		FTI_Print(str, FTI_DBUG);
		long pos = 0;
		// Checkpoint files exchange
		while (pos < fs) {
			if ((fs - pos) < FTI_Conf->transferSize)
				bSize = fs - pos;

			size_t bytes;
			FREAD(bytes,readData, sizeof(char), bSize, lfd,"pf",readData,gfd);
			size_t wBytes = 0;	
			FWRITE(wBytes,readData, sizeof(char), bytes, gfd,"pf",readData,lfd);
			pos = pos + bytes;
		}
		free(readData);
		fclose(lfd);
		fclose(gfd);
	}
	return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It flushes the local ckpt. files in to the PFS using MPI-I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      level           The level from which ckpt. files are flushed.
  @return     integer         FTI_SCES if successful.

  This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FlushMPI(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
		FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
	int res;
	FTI_Print("Starting checkpoint post-processing L4 using MPI-IO.", FTI_DBUG);
	// enable collective buffer optimization
	MPI_Info info;
	MPI_Info_create(&info);
	MPI_Info_set(info, "romio_cb_write", "enable");
	// TODO enable to set stripping unit in the config file (Maybe also other hints)
	// set stripping unit to 4MB
	MPI_Info_set(info, "stripping_unit", "4194304");

	// open parallel file (collective call)
	MPI_File pfh; // MPI-IO file handle
	char gfn[FTI_BUFS], str[FTI_BUFS], ckptFile[FTI_BUFS];
	snprintf(ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
	snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, ckptFile);
#ifdef LUSTRE
	if (FTI_Topo->splitRank == 0) {
		res = llapi_file_create(gfn, FTI_Conf->stripeUnit, FTI_Conf->stripeOffset, FTI_Conf->stripeFactor, 0);
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
	res = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &pfh);
	if (res != 0) {
		errno = 0;
		char mpi_err[FTI_BUFS];
		MPI_Error_string(res, mpi_err, NULL);
		snprintf(str, FTI_BUFS, "Unable to create file during MPI-IO flush [MPI ERROR - %i] %s", res, mpi_err);
		FTI_Print(str, FTI_EROR);
		MPI_Info_free(&info);
		return FTI_NSCS;
	}
	MPI_Info_free(&info);

	int proc, startProc, endProc;
	if (FTI_Topo->amIaHead) {
		startProc = 1;
		endProc = FTI_Topo->nodeSize;
	}
	else {
		startProc = 0;
		endProc = 1;
	}
	int nbProc = endProc - startProc;
	MPI_Offset* localFileSizes = talloc(MPI_Offset, nbProc);
	char* localFileNames = talloc(char, FTI_BUFS * endProc);
	int* splitRanks = talloc(int, endProc); //rank of process in FTI_COMM_WORLD
	for (proc = startProc; proc < endProc; proc++) {
		if (level == 0) {
			snprintf(&localFileNames[proc * FTI_BUFS], FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
		}
		else {
			snprintf(&localFileNames[proc * FTI_BUFS], FTI_BUFS, "%s/%s", FTI_Ckpt[level].dir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
		}
		if (FTI_Topo->amIaHead) {
			splitRanks[proc] = (FTI_Topo->nodeSize - 1) * FTI_Topo->nodeID + proc - 1; //determine process splitRank if head
		}
		else {
			splitRanks[proc] = FTI_Topo->splitRank;
		}
		localFileSizes[proc - startProc] = FTI_Exec->meta[level].fs[proc]; //[proc - startProc] to get index from 0
	}

	MPI_Offset* allFileSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
	MPI_Allgather(localFileSizes, nbProc, MPI_OFFSET, allFileSizes, nbProc, MPI_OFFSET, FTI_COMM_WORLD);
	free(localFileSizes);


	for (proc = startProc; proc < endProc; proc++) {
		MPI_Offset offset = 0;
		int i;
		for (i = 0; i < splitRanks[proc]; i++) {
			offset += allFileSizes[i];
		}

		FILE* lfd = fopen(&localFileNames[FTI_BUFS * proc], "rb");
		if (lfd == NULL) {
			FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
			free(localFileNames);
			free(allFileSizes);
			free(splitRanks);
			return FTI_NSCS;
		}

		char* readData = talloc(char, FTI_Conf->transferSize);
		long bSize = FTI_Conf->transferSize;
		long fs = FTI_Exec->meta[level].fs[proc];

		long pos = 0;
		// Checkpoint files exchange
		while (pos < fs) {
			if ((fs - pos) < FTI_Conf->transferSize) {
				bSize = fs - pos;
			}

			size_t bytes;
#warning I also need to close pfh file but this marcro is not yet ready
			//                MPI_File_close(&pfh);
			FREAD(bytes,readData, sizeof(char), bSize, lfd,"pppp",localFileNames,allFileSizes,splitRanks,readData);

			MPI_Datatype dType;
			MPI_Type_contiguous(bytes, MPI_BYTE, &dType);
			MPI_Type_commit(&dType);


			res = MPI_File_write_at(pfh, offset, readData, 1, dType, MPI_STATUS_IGNORE);
			// check if successful
			if (res != 0) {
				errno = 0;
				char mpi_err[FTI_BUFS];
				MPI_Error_string(res, mpi_err, NULL);
				snprintf(str, FTI_BUFS, "Failed to write data to PFS during MPIIO Flush [MPI ERROR - %i] %s", res, mpi_err);
				FTI_Print(str, FTI_EROR);
				free(localFileNames);
				free(splitRanks);
				free(allFileSizes);
				fclose(lfd);
				MPI_File_close(&pfh);
				return FTI_NSCS;
			}
			MPI_Type_free(&dType);
			offset += bytes;
			pos = pos + bytes;
		}
		free(readData);
		fclose(lfd);
	}
	free(localFileNames);
	free(allFileSizes);
	free(splitRanks);
	MPI_File_close(&pfh);
	return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It flushes the local ckpt. files in to the PFS using SIONlib.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      level           The level from which ckpt. files are flushed.
  @return     integer         FTI_SCES if successful.

  This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_FlushSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
		FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
	int proc, startProc, endProc;
	if (FTI_Topo->amIaHead) {
		startProc = 1;
		endProc = FTI_Topo->nodeSize;
	}
	else {
		startProc = 0;
		endProc = 1;
	}
	int nbProc = endProc - startProc;

	long* localFileSizes = talloc(long, nbProc);
	char* localFileNames = talloc(char, FTI_BUFS * nbProc);
	int* splitRanks = talloc(int, nbProc); //rank of process in FTI_COMM_WORLD
	for (proc = startProc; proc < endProc; proc++) {
		// Open local file case 0:
		if (level == 0) {
			snprintf(&localFileNames[(proc-startProc) * FTI_BUFS], FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
		}
		else {
			snprintf(&localFileNames[(proc-startProc) * FTI_BUFS], FTI_BUFS, "%s/%s", FTI_Ckpt[level].dir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
		}
		if (FTI_Topo->amIaHead) {
			splitRanks[proc - startProc] = (FTI_Topo->nodeSize - 1) * FTI_Topo->nodeID + proc - 1; //[proc - startProc] to get index from 0
		}
		else {
			splitRanks[proc - startProc] = FTI_Topo->splitRank; //[proc - startProc] to get index from 0
		}
		localFileSizes[proc - startProc] = FTI_Exec->meta[level].fs[proc]; //[proc - startProc] to get index from 0
	}

	int rank, ckptID;
	//  sscanf(&FTI_Exec->meta[level].ckptFile[0], "Ckpt%d-Rank%d.fti", &ckptID, &rank);
	snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
	//  snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", ckptID);
	snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, str);


	int numFiles = 1;
	int nlocaltasks = nbProc;
	int* file_map = calloc(nbProc, sizeof(int));
	int* ranks = talloc(int, nbProc);
	int* rank_map = talloc(int, nbProc);
	sion_int64* chunkSizes = talloc(sion_int64, nbProc);
	int fsblksize = -1;
	int i;
	for (i = 0; i < nbProc; i++) {
		chunkSizes[i] = localFileSizes[i];
		ranks[i] = splitRanks[i];
		rank_map[i] = splitRanks[i];
	}
	int sid = sion_paropen_mapped_mpi(fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL);
	if (sid == -1) {
		FTI_Print("Cannot open with sion_paropen_mapped_mpi.", FTI_EROR);

		free(file_map);
		free(ranks);
		free(rank_map);
		free(chunkSizes);

		return FTI_NSCS;
	}

	for (proc = startProc; proc < endProc; proc++) {
		FILE* lfd = fopen(&localFileNames[FTI_BUFS * proc], "rb");
		if (lfd == NULL) {
			FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
			free(localFileNames);
			free(splitRanks);
			sion_parclose_mapped_mpi(sid);
			free(file_map);
			free(ranks);
			free(rank_map);
			free(chunkSizes);
			return FTI_NSCS;
		}


		int res = sion_seek(sid, splitRanks[proc - startProc], SION_CURRENT_BLK, SION_CURRENT_POS);
		if (res != SION_SUCCESS) {
			errno = 0;
			snprintf(str, FTI_BUFS, "SIONlib: unable to set file pointer");
			FTI_Print(str, FTI_EROR);
			free(localFileNames);
			free(splitRanks);
			fclose(lfd);
			sion_parclose_mapped_mpi(sid);
			free(file_map);
			free(ranks);
			free(rank_map);
			free(chunkSizes);
			return FTI_NSCS;
		}

		char *readData = talloc(char, FTI_Conf->transferSize);
		long bSize = FTI_Conf->transferSize;
		long fs = FTI_Exec->meta[level].fs[proc];

		long pos = 0;
		// Checkpoint files exchange
		while (pos < fs) {
			if ((fs - pos) < FTI_Conf->transferSize)
				bSize = fs - pos;

			size_t bytes;
#warning I need to also close sion file
			//sion_parclose_mapped_mpi(sid);
			FREAD(bytes,readData, sizeof(char), bSize, lfd,"pppppppp",localFileNames,splitRanks,readData,file_map,ranks,rank_map,chunkSizes);


			long data_written = sion_fwrite(readData, sizeof(char), bytes, sid);

			if (data_written < 0) {
				FTI_Print("Sionlib: could not write data", FTI_EROR);
				free(localFileNames);
				free(splitRanks);
				free(readData);
				fclose(lfd);
				sion_parclose_mapped_mpi(sid);
				free(file_map);
				free(ranks);
				free(rank_map);
				free(chunkSizes);
				return FTI_NSCS;
			}

			pos = pos + bytes;
		}
	}
	free(localFileNames);
	free(splitRanks);
	sion_parclose_mapped_mpi(sid);
	free(file_map);
	free(ranks);
	free(rank_map);
	free(chunkSizes);
}
#endif
