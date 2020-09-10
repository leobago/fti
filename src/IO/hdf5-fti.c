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
 *  @file   hdf5.c
 *  @date   November, 2017
 *  @brief  Funtions to support HDF5 checkpointing.
 */

#ifdef ENABLE_HDF5
#include <sys/types.h>
#include <dirent.h>
#include "../interface.h"

static hid_t _file_id;

/*-------------------------------------------------------------------------*/
/**
  @brief      activates heads for checkpoint postprocessing.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      status          preprocessing checkpoint status.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_ActivateHeadsHDF5(FTIT_configuration* FTI_Conf,
 FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
 FTIT_checkpoint* FTI_Ckpt, int status) {
    FTI_Exec->wasLastOffline = 1;
    // Head needs ckpt. ID to determine ckpt file name.
    int value = FTI_BASE + FTI_Exec->ckptMeta.level;  // Token to send to head
    if (status != FTI_SCES) {  // If Writing checkpoint failed
        value = FTI_REJW;  // Send reject checkpoint token to head
    }
    MPI_Send(&value, 1, MPI_INT, FTI_Topo->headRank, FTI_Conf->ckptTag,
     FTI_Exec->globalComm);
    MPI_Send(&FTI_Exec->ckptMeta.ckptId, 1, MPI_INT, FTI_Topo->headRank,
     FTI_Conf->ckptTag, FTI_Exec->globalComm);
    MPI_Send(&FTI_Exec->h5SingleFile, 1, MPI_C_BOOL, FTI_Topo->headRank,
     FTI_Conf->ckptTag, FTI_Exec->globalComm);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It creates h5datatype (hid_t) from definitions in FTI_Types
  @param      ftiType        FTI_Type type

  This function creates (opens) hdf5 compound type. Should be called only
  before saving checkpoint in HDF5 format. Build-in FTI's types are always open.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CreateComplexType(FTIT_type* ftiType) {
    char str[FTI_BUFS];

    // Sanity Checks
    if (!ftiType) return;
    if (ftiType->h5datatype >= 0) {
        snprintf(str, sizeof(str),
         "Type [%d] is already created.", ftiType->id);
        FTI_Print(str, FTI_DBUG);
        return;
    }
    // If type is simple (i.e initialized with FTI_InitType)
    if (!FTI_IsTypeComplex(ftiType)) {
        snprintf(str, sizeof(str), "Creating type [%d] as array of bytes.",
         ftiType->id);
        FTI_Print(str, FTI_DBUG);
        ftiType->h5datatype = H5Tcopy(H5T_NATIVE_CHAR);
        H5Tset_size(ftiType->h5datatype, ftiType->size);
        return;
    }
    // If type is complex (i.e initialized with FTI_InitComplexType)
    hid_t partTypes[FTI_BUFS];
    int i;
    // for each field create and rank-dimension array if needed
    for (i = 0; i < ftiType->structure->length; i++) {
        snprintf(str, sizeof(str), "Type [%d] trying to create new type [%d].",
         ftiType->id, ftiType->structure->field[i].type->id);
        FTI_Print(str, FTI_DBUG);
        FTI_CreateComplexType(ftiType->structure->field[i].type);
        partTypes[i] = ftiType->structure->field[i].type->h5datatype;
        if (ftiType->structure->field[i].rank > 1) {
            // need to create rank-dimension array type
            hsize_t dims[FTI_BUFS];
            int j;
            for (j = 0; j < ftiType->structure->field[i].rank; j++) {
                dims[j] = ftiType->structure->field[i].dimLength[j];
            }
            snprintf(str, sizeof(str), "Type [%d] trying to create %d-D array"
            " of type [%d].", ftiType->id, ftiType->structure->field[i].rank,
             ftiType->structure->field[i].type->id);
            FTI_Print(str, FTI_DBUG);
            partTypes[i] = H5Tarray_create(ftiType->structure->field[i]
                .type->h5datatype,
                ftiType->structure->field[i].rank, dims);
        } else {
            if (ftiType->structure->field[i].dimLength[0] > 1) {
                // need to create 1-dimension array type
                snprintf(str, sizeof(str), "Type [%d] trying to create 1-D"
                " [%d] array of type [%d].", ftiType->id,
                ftiType->structure->field[i].dimLength[0],
                ftiType->structure->field[i].type->id);
                FTI_Print(str, FTI_DBUG);
                hsize_t dim = ftiType->structure->field[i].dimLength[0];
                partTypes[i] = H5Tarray_create(ftiType->structure->
                    field[i].type->h5datatype, 1, &dim);
            }
        }
    }
    // create new HDF5 compound datatype
    snprintf(str, sizeof(str), "Creating type [%d].", ftiType->id);
    FTI_Print(str, FTI_DBUG);
    ftiType->h5datatype = H5Tcreate(H5T_COMPOUND, ftiType->size);
    snprintf(str, sizeof(str),
     "Type [%d] has hid_t %d.", ftiType->id, (int32_t)ftiType->h5datatype);
    FTI_Print(str, FTI_DBUG);
    if (ftiType->h5datatype < 0) {
        FTI_Print("FTI failed to create HDF5 type.", FTI_WARN);
    }
    // inserting component fields into compound datatype
    for (i = 0; i < ftiType->structure->length; i++) {
        snprintf(str, sizeof(str), "Insering type [%d] into new type [%d].",
         ftiType->structure->field[i].type->id, ftiType->id);
        FTI_Print(str, FTI_DBUG);
        herr_t res = H5Tinsert(ftiType->h5datatype,
         ftiType->structure->field[i].name,
          ftiType->structure->field[i].offset, partTypes[i]);
        if (res < 0) {
          snprintf(str, sizeof(str), "FTI faied to insert type [%d] in complex type [%d].",
            ftiType->structure->field[i].type->id, ftiType->id);
            FTI_Print(str, FTI_WARN);
        }
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It closes h5datatype
  @param      ftiType        FTI_Type type

  This function destroys (closes) hdf5 compound type. Should be called
  after saving checkpoint in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CloseComplexType(FTIT_type* ftiType) {
    if (!ftiType) return;

    char str[FTI_BUFS];
    if (ftiType->h5datatype == -1 || ftiType->id < 11) {
        // This type already closed or build-in type
        snprintf(str, sizeof(str), "Cannot close type [%d]. Build in "
            "or already closed.", ftiType->id);
        FTI_Print(str, FTI_DBUG);
        return;
    }

    if (ftiType->structure != NULL) {
        // array of bytes don't have structure
        int i;
        // close each field
        for (i = 0; i < ftiType->structure->length; i++) {
            snprintf(str, sizeof(str), "Closing type [%d] of compound type"
            " [%d].", ftiType->structure->field[i].type->id, ftiType->id);
            FTI_Print(str, FTI_DBUG);
            FTI_CloseComplexType(ftiType->structure->field[i].type);
        }
    }

    // close HDF5 datatype
    snprintf(str, sizeof(str), "Closing type [%d].", ftiType->id);
    FTI_Print(str, FTI_DBUG);
    herr_t res = H5Tclose(ftiType->h5datatype);
    if (res < 0) {
        FTI_Print("FTI failed to close HDF5 type.", FTI_WARN);
    }
    ftiType->h5datatype = -1;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It creates a group and all it's children
  @param      ftiGroup        FTI_H5Group to be create
  @param      parentGroup     hid_t of the parent

  This function creates hdf5 group and all it's children. Should be
  called only before saving checkpoint in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CreateGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup,
 FTIT_H5Group** FTI_Group) {
    ftiGroup->h5groupID = H5Gcreate2(parentGroup, ftiGroup->name, H5P_DEFAULT,
     H5P_DEFAULT, H5P_DEFAULT);
    if (ftiGroup->h5groupID < 0) {
        FTI_Print("FTI failed to create HDF5 group.", FTI_WARN);
        return;
    }

    int i;
    for (i = 0; i < ftiGroup->childrenNo; i++) {
        FTI_CreateGroup(FTI_Group[ftiGroup->childrenID[i]],
         ftiGroup->h5groupID, FTI_Group);  // Try to create the child
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It opens a group and all it's children
  @param      ftiGroup        FTI_H5Group to be opened
  @param      parentGroup     hid_t of the parent

  This function opens hdf5 group and all it's children. Should be
  called only before recovery in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_OpenGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup,
 FTIT_H5Group** FTI_Group) {
    ftiGroup->h5groupID = H5Gopen2(parentGroup, ftiGroup->name, H5P_DEFAULT);
    if (ftiGroup->h5groupID < 0) {
        FTI_Print("FTI failed to open HDF5 group.", FTI_WARN);
        return;
    }

    int i;
    for (i = 0; i < ftiGroup->childrenNo; i++) {
        FTI_OpenGroup(FTI_Group[ftiGroup->childrenID[i]],
         ftiGroup->h5groupID, FTI_Group);  // Try to open the child
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It closes a group and all it's children
  @param      ftiGroup        FTI_H5Group to be closed

  This function closes (destoys) hdf5 group and all it's children. Should be
  called only after saving checkpoint in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CloseGroup(FTIT_H5Group* ftiGroup, FTIT_H5Group** FTI_Group) {
    char str[FTI_BUFS];
    if (ftiGroup->h5groupID == -1) {
        // This group already closed, in tree this is error
        snprintf(str, FTI_BUFS, "Group %s is already closed?", ftiGroup->name);
        FTI_Print(str, FTI_WARN);
        return;
    }

    int i;
    for (i = 0; i < ftiGroup->childrenNo; i++) {
        // Try to close the child
        FTI_CloseGroup(FTI_Group[ftiGroup->childrenID[i]], FTI_Group);
    }

    herr_t res = H5Gclose(ftiGroup->h5groupID);
    if (res < 0) {
        FTI_Print("FTI failed to close HDF5 group.", FTI_WARN);
    }
    ftiGroup->h5groupID = -1;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Checks groups and datasets (callable recursively) 
  @param      gid             Parent group ID.
  @param      fn              File name.
  @return     integer         FTI_SCES if successful.

  This function analyses the group structure recursivley starting at group
  'gid'. It steps down in sub groups and checks datasets of consistency.
  The consistency check is performed if the dataset was created with the 
  fletcher32 filter activated. A dataset read will return a negative value 
  in that case if dataset corrupted.

 **/
/*-------------------------------------------------------------------------*/
int FTI_ScanGroup(hid_t gid, char* fn) {
    int res = FTI_SCES;
    char errstr[FTI_BUFS];
    hsize_t nobj;
    if (H5Gget_num_objs(gid, &nobj) >= 0) {
        int i;
        for (i = 0; i < nobj; i++) {
            int objtype;
            char dname[FTI_BUFS];
            char gname[FTI_BUFS];
            // determine if element is group or dataset
            objtype = H5Gget_objtype_by_idx(gid, (size_t)i);
            if (objtype == H5G_DATASET) {
                H5Gget_objname_by_idx(gid, (hsize_t)i, dname,
                 (size_t)FTI_BUFS);
                // open dataset
                hid_t did = H5Dopen1(gid, dname);
                if (did > 0) {
                    hid_t sid = H5Dget_space(did);
                    hid_t tid = H5Dget_type(did);
                    int drank = H5Sget_simple_extent_ndims(sid);
                    size_t typeSize = H5Tget_size(tid);
                    hsize_t *count = (hsize_t*) calloc(drank, sizeof(hsize_t));
                    hsize_t *offset = (hsize_t*) calloc(drank, sizeof(hsize_t));
                    count[0] = 1;
                    char* buffer = (char*) malloc(typeSize);
                    hid_t msid = H5Screate_simple(drank, count, NULL);
                    H5Sselect_hyperslab(sid, H5S_SELECT_SET, offset, NULL,
                     count, NULL);
                    // read element to trigger checksum comparison
                    herr_t status = H5Dread(did, tid, msid, sid, H5P_DEFAULT,
                     buffer);
                    if (status < 0) {
                        snprintf(errstr, FTI_BUFS, "unable to read from"
                        " dataset '%s' in file '%s'!", dname, fn);
                        FTI_Print(errstr, FTI_WARN);
                        res += FTI_NSCS;
                    }
                    H5Dclose(did);
                    H5Sclose(msid);
                    H5Sclose(sid);
                    H5Tclose(tid);
                    free(count);
                    free(offset);
                    free(buffer);
                } else {
                    snprintf(errstr, FTI_BUFS, "failed to open dataset '%s' in"
                    " file '%s'", dname, fn);
                    FTI_Print(errstr, FTI_WARN);
                    res += FTI_NSCS;
                }
            }
            // step down other group
            if (objtype == H5G_GROUP) {
                H5Gget_objname_by_idx(gid, (hsize_t)i, gname,
                 (size_t) FTI_BUFS);
                hid_t sgid = H5Gopen1(gid, gname);
                if (sgid > 0) {
                    res += FTI_ScanGroup(sgid, fn);
                    H5Gclose(sgid);
                } else {
                    snprintf(errstr, FTI_BUFS,
                     "failed to open group '%s' in file '%s'", gname, fn);
                    FTI_Print(errstr, FTI_WARN);
                    res += FTI_NSCS;
                }
            }
        }
    } else {
        snprintf(errstr, FTI_BUFS,
         "failed to get number of elements in file '%s'", fn);
        FTI_Print(errstr, FTI_WARN);
        res += FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns the file position.
  @param      fileDesc        The file descriptor.
  @return     integer         The position in the file.
 **/
/*-------------------------------------------------------------------------*/
size_t FTI_GetHDF5FilePos(void *fileDesc) {
    return 0;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Opens and HDF5 file (Only for write).
  @param      fileDesc        The file descriptor.
  @return     integer         FTI_SCES on success.
 **/
/*-------------------------------------------------------------------------*/
int FTI_HDF5Open(char *fn, void *fileDesc) {
    WriteHDF5Info_t *fd = (WriteHDF5Info_t*) fileDesc;
    char str[FTI_BUFS];
    // Creating new hdf5 file
    if (fd->FTI_Exec->h5SingleFile && fd->FTI_Conf->h5SingleFileIsInline) {
        hid_t plid = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plid, FTI_COMM_WORLD, MPI_INFO_NULL);
        fd->file_id = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, plid);
        H5Pclose(plid);
    } else {
        fd->file_id = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    if (fd->file_id < 0) {
        snprintf(str, sizeof(str),
         "FTI checkpoint file (%s) could not be opened.", fn);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    fd->FTI_Exec->H5groups[0]->h5groupID = fd->file_id;
    FTIT_H5Group* rootGroup = fd->FTI_Exec->H5groups[0];

    int i;
    for (i = 0; i < rootGroup->childrenNo; i++) {
        FTI_CreateGroup(fd->FTI_Exec->H5groups[rootGroup->childrenID[i]],
         fd->file_id, fd->FTI_Exec->H5groups);
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Commits a datatype in the hdf5 file format.
  @param      FTI_Exec          Execution environment parameters.
  @param data       Variable metadata to commit.
  @return   integer         FTI_SCES on success;
 **/
/*-------------------------------------------------------------------------*/
int FTI_CommitDataType(FTIT_execution *FTI_Exec, FTIT_dataset *data) {
    char str[FTI_BUFS];
    int toCommit = data->type->h5datatype < 0;
    FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];
    snprintf(str, sizeof(str),
     "Calling CreateComplexType [%d] with hid_t %d",
     data->type->id, (int32_t)data->type->h5datatype);
    FTI_Print(str, FTI_DBUG);
    FTI_CreateComplexType(data->type);
    if (toCommit == 1) {
        char name[FTI_BUFS];
        if (data->type->structure == NULL) {
            // this is the array of bytes with no name
            snprintf(name, sizeof(name), "Type%d", data->type->id);
        } else {
            strncpy(name, data->type->structure->name, FTI_BUFS);
        }

        herr_t res = H5Tcommit(data->type->h5group->h5groupID, name,
         data->type->h5datatype, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Datatype #%d could not be commited", data->id);
            FTI_Print(str, FTI_EROR);
            int j;
            for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
                FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]],
                 FTI_Exec->H5groups);
            }
            return FTI_NSCS;
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It checks if an hdf5 file exist and the contents are  'correct'.
  @param      fn              The ckpt. file name to check.
  @param      fs              The ckpt. file size to check.
  @param      checksum        The file checksum to check In this case is should be NULL.
  @return     integer         0 if file seems correct, 1 if not .

  This function checks whether a file exist or not and if its size is
  the expected one.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckHDF5File(char* fn, int32_t fs, char* checksum) {
    char str[FTI_BUFS];
    if (access(fn, F_OK) == 0) {
        struct stat fileStatus;
        if (stat(fn, &fileStatus) == 0) {
            if (fileStatus.st_size == fs) {
                hid_t file_id = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
                if (file_id < 0) {
                    snprintf(str, sizeof(str),
                     "Corrupted Checkpoint File: \"%s\"", fn);
                    FTI_Print(str, FTI_WARN);
                    return 1;
                } else {
                    H5Fclose(file_id);
                    return 0;
                }
            } else {
                return 1;
            }
        } else {
            return 1;
        }
    } else {
        char str[FTI_BUFS];
        snprintf(str, sizeof(str), "Missing file: \"%s\"", fn);
        FTI_Print(str, FTI_WARN);
        return 1;
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Calculates the  number of bytes I need to fetch from the GPU.
  @param      sizeOfElement   Element size for the dataset
  @param      maxBytes        The maximum amount of bytes I can fetch from GPU.
  @param      count           Array storing the number of dimensions I can fetch each time
  @param      numOfDimensions Number of dimensions of this dataset 
  @param      dimensions      maximum index of each dimension.
  @param      *sep            Stores the maximum dimension I can fetch from the GPU 
  @return     integer         number of bytes to fetch each time from the GPU.

  This function has a dual functionality. It computes the number of bytes I need to fetch
  from the GPU side. It also computes the "count" parameter. In other words it computes
  how many elements from each dimension I can compute from the total Bytes I fetched 
  from the GPU

 **/
/*-------------------------------------------------------------------------*/
hsize_t FTI_calculateCountDim(size_t sizeOfElement, hsize_t maxBytes,
 hsize_t *count, int numOfDimensions, hsize_t *dimensions, hsize_t *sep) {
    int i;
    memset(count, 0, sizeof(hsize_t)*numOfDimensions);
    size_t maxElements = maxBytes/sizeOfElement;
    hsize_t bytesToFetch;

    if (maxElements == 0)
        maxElements = 1;

    hsize_t *dimensionSize = (hsize_t *)malloc(sizeof(hsize_t)*
        (numOfDimensions+1));
    dimensionSize[numOfDimensions] = 1;

    // Calculate how many elements does each whole dimension holds
    for (i = numOfDimensions - 1; i >=0 ; i--) {
        dimensionSize[i] = dimensionSize[i+1] * dimensions[i];
    }

    // Find which is the maximum dimension that I can fetch continuously.
    for (i = numOfDimensions ; i >= 0; i--) {
        if (maxElements < dimensionSize[i]) {
            break;
        }
    }

    // I is =-1 when I can fetch the whole buffer
    if (i == -1) {
        *sep = 0;
        bytesToFetch = dimensionSize[*sep+1] * dimensions[*sep] *
         sizeOfElement;
        memcpy(count, dimensions, sizeof(hsize_t)*numOfDimensions);
        return bytesToFetch;
    }


    // Calculate the maxium elements of this dimension that I can get
    // This number should be a multiple of the total dimension lenght
    // of this dimension.
    *sep = i;
    int fetchElements = 0;
    for (i = maxElements/(dimensionSize[*sep+1]) ; i >= 1; i--) {
        if (dimensions[*sep]%i == 0) {
            fetchElements = i;
            break;
        }
    }

    // Fill in the count array and return.
    for (i = 0 ; i < *sep ; i++)
        count[i] = 1;

    count[*sep] = fetchElements;
    for (i = *sep+1; i < numOfDimensions ; i++)
        count[i] = dimensions[i];

    bytesToFetch = dimensionSize[*sep+1] * count[*sep] * sizeOfElement;

    return bytesToFetch;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes the elements to the HDF5 file.
  @param      dataspace       dataspace (shape) of the data to be stored.
  @param      dataType        data type of the data to be stored.
  @param      dataset         dataset of the data to be stored.
  @param      count           number of indexes for each dimension to be stored.
  @param      offset          describes the offset from the begining of the data to be stored 
  @param      ranks           number of dimensions of this dataset 
  @return     integer         FTI_SCES on succesfull write. 

  This function uses hyperslabs to store a subset of the entire dataspace
  to the checkpoint file.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteElements(hid_t dataspace, hid_t dataType, hid_t dataset,
 hsize_t *count, hsize_t *offset, hsize_t ranks, void *ptr) {
    char str[FTI_BUFS];
    hid_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset,
     NULL, count, NULL);
    hsize_t *dims_in = (hsize_t*) malloc(sizeof(hsize_t)*ranks);
    memcpy(dims_in, count, ranks*sizeof(hsize_t));
    hid_t memspace = H5Screate_simple(ranks, dims_in, NULL);
    hsize_t *offset_in = (hsize_t*)calloc(ranks, sizeof(ranks));
    status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_in, NULL,
     count, NULL);
    status = H5Dwrite(dataset, dataType, memspace, dataspace, H5P_DEFAULT,
     ptr);
    if (status < 0) {
        free(offset);
        free(count);
        snprintf(str, sizeof(str), "Dataset could not be written");
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    free(offset_in);
    free(dims_in);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Read the elements from the checkpoint file.
  @param      dataspace       dataspace (shape) of the data to be stored.
  @param      dataType        data type of the data to be stored.
  @param      dataset         dataset of the data to be stored.
  @param      count           number of indexes for each dimension to be stored.
  @param      offset          describes the offset from the begining of the data to be stored 
  @param      ranks           number of dimensions of this dataset 
  @return     integer         FTI_SCES on succesfull write. 

  This function uses hyperslabs to read a subset of the entire dataspace
  from the checkpoint file.
 **/
/*-------------------------------------------------------------------------*/
int FTI_ReadElements(hid_t dataspace, hid_t dimType, hid_t dataset,
 hsize_t *count, hsize_t *offset, hsize_t ranks, void *ptr) {
    char str[FTI_BUFS];
    hid_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset,
     NULL, count, NULL);
    hsize_t *dims_out = (hsize_t*)malloc(sizeof(hsize_t)*ranks);
    memcpy(dims_out, count, ranks*sizeof(hsize_t));
    hid_t memspace = H5Screate_simple(ranks, dims_out, NULL);
    hsize_t *offset_out = (hsize_t*)calloc(ranks, sizeof(ranks));
    status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out,
     NULL, count, NULL);
    status = H5Dread(dataset, dimType, memspace, dataspace, H5P_DEFAULT, ptr);
    if (status < 0) {
        free(offset);
        free(count);
        snprintf(str, sizeof(str), "Unable to read dataset");
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    free(offset_out);
    free(dims_out);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Advances the offset of an n-dimensional protected variable.
  @param      sep             The dimension index which is sliced. Dimension on the right side
  are transfered as a whole. Dimensions on the left side of sep
  are incrementally transfered.
  @param      start           coordinates of the n-dimensional space descirbing what I have processed up to 
  now.
  @param      add             How many elements I have processed.
  @param      dims            The entire n-dimensional space 
  @param      ranks           number of dimensions of this dataset 
  @return     integer         Return 1 only when the entire data set is computed (This is not used in the 
  current implementation). 
  This function performs acutally a simle addition on a n-dimensional spase
  start = start+ offset. I am processing only dimensions lower than "sep" as the higher
  ones are ALWAYS completely tranfered from/to the host.
 **/
/*-------------------------------------------------------------------------*/
int FTI_AdvanceOffset(hsize_t sep,  hsize_t *start, hsize_t *add,
 hsize_t *dims, hsize_t rank) {
    int i;
    hsize_t carryOut = 0;
    hsize_t temp;
    temp = start[sep] + add[sep];

    if (temp >= dims[sep]) {
        start[sep] = temp % dims[sep];
        carryOut = 1;
    } else {
        start[sep] = temp;
        carryOut = 0;
    }

    for (i = sep-1; i >= 0 &&carryOut ; i--) {
        temp = start[i] + carryOut;
        if (temp >= dims[i]) {
            start[i] = temp % dims[i];
            carryOut = 1;
        } else {
            start[i] = temp;
            carryOut = 0;
        }
    }
    return carryOut;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes a  protected variable to the checkpoint file.
  @param      data     The protected variable to be written. 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

  The function Write the data of a single FTI_Protect variable to the HDF5 file. 
  If the data are on the HOST CPU side, all the data are tranfered with a single call.
  If the data are on the GPU side, we use hyperslabs to slice the data and asynchronously
  move data from the GPU side to the host side and then to the filesytem.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteHDF5Var(FTIT_dataset *data, FTIT_execution* FTI_Exec) {
    int j;
    hsize_t dimLength[32];
    char str[FTI_BUFS];
    int res;
    hid_t dcpl;

    for (j = 0; j < data->rank; j++) {
        dimLength[j] = data->dimLength[j];
    }

    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    res = H5Pset_fletcher32(dcpl);
    res = H5Pset_chunk(dcpl, data->rank, dimLength);

    hid_t dataspace = H5Screate_simple(data->rank, dimLength, NULL);
    hid_t dataset;
    if (FTI_Exec->h5SingleFile) {
        hid_t globalDataset = data->sharedData.dataset->hid;
        dataset = H5Dcreate2(globalDataset, data->name,
         data->type->h5datatype, dataspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        int rank = 1;
        hsize_t att_dims = data->sharedData.dataset->rank;
        hid_t att_space = H5Screate_simple(rank, &att_dims, NULL);
        hid_t att_offset = H5Acreate(dataset, "offset", H5T_NATIVE_HSIZE,
         att_space, H5P_DEFAULT, H5P_DEFAULT);
        hid_t att_count = H5Acreate(dataset, "count", H5T_NATIVE_HSIZE,
         att_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(att_offset, H5T_NATIVE_HSIZE, data->sharedData.offset);
        H5Awrite(att_count, H5T_NATIVE_HSIZE, data->sharedData.count);
        H5Aclose(att_offset);
        H5Aclose(att_count);
    } else {
        dataset = H5Dcreate2(data->h5group->h5groupID, data->name,
         data->type->h5datatype, dataspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    // If my data are stored in the CPU side
    // Just store the data to the file and return;
#ifdef GPUSUPPORT
    if (!data->isDevicePtr) {
#endif
        res = H5Dwrite(dataset, data->type->h5datatype, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, data->ptr);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }

        res = H5Pclose(dcpl);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }

        res = H5Dclose(dataset);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        res = H5Sclose(dataspace);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        return FTI_SCES;
#ifdef GPUSUPPORT
    }

    // This code is only executed in the GPU case.

    hsize_t *count = (hsize_t*) malloc(sizeof(hsize_t)*data->rank);
    hsize_t *offset = (hsize_t*) calloc(data->rank, sizeof(hsize_t));

    if (!count|| !offset) {
        snprintf(str, sizeof(str),
         "Could Not allocate count and offset regions");
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }


    hsize_t seperator;
    hsize_t fetchBytes = FTI_getHostBuffSize();
    fetchBytes = FTI_calculateCountDim(data->eleSize, fetchBytes, count,
     data->rank, dimLength, &seperator);
    snprintf(str, sizeof(str),
     "GPU-Device Message: I Will Fetch %lld Bytes Per Stream Request",
      fetchBytes);
    FTI_Print(str, FTI_DBUG);


    FTIT_data_prefetch prefetcher;
    prefetcher.fetchSize = fetchBytes;
    prefetcher.totalBytesToFetch = data->size;
    prefetcher.isDevice = data->isDevicePtr;
    prefetcher.dptr = data->devicePtr;
    size_t bytesToWrite;
    FTI_InitPrefetcher(&prefetcher);
    unsigned char *basePtr = NULL;


    if (FTI_Try(FTI_getPrefetchedData(&prefetcher, &bytesToWrite, &basePtr),
     "Fetch next memory block from GPU to write to HDF5") !=  FTI_SCES) {
        return FTI_NSCS;
    }

    while (basePtr) {
        res = FTI_WriteElements(dataspace, data->type->h5datatype, dataset,
         count, offset, data->rank , basePtr);
        if (res != FTI_SCES) {
            free(offset);
            free(count);
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        FTI_AdvanceOffset(seperator, offset, count, dimLength, data->rank);

        if (FTI_Try(FTI_getPrefetchedData(&prefetcher, &bytesToWrite,
         &basePtr), "Fetch next memory block from GPU to"
         " write to HDF5") != FTI_SCES) {
            return FTI_NSCS;
        }
    }


    res = H5Dclose(dataset);
    if (res < 0) {
        free(offset);
        free(count);
        snprintf(str, sizeof(str),
         "Dataset #%d could not be written", data->id);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    res = H5Sclose(dataspace);
    if (res < 0) {
        free(offset);
        free(count);
        snprintf(str, sizeof(str),
         "Dataset #%d could not be written", data->id);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    free(offset);
    free(count);
    return FTI_SCES;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Reads a  protected variable to the checkpoint file.
  @param      data     The Var we will read from the  to the Checkpoint file 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

  The function reads the data of a single FTI_Protect variable to the HDF5 file. 
  If the data are on the HOST CPU side, all the data are tranfered with a single call.
  If the data should move to the GPU side, we use hyperslabs to slice the data and asynchronously
  move data from the File to the CPU and then to GPU side.
 **/
/*-------------------------------------------------------------------------*/
int FTI_ReadHDF5Var(FTIT_dataset *data) {
    char str[FTI_BUFS];
    herr_t res;

    hid_t dataset = H5Dopen(data->h5group->h5groupID, data->name, H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);

    // If my data are stored in the CPU side
    // Just store the data to the file and return;
#ifdef GPUSUPPORT
    if (!data->isDevicePtr) {
#endif
        res = H5Dread(dataset, data->type->h5datatype, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, data->ptr);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Dataset #%d could not be read", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }

        res = H5Dclose(dataset);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Dataset #%d could not be closed", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        res = H5Sclose(dataspace);
        if (res < 0) {
            snprintf(str, sizeof(str),
             "Dataset #%d dataspace could not be closed", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        return FTI_SCES;
#ifdef GPUSUPPORT
    }

    hsize_t dimLength[32];
    int j;
    for (j = 0; j < data->rank; j++) {
        dimLength[j] = data->dimLength[j];
    }

    // This code is only executed in the GPU case.


    hsize_t *count = (hsize_t*)malloc(sizeof(hsize_t)*data->rank);
    hsize_t *offset = (hsize_t*)calloc(data->rank, sizeof(hsize_t));

    if (!count|| !offset) {
        snprintf(str, sizeof(str),
         "Could Not allocate count and offset regions");
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }


    hsize_t seperator;
    size_t fetchBytes;
    size_t hostBufSize = FTI_getHostBuffSize();
    // Calculate How many dimension I can compute each time
    // and how bug should the HOST-GPU communication buffer should be

    fetchBytes = FTI_calculateCountDim(data->eleSize, hostBufSize, count,
     data->rank, dimLength, &seperator);

    // If the buffer is smaller than the minimum amount
    // then I need to allocate a bigger one.
    if (hostBufSize < fetchBytes) {
        if (FTI_Try(FTI_DestroyDevices(),
         "Deleting host buffers") != FTI_SCES) {
            free(offset);
            free(count);
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }

        if (FTI_Try (FTI_InitDevices(fetchBytes),
         "Allocating host buffers") != FTI_SCES) {
            free(offset);
            free(count);
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
    }

    unsigned char *basePtr = NULL;
    int id = 0;
    int prevId = 1;
    hsize_t totalBytes = data->size;
    cudaStream_t streams[2];
    // Create the streams for the asynchronous data movement.
    CUDA_ERROR_CHECK(cudaStreamCreate(&(streams[0])));
    CUDA_ERROR_CHECK(cudaStreamCreate(&(streams[1])));
    unsigned char *dPtr = data->devicePtr;
    // Perform the while loop until all data
    // are processed.
    while (totalBytes) {
        basePtr = FTI_getHostBuffer(id);
        // Read file
        res = FTI_ReadElements(dataspace, data->type->h5datatype, dataset,
         count, offset, data->rank , basePtr);
        CUDA_ERROR_CHECK(cudaMemcpyAsync(dPtr, basePtr, fetchBytes,
         cudaMemcpyHostToDevice, streams[id]));
        if (res != FTI_SCES) {
            free(offset);
            free(count);
            snprintf(str, sizeof(str),
             "Dataset #%d could not be written", data->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        // Increase accordingly the file offset
        FTI_AdvanceOffset(seperator, offset, count, dimLength, data->rank);
        // Syncing the cuda stream.
        CUDA_ERROR_CHECK(cudaStreamSynchronize(streams[prevId]));
        prevId = id;
        id = (id + 1)%2;
        dPtr = dPtr + fetchBytes;
        totalBytes -= fetchBytes;
    }
    CUDA_ERROR_CHECK(cudaStreamSynchronize(streams[prevId]));
    CUDA_ERROR_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_ERROR_CHECK(cudaStreamDestroy(streams[1]));

    res = H5Dclose(dataset);
    if (res < 0) {
        free(offset);
        free(count);
        snprintf(str, sizeof(str),
         "Dataset #%d could not be written", data->id);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    res = H5Sclose(dataspace);
    if (res < 0) {
        free(offset);
        free(count);
        snprintf(str, sizeof(str), "Dataset #%d could not be written",
         data->id);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    free(offset);
    free(count);
    return FTI_SCES;
#endif
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Writes the specific variable in the file.
  @param      data     The Var we will write to the Checkpoint file 
  @param      write_info      The fileDescriptor 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteHDF5Data(FTIT_dataset *data, void *write_info) {
    WriteHDF5Info_t *fd = (WriteHDF5Info_t *) write_info;
    char str[FTI_BUFS];
    int res;
    FTIT_H5Group* rootGroup = fd->FTI_Exec->H5groups[0];

    if (fd->FTI_Exec->h5SingleFile && fd->FTI_Conf->h5SingleFileIsInline) {
        res = FTI_WriteSharedFileData(*data);
    } else {
        FTI_CommitDataType(fd->FTI_Exec, data);
        res = FTI_WriteHDF5Var(data, fd->FTI_Exec);
    }
    if (res != FTI_SCES) {
        int j;
        snprintf(str, sizeof(str), "Dataset #%d could not be written",
         data->id);
        FTI_Print(str, FTI_EROR);
        for (j = 0; j < fd->FTI_Exec->H5groups[0]->childrenNo; j++) {
            FTI_CloseGroup(fd->FTI_Exec->H5groups[rootGroup->childrenID[j]],
             fd->FTI_Exec->H5groups);
        }
        H5Fclose(fd->file_id);
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Closes the HDF5 file  
  @param      fileDesc          The fileDescriptor 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

 **/
/*-------------------------------------------------------------------------*/
int  FTI_HDF5Close(void *fileDesc) {
    int i, j, status = FTI_SCES;
    WriteHDF5Info_t *fd = (WriteHDF5Info_t *)fileDesc;
    FTIT_H5Group* rootGroup = fd->FTI_Exec->H5groups[0];
    FTIT_keymap* FTI_Data = fd->FTI_Data;

    FTIT_dataset* data;
    if (FTI_Data->data(&data, fd->FTI_Exec->nbVar) != FTI_SCES)
        return FTI_NSCS;

    for (i = 0; i < fd->FTI_Exec->nbVar; i++) {
        FTI_CloseComplexType(data[i].type);
    }

    for (j = 0; j < fd->FTI_Exec->H5groups[0]->childrenNo; j++) {
        FTI_CloseGroup(fd->FTI_Exec->H5groups[rootGroup->childrenID[j]],
         fd->FTI_Exec->H5groups);
    }

    if (fd->FTI_Exec->h5SingleFile) {
        if (fd->FTI_Conf->h5SingleFileIsInline) {
            FTI_CloseGlobalDatasets(fd->FTI_Exec);
        } else {
            FTI_CloseGlobalDatasetsAsGroups(fd->FTI_Exec);
        }
    }

    // close file
    fd->FTI_Exec->H5groups[0]->h5groupID = -1;
    if (H5Fclose(fd->file_id) < 0) {
        FTI_Print("FTI checkpoint file could not be closed.", FTI_EROR);
        return FTI_NSCS;
    }
    if (fd->FTI_Exec->h5SingleFile) {
        bool removeLastFile = !fd->FTI_Conf->h5SingleFileKeep &&
         (bool)strcmp(fd->FTI_Exec->h5SingleFileLast, "") &&
          fd->FTI_Conf->h5SingleFileIsInline;
        if (removeLastFile && !fd->FTI_Topo->splitRank) {
            status = remove(fd->FTI_Exec->h5SingleFileLast);
            if ((status != ENOENT) && (status != 0)) {
                char errstr[FTI_BUFS];
                snprintf(errstr, FTI_BUFS, "failed to remove last VPR "
                    "file '%s'", fd->FTI_Exec->h5SingleFileLast);
                FTI_Print(errstr, FTI_EROR);
            } else {
                status = FTI_SCES;
            }
        }
        if (status == FTI_SCES) {
            snprintf(fd->FTI_Exec->h5SingleFileLast, FTI_BUFS,
             "%s/%s-ID%08d.h5", fd->FTI_Conf->h5SingleFileDir,
             fd->FTI_Conf->h5SingleFilePrefix, fd->FTI_Exec->ckptMeta.ckptId);
        }
    }

    return status;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes the files for the upcoming checkpoint.  
  @param      FTI_Conf          Configuration of FTI 
  @param      FTI_Exec          Execution environment options 
  @param      FTI_Topo          Topology of nodes
  @param      FTI_Ckpt          Checkpoint configurations
  @param      FTI_Data          Data to be stored
  @return     void*             Return void pointer to file descriptor 

 **/
/*-------------------------------------------------------------------------*/
void *FTI_InitHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_keymap *FTI_Data) {
    FTI_Print("I/O mode: HDF5.", FTI_DBUG);

    if (FTI_Exec->h5SingleFile) {
        if (!FTI_Conf->h5SingleFileEnable) {
            FTI_Print("VPR is disabled. Please enable with "
                "'h5_single_file_enable=1'!", FTI_WARN);
            return NULL;
        }
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // Cannot check dimensions when using icp! FIXME
        // will only succeed when all subsets are added. In iCP this might not
        // have been happen yet until this point.
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // if (FTI_CheckDimensions(FTI_Data, FTI_Exec) != FTI_SCES) {
        //    FTI_Print("Dimension missmatch in VPR file. Checkpoint failed!",
        //    FTI_WARN);
        //    return NULL;
        //}
    }

    char  fn[FTI_BUFS];
    int level = FTI_Exec->ckptMeta.level;

    // update ckpt file name
    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.%s",
     FTI_Exec->ckptMeta.ckptId, FTI_Topo->myRank, FTI_Conf->suffix);

    // If inline L4 save directly to global directory
    if (level == 4 && FTI_Ckpt[4].isInline && !FTI_Exec->h5SingleFile) {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir,
         FTI_Exec->ckptMeta.ckptFile);
    } else if (FTI_Exec->h5SingleFile && FTI_Conf->h5SingleFileIsInline) {
        snprintf(fn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->gTmpDir,
         FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptMeta.ckptId);
    } else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir,
         FTI_Exec->ckptMeta.ckptFile);
    }

    int i;
    WriteHDF5Info_t *fd = (WriteHDF5Info_t*) malloc(sizeof(WriteHDF5Info_t));;
    fd->FTI_Exec = FTI_Exec;
    fd->FTI_Data = FTI_Data;
    fd->FTI_Conf = FTI_Conf;
    fd->FTI_Topo = FTI_Topo;

    FTI_HDF5Open(fn, fd);

    if (FTI_Exec->h5SingleFile) {
        FTIT_dataset* data;
        if (FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) return NULL;
        for (i = 0; i < FTI_Exec->nbVar; i++) {
            FTI_CommitDataType(FTI_Exec, &data[i]);
        }
        if (fd->FTI_Conf->h5SingleFileIsInline) {
            FTI_CreateGlobalDatasets(FTI_Exec);
        } else {
            FTI_CreateGlobalDatasetsAsGroups(FTI_Exec);
        }
    }
    return (void *) fd;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt using HDF5 file format.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data) {
    // write data
    WriteHDF5Info_t *fd = FTI_InitHDF5(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt,
     FTI_Data);
    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) return FTI_NSCS;
    int i= 0; for (; i < FTI_Exec->nbVar; i++) {
        FTI_WriteHDF5Data(&data[i], fd);
    }
    FTI_HDF5Close(fd);
    free(fd);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It loads the HDF5 checkpoint data.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function loads the checkpoint data from the checkpoint file and
  it updates checkpoint information.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data) {
    char str[FTI_BUFS], fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir,
     FTI_Exec->ckptMeta.ckptFile);
    if (FTI_Exec->h5SingleFile) {
        snprintf(fn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->h5SingleFileDir,
         FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptId);
    }

    snprintf(str, sizeof(str),
     "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);

    hid_t file_id;

    // Open hdf5 file
    if (FTI_Exec->h5SingleFile) {
        hid_t plid = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plid, FTI_COMM_WORLD, MPI_INFO_NULL);
        file_id = H5Fopen(fn, H5F_ACC_RDONLY, plid);
        H5Pclose(plid);
    } else {
        file_id = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
    }
    if (file_id < 0) {
        snprintf(str, FTI_BUFS, "Could not open FTI checkpoint file '%s'.", fn);
        FTI_Print(str, FTI_EROR);
        return FTI_NREC;
    }
    FTI_Exec->H5groups[0]->h5groupID = file_id;
    FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

    int i;
    for (i = 0; i < FTI_Exec->H5groups[0]->childrenNo; i++) {
        FTI_OpenGroup(FTI_Exec->H5groups[rootGroup->childrenID[i]], file_id,
         FTI_Exec->H5groups);
    }

    FTIT_dataset* data;

    if (FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) return FTI_NSCS;

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_CreateComplexType(data[i].type);
    }

    if (FTI_Exec->h5SingleFile) {
        FTI_OpenGlobalDatasets(FTI_Exec);
    }

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        herr_t res;
        if (FTI_Exec->h5SingleFile) {
            res = FTI_ReadSharedFileData(data[i]);
        } else {
            res = FTI_ReadHDF5Var(&data[i]);
        }
        if (res < 0) {
            FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
            int j;
            for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
                FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]],
                 FTI_Exec->H5groups);
            }
            H5Fclose(file_id);
            return FTI_NREC;
        }
    }
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_CloseComplexType(data[i].type);
    }

    int j;
    for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
        FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]],
         FTI_Exec->H5groups);
    }

    if (FTI_Exec->h5SingleFile) {
        FTI_CloseGlobalDatasets(FTI_Exec);
    }

    FTI_Exec->H5groups[0]->h5groupID = -1;
    if (H5Fclose(file_id) < 0) {
        FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }
    FTI_Exec->reco = 0;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initialization for seperate variable recovery.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function prepares the checkpoint file to enable the recovery of
  the protected variables seperately.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarInitHDF5(FTIT_configuration* FTI_Conf,
 FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt) {
    char str[FTI_BUFS], fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir,
     FTI_Exec->ckptMeta.ckptFile);

    if (FTI_Exec->h5SingleFile) {
        snprintf(fn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->h5SingleFileDir,
         FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptId);
    }
    if (FTI_Exec->h5SingleFile) {
        hid_t plid = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plid, FTI_COMM_WORLD, MPI_INFO_NULL);
        _file_id = H5Fopen(fn, H5F_ACC_RDONLY, plid);
        H5Pclose(plid);
    } else {
        _file_id = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
    }

    if (_file_id < 0) {
        snprintf(str, FTI_BUFS, "Could not open FTI checkpoint file '%s'.", fn);
        FTI_Print(str, FTI_EROR);
        return FTI_NREC;
    }
    FTI_Exec->H5groups[0]->h5groupID = _file_id;
    FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

    int i;
    for (i = 0; i < FTI_Exec->H5groups[0]->childrenNo; i++) {
        FTI_OpenGroup(FTI_Exec->H5groups[rootGroup->childrenID[i]], _file_id,
         FTI_Exec->H5groups);
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalize for seperate variable recovery.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function finalizes the checkpoint file after the recovery of
  the protected variables seperately.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarFinalizeHDF5(FTIT_configuration* FTI_Conf,
 FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data) {
    int i;
    FTIT_dataset* data;

    if (FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) return FTI_NSCS;

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_CloseComplexType(data[i].type);
    }

    FTI_Exec->H5groups[0]->h5groupID = _file_id;
    FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];
    for (i = 0; i < FTI_Exec->H5groups[0]->childrenNo; i++) {
        FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[i]],
         FTI_Exec->H5groups);
    }

    if (H5Fclose(_file_id) < 0) {
        FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      During the restart, recovers the given variable
  @param      id              Variable to recover
  @return     int             FTI_SCES if successful.

  During a restart process, this function recovers the variable specified
  by the given id.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data, int id) {
    FTIT_dataset* data;
    char str[FTI_BUFS], fn[FTI_BUFS];

    if (FTI_Data->get(&data, id) != FTI_SCES) return FTI_NSCS;

    if (!data) {
        FTI_Print("could not find ID!", FTI_WARN);
        return FTI_NSCS;
    }

    FTI_CreateComplexType(data->type);

    if (FTI_Exec->h5SingleFile) {
        FTI_OpenGlobalDatasets(FTI_Exec);
    }

    herr_t res;

    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir,
     FTI_Exec->ckptMeta.ckptFile);

    if (FTI_Exec->h5SingleFile) {
        snprintf(fn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->h5SingleFileDir,
         FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptId);
    } else {
        if (data->size != data->sizeStored) {
            snprintf(str, sizeof(str), "Cannot recover %d bytes to "
                "protected variable (ID %d) size: %d",
                    data->sizeStored, data->id,
                    data->size);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
        snprintf(str, sizeof(str),
         "Trying to load FTI checkpoint file (%s)...", fn);
    }

    FTI_Print(str, FTI_DBUG);

    if (FTI_Exec->h5SingleFile) {
        res = FTI_ReadSharedFileData(*data);
    } else {
        res = FTI_ReadHDF5Var(data);
    }

    if (res < 0) {
        FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
    }

    FTI_CloseComplexType(data->type);

    if (FTI_Exec->h5SingleFile) {
        FTI_CloseGlobalDatasets(FTI_Exec);
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Opens global dataset in VPR file.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  This function is the analog to 'FTI_CreateGlobalDatasets' on recovery.
 **/
/*-------------------------------------------------------------------------*/
int FTI_OpenGlobalDatasets(FTIT_execution* FTI_Exec) {
    hsize_t *dims = NULL;
    hsize_t *maxDims = NULL;

    char errstr[FTI_BUFS];
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while (dataset) {
        dims = (hsize_t*) realloc(dims, sizeof(hsize_t)*dataset->rank);
        maxDims = (hsize_t*) realloc(maxDims, sizeof(hsize_t)*dataset->rank);

        // open dataset
        hid_t loc = dataset->location->h5groupID;
        hid_t tid = dataset->type->h5datatype;
        dataset->hdf5TypeId = tid;

        dataset->hid = H5Dopen(loc, dataset->name, H5P_DEFAULT);
        if (dataset->hid < 0) {
            snprintf(errstr, FTI_BUFS, "failed to open dataset '%s'",
             dataset->name);
            FTI_Print(errstr, FTI_WARN);
            return FTI_NSCS;
        }

        // get file space and check if rank and dimension
        // coincide for file and execution
        hid_t fsid = H5Dget_space(dataset->hid);
        if (fsid > 0) {
            int rank = H5Sget_simple_extent_ndims(fsid);
            if (rank == dataset->rank) {
                H5Sget_simple_extent_dims(fsid, dims, maxDims);
                if (memcmp(dims, dataset->dimension, sizeof(hsize_t)*rank)) {
                    snprintf(errstr, FTI_BUFS, "stored and requested "
                        "dimensions of dataset '%s' differ!", dataset->name);
                    FTI_Print(errstr, FTI_WARN);
                    return FTI_NSCS;
                }
            } else {
                snprintf(errstr, FTI_BUFS, "stored and requested rank of "
                    "dataset '%s' differ (stored:%d != requested:%d)!",
                     dataset->name, rank, dataset->rank);
                FTI_Print(errstr, FTI_WARN);
                return FTI_NSCS;
            }
        } else {
            snprintf(errstr, FTI_BUFS, "failed to acquire data space"
            " information of dataset '%s'", dataset->name);
            FTI_Print(errstr, FTI_WARN);
            return FTI_NSCS;
        }
        dataset->fileSpace = fsid;

        dataset->initialized = true;

        dataset = dataset->next;
    }

    if (dims) { free(dims); }
    if (maxDims) { free(maxDims); }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Reads a sub-set of a global dataset on recovery.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
herr_t FTI_ReadSharedFileData(FTIT_dataset FTI_Data) {
    // hdf5 datatype
    hid_t tid = FTI_Data.sharedData.dataset->hdf5TypeId;

    // dataset hdf5-id
    hid_t did = FTI_Data.sharedData.dataset->hid;

    // shared dataset file space
    hid_t fsid = FTI_Data.sharedData.dataset->fileSpace;

    // shared dataset rank
    int ndim = FTI_Data.sharedData.dataset->rank;

    // shared dataset array of nummber of elements in each dimension
    hsize_t *count = FTI_Data.sharedData.count;

    // shared dataset array of the offsets for each dimension
    hsize_t *offset = FTI_Data.sharedData.offset;

    // create dataspace for subset of shared dataset
    hid_t msid = H5Screate_simple(ndim, count, NULL);
    if (msid < 0) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS, "Unable to create space for var-id %d "
            "in dataset #%d", FTI_Data.id, FTI_Data.sharedData.dataset->id);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }

    // select range in shared dataset in file
    H5Sselect_hyperslab(fsid, H5S_SELECT_SET, offset, NULL, count, NULL);

    // enable collective buffering
    hid_t plid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);

    // write data in file
    herr_t status = H5Dread(did, tid, msid, fsid, plid, FTI_Data.ptr);
    if (status < 0) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS, "Unable to read var-id %d from dataset #%d",
         FTI_Data.id, FTI_Data.sharedData.dataset->id);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }

    H5Sclose(msid);
    H5Pclose(plid);

    return status;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Closes global datasets in VPR file 
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_CloseGlobalDatasets(FTIT_execution* FTI_Exec) {
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while (dataset) {
        H5Sclose(dataset->fileSpace);

        H5Dclose(dataset->hid);

        dataset = dataset->next;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Closes global datasets in VPR file 
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_CloseGlobalDatasetsAsGroups(FTIT_execution* FTI_Exec) {
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while (dataset) {
        H5Gclose(dataset->hid);

        dataset = dataset->next;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes global dataset subsets into VPR file.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
herr_t FTI_WriteSharedFileData(FTIT_dataset FTI_Data) {
    if (FTI_Data.sharedData.dataset) {
        // hdf5 datatype
        hid_t tid = FTI_Data.sharedData.dataset->hdf5TypeId;

        // dataset hdf5-id
        hid_t did = FTI_Data.sharedData.dataset->hid;

        // shared dataset file space
        hid_t fsid = FTI_Data.sharedData.dataset->fileSpace;

        // shared dataset rank
        int ndim = FTI_Data.sharedData.dataset->rank;

        // shared dataset array of nummber of elements in each dimension
        hsize_t *count = FTI_Data.sharedData.count;

        // shared dataset array of the offsets for each dimension
        hsize_t *offset = FTI_Data.sharedData.offset;

        // create dataspace for subset of shared dataset
        hid_t msid = H5Screate_simple(ndim, count, NULL);
        if (msid < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS,
             "Unable to create space for var-id %d in dataset #%d",
              FTI_Data.id, FTI_Data.sharedData.dataset->id);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        // select range in shared dataset in file
        if (H5Sselect_hyperslab(fsid, H5S_SELECT_SET, offset,
         NULL, count, NULL) < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS,
             "Unable to select sub-space for var-id %d in dataset #%d",
              FTI_Data.id, FTI_Data.sharedData.dataset->id);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        // enable collective buffering
        hid_t plid = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);

        // write data in file
        if (H5Dwrite(did, tid, msid, fsid, plid, FTI_Data.ptr) < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS,
             "Unable to write var-id %d of dataset #%d", FTI_Data.id,
              FTI_Data.sharedData.dataset->id);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        H5Sclose(msid);
        H5Pclose(plid);
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      returns rank of global dataset.
  @param      did       HDF5 internal dataset id (hid_t).
  @return     integer   rank of global dataset.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetDatasetRankReco(hid_t did) {
    hid_t sid;

    sid = H5Dget_space(did);

    int drank = H5Sget_simple_extent_ndims(sid);

    H5Sclose(sid);

    return drank;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      returns span of global dataset.
  @param      did       HDF5 internal dataset id (hid_t).
  @param      span      span of global dataset 
  @return     integer   FTI_SCES if successful.

  this function stores the global dataset dimension in the array variable 
  span. The array has to be allocated to the respective rank beforehand.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetDatasetSpanReco(hid_t did, hsize_t * span) {
    hid_t sid;

    sid = H5Dget_space(did);

    H5Sget_simple_extent_dims(sid, span, NULL);

    H5Sclose(sid);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Checks for matching dimension sizes of sub-sets
  @param      FTI_Data        Dataset metadata.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  This function counts the number of elements of all sub-sets contained in a 
  particular global dataset and accumulates to a total value. If the accu-
  mulated value matches the number of elements defined for the global data-
  set, FTI_SCES is returned. This function is called before the checkpoint
  and before the recovery.

  @todo it would be great to check for region overlapping too.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckDimensions(FTIT_keymap * FTI_Data, FTIT_execution * FTI_Exec) {
    // NOTE checking for overlap is complicated and likely expensive
    // since it requires sorting within all contributing processes.
    // Thus, we check currently only the number of elements.
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while (dataset) {
        int i, j;
        // sum of local elements
        hsize_t numElemLocal = 0, numElemGlobal;
        for (i = 0; i < dataset->numSubSets; ++i) {
            FTIT_dataset* data;

            if (FTI_Data->get(&data, dataset->varId[i]) != FTI_SCES)
                return FTI_NSCS;

            if (!data) {
                FTI_Print("could not find ID!", FTI_WARN);
                return FTI_NSCS;
            }
            hsize_t numElemSubSet = 1;
            for (j = 0; j < dataset->rank; j++) {
                numElemSubSet *= data->sharedData.count[j];
            }
            numElemLocal += numElemSubSet;
        }
        // number of elements in global dataset
        hsize_t numElem = 1;
        for (i = 0; i < dataset->rank; ++i) {
            numElem *= dataset->dimension[i];
        }
        MPI_Allreduce(&numElemLocal, &numElemGlobal, 1, MPI_UINT64_T,
         MPI_SUM, FTI_COMM_WORLD);
        if (numElem != numElemGlobal) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Number of elements of subsets "
                "(accumulated) do not match number of elements defined "
                "for global dataset #%d!", dataset->id);
            FTI_Print(errstr, FTI_WARN);
            return FTI_NSCS;
        }
        dataset = dataset->next;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Checks if VPR file on restart
  @param      FTI_Conf        Configuration metadata.
  @param      ckptId          Checkpoint ID.
  @return     integer         FTI_SCES if successful.

  Checks if restart is possible for VPR file. 
  1) Checks if file exist for prefix and directory, defined in config file.
  2) Checks if is regular file.
  3) Checks if groups and datasets can be accessed and if datasets can be 
  read

  If file found and sane, ckptId is set and FTI_SCES is returned.

 **/
/*-------------------------------------------------------------------------*/
int FTI_H5CheckSingleFile(FTIT_configuration* FTI_Conf, int *ckptId) {
    char errstr[FTI_BUFS];
    char fn[FTI_BUFS];
    int res = FTI_SCES;
    struct stat st;

    struct dirent *entry;
    DIR *dir = opendir(FTI_Conf->h5SingleFileDir);

    if (dir == NULL) {
        snprintf(errstr, FTI_BUFS, "VPR directory '%s' could not be accessed.",
         FTI_Conf->h5SingleFileDir);
        FTI_Print(errstr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    *ckptId = -1;

    bool found = false;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            int len = strlen(entry->d_name);
            if (len > 14) {
                char fileRoot[FTI_BUFS];
                bzero(fileRoot, FTI_BUFS);
                memcpy(fileRoot, entry->d_name, len - 14);
                char fileRootExpected[FTI_BUFS];
                snprintf(fileRootExpected, FTI_BUFS, "%s",
                 FTI_Conf->h5SingleFilePrefix);
                if (strncmp(fileRootExpected, fileRoot, FTI_BUFS) == 0) {
                    int id_tmp;
                    sscanf(entry->d_name + len - 14 + 3, "%08d.h5", &id_tmp);
                    if (id_tmp > *ckptId) {
                        *ckptId = id_tmp;
                        snprintf(fn, FTI_BUFS, "%s/%s-ID%08d.h5",
                         FTI_Conf->h5SingleFileDir,
                         FTI_Conf->h5SingleFilePrefix, *ckptId);
                    }
                    found = true;
                }
            }
        }
    }

    if (!found) {
        snprintf(errstr, FTI_BUFS, "unable to find matching VPR file "
            "(filename pattern: '%s-ID########.h5')!",
             FTI_Conf->h5SingleFilePrefix);
        FTI_Print(errstr, FTI_WARN);
        return FTI_NSCS;
    }

    stat(fn, &st);
    if (S_ISREG(st.st_mode)) {
        hid_t fid = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid > 0) {
            hid_t gid = H5Gopen1(fid, "/");
            if (gid > 0) {
                res += FTI_ScanGroup(gid, fn);
                H5Gclose(gid);
            } else {
                snprintf(errstr, FTI_BUFS,
                 "failed to access root group in file '%s'", fn);
                FTI_Print(errstr, FTI_WARN);
                res = FTI_NSCS;
            }
            H5Fclose(fid);
        } else {
            snprintf(errstr, FTI_BUFS, "failed to open file '%s'", fn);
            FTI_Print(errstr, FTI_WARN);
            res = FTI_NSCS;
        }
    } else {
        snprintf(errstr, FTI_BUFS, "'%s', is not a regular file!", fn);
        FTI_Print(errstr, FTI_WARN);
        res = FTI_NSCS;
    }
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      frees all memory allocated for VPR.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Data to be stored
 **/
/*-------------------------------------------------------------------------*/
void FTI_FreeVPRMem(FTIT_execution* FTI_Exec, FTIT_keymap* FTI_Data) {
    FTIT_globalDataset * dataset = FTI_Exec->globalDatasets;
    while (dataset) {
        free(dataset->dimension);
        free(dataset->varId);
        dataset->dimension = NULL;
        dataset->varId = NULL;
        FTIT_globalDataset * curr = dataset;
        dataset = dataset->next;
        free(curr);
    }

    FTIT_dataset* data;
    if ((FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) || !data) return;

    int i = 0; for (; i < FTI_Exec->nbVar; i++) {
        if (data[i].sharedData.offset) {
            free(data[i].sharedData.offset);
        }
        if (data[i].sharedData.count) {
            free(data[i].sharedData.count);
        }
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It creates the global dataset in the VPR file.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  Creates global dataset (shared among all ranks) in VPR file. The dataset
  position will be the group assigned to it by calling the FTI API function 
  'FTI_DefineGlobalDataset'.
 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateGlobalDatasets(FTIT_execution* FTI_Exec) {
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while (dataset) {
        // create file space
        dataset->fileSpace = H5Screate_simple(dataset->rank,
         dataset->dimension, NULL);

        if (dataset->fileSpace < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS,
             "Unable to create space for dataset #%d", dataset->id);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        // create dataset
        hid_t loc = dataset->location->h5groupID;
        hid_t tid = dataset->type->h5datatype;
        dataset->hdf5TypeId = tid;
        hid_t fsid = dataset->fileSpace;

        // FLETCHER CHECKSUM NOT SUPPORTED FOR PARALLEL I/O IN HDF5
        hid_t plid = H5Pcreate(H5P_DATASET_CREATE);
        // H5Pset_fletcher32 (dcpl);
        //
        // hsize_t *chunk = malloc(sizeof(hsize_t) * dataset->rank);
        // chunk[0] = chunk[1] = 4096;
        // H5Pset_chunk (dcpl, 2, chunk);

        dataset->hid = H5Dcreate(loc, dataset->name, tid, fsid, H5P_DEFAULT,
         plid, H5P_DEFAULT);
        if (dataset->hid < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to create dataset #%d",
             dataset->id);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        H5Pclose(plid);

        dataset->initialized = true;

        dataset = dataset->next;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It creates the global dataset as group in the preprocessing file.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  When head=1 and singleshared file is not inline, the information of the
  global dataset is stored as attribute inside a group with name of
  the global dataset. all the corresponding subsets will be stored inside
  this group.
 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateGlobalDatasetsAsGroups(FTIT_execution* FTI_Exec) {
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while (dataset) {
        // create dataset as group
        hid_t loc = dataset->location->h5groupID;

        dataset->hid = H5Gcreate2(loc, dataset->name, H5P_DEFAULT,
         H5P_DEFAULT, H5P_DEFAULT);
        if (dataset->hid < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to create dataset #%d as"
            " group", dataset->id);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        int rank = 1;
        hsize_t att_dims = dataset->rank;
        hid_t att_space = H5Screate_simple(rank, &att_dims, NULL);
        hid_t aid = H5Acreate(dataset->hid, "global_dimension",
         H5T_NATIVE_HSIZE, att_space, H5P_DEFAULT, H5P_DEFAULT);
        herr_t err = H5Awrite(aid, H5T_NATIVE_HSIZE, dataset->dimension);
        if (err) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to add attributes to"
            " dataset #%d", dataset->id);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        H5Aclose(aid);

        dataset->initialized = true;

        dataset = dataset->next;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      reads attribute from object (either offset or count).
  @param      oid       HDF5 internal object id 
  @param      name      object name 
  @param      buffer    storage for offset or count arrays
  @return     integer   FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_ReadAttributeHDF5(hid_t oid, char* name, hsize_t* buffer) {
    hid_t aid, sid;
    hsize_t rank;
    aid = H5Aopen(oid, name, H5P_DEFAULT);
    if (aid < 0) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS,
         "Unable to access attributes from '%s'", name);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    sid = H5Aget_space(aid);
    H5Sget_simple_extent_dims(sid, &rank, NULL);
    herr_t err = H5Aread(aid, H5T_NATIVE_HSIZE, buffer);
    if (err) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS,
         "Unable to read attributes from '%s'", name);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    H5Sclose(sid);
    H5Aclose(aid);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      returns rank of attribute from object.
  @param      oid       HDF5 internal object id 
  @return     integer   rank of attribute.

  This function returns the rank of the attribute of object. This rank 
  corresponds to the global dataset rank.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetDatasetRankFlush(hid_t oid) {
    hid_t aid, sid;
    hsize_t rank;
    aid = H5Aopen_idx(oid, 0);
    if (aid < 0) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS, "Unable to access dataset attributes");
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    sid = H5Aget_space(aid);
    H5Sget_simple_extent_dims(sid, &rank, NULL);
    H5Sclose(sid);
    H5Aclose(aid);

    return rank;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      returns active HDF5 type of global dataset.
  @param      did       HDF5 internal dataset id 
  @return     integer   rank of attribute.
 **/
/*-------------------------------------------------------------------------*/
hid_t FTI_GetDatasetTypeFlush(hid_t gid) {
    char objname[FTI_BUFS];
    H5Gget_objname_by_idx(gid, 0, objname, FTI_BUFS);
    hid_t did = H5Dopen(gid, objname, H5P_DEFAULT);
    if (did < 0) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS, "Unable to access dataset '%s'", objname);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    hid_t tid = H5Dget_type(did);
    H5Dclose(did);
    return tid;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      This function adds local subsets from each rank of head to
  global dataset.
  @param      gid           HDF5 internal object id 
  @param      loc           HDF5 internal parent node id 
  @param      datasetname   global dataset name 
  @return     integer   FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_MergeDatasetSingleFile(hid_t gid, hid_t loc, char *datasetname) {
    hid_t did, sid, tid, subset;
    int i;
    hsize_t n;
    char subsetname[FTI_BUFS];

    void* data;

    int datasetrank = FTI_GetDatasetRankFlush(gid);
    hsize_t* globaldimension = talloc(hsize_t, datasetrank);
    hsize_t* offset = talloc(hsize_t, datasetrank);
    hsize_t* count = talloc(hsize_t, datasetrank);

    FTI_ReadAttributeHDF5(gid, "global_dimension", globaldimension);

    sid = H5Screate_simple(datasetrank, globaldimension, NULL);

    free(globaldimension);

    tid = FTI_GetDatasetTypeFlush(gid);
    hid_t dcplid = H5Pcreate(H5P_DATASET_CREATE);

    if (H5Lexists(loc, datasetname, H5P_DEFAULT)) {
        did = H5Dopen(loc, datasetname, H5P_DEFAULT);
    } else {
        did = H5Dcreate(loc, datasetname, tid, sid, H5P_DEFAULT, dcplid,
         H5P_DEFAULT);
    }
    if (did < 0) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS, "Unable to access global dataset '%s'",
         datasetname);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }

    H5Pclose(dcplid);
    hid_t plid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);

    H5Gget_num_objs(gid, &n);
    for (i = 0; i < n; i++) {
        H5Gget_objname_by_idx(gid, i, subsetname, FTI_BUFS);
        subset = H5Dopen(gid, subsetname, H5P_DEFAULT);
        if (subset < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to access subset '%s' of global"
            " dataset '%s'", subsetname, datasetname);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        size_t typesize = H5Tget_size(tid);
        FTI_ReadAttributeHDF5(subset, "offset", offset);
        FTI_ReadAttributeHDF5(subset, "count", count);
        hid_t msid = H5Screate_simple(datasetrank, count, NULL);
        H5Sselect_hyperslab(sid, H5S_SELECT_SET, offset, NULL, count, NULL);

        size_t memsize = typesize*H5Sget_simple_extent_npoints(msid);
        data = malloc(memsize);
        if (!data) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to allocate %lu bytes to merge"
            " subset '%s' to global dataset '%s'", memsize,
             subsetname, datasetname);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        herr_t err = H5Dread(subset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        if (err) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to read from subset '%s' of"
            " global dataset '%s'", subsetname, datasetname);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        err = H5Dwrite(did, tid, msid, sid, plid, data);
        if (err) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to write subset '%s' to global"
            " dataset '%s'", subsetname, datasetname);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        H5Sclose(msid);
        H5Dclose(subset);
        free(data);
    }

    H5Pclose(plid);
    H5Sclose(sid);
    H5Tclose(tid);
    H5Dclose(did);

    free(count);
    free(offset);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      This function copies the group structure to global single file.
  @param      orig           HDF5 internal local object id 
  @param      copy           HDF5 internal global object id 
  @return     integer   FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_MergeObjectsSingleFile(hid_t orig, hid_t copy) {
    hsize_t n, na;
    int h5objtype;
    hid_t childorig;
    hid_t childcopy;
    char groupname[FTI_BUFS];
    char childname[FTI_BUFS];

    H5Iget_name(orig, groupname, FTI_BUFS);

    H5Gget_num_objs(orig, &n);
    int i = 0; for (; i < n; i++) {
        H5Gget_objname_by_idx(orig, i, childname, FTI_BUFS);
        h5objtype = H5Gget_objtype_by_idx(orig, i);

        switch (h5objtype) {
            case H5G_GROUP:
                childorig = H5Gopen(orig, childname, H5P_DEFAULT);
                if (childorig < 0) {
                    char errstr[FTI_BUFS];
                    snprintf(errstr, FTI_BUFS, "Unable to open '%s'",
                     childname);
                    FTI_Print(errstr, FTI_EROR);
                    return FTI_NSCS;
                }
                na = H5Aget_num_attrs(childorig);
                if (na > 0) {
                    FTI_MergeDatasetSingleFile(childorig, copy, childname);
                } else {
                    if (H5Lexists(copy, childname, H5P_DEFAULT)) {
                        childcopy = H5Gopen(copy, childname, H5P_DEFAULT);
                        if (childcopy < 0) {
                            char errstr[FTI_BUFS];
                            snprintf(errstr, FTI_BUFS, "Unable to open '%s'",
                             childname);
                            FTI_Print(errstr, FTI_EROR);
                            return FTI_NSCS;
                        }
                    } else {
                        childcopy = H5Gcreate(copy, childname, 0, H5P_DEFAULT,
                         H5P_DEFAULT);
                        if (childcopy < 0) {
                            char errstr[FTI_BUFS];
                            snprintf(errstr, FTI_BUFS, "Unable to create '%s'",
                             childname);
                            FTI_Print(errstr, FTI_EROR);
                            return FTI_NSCS;
                        }
                    }
                    FTI_MergeObjectsSingleFile(childorig, childcopy);
                    H5Gclose(childcopy);
                }
                H5Gclose(childorig);
                break;
        }
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      This function merges preprocessed checkpoint files to global
  single file on PFS.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Topo        Topology metadata.
  @return     integer   FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FlushH5SingleFile(FTIT_execution* FTI_Exec,
 FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo) {
    char fn[FTI_BUFS], tmpfn[FTI_BUFS], lfn[FTI_BUFS];
    hid_t fid, lfid;
    snprintf(tmpfn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->gTmpDir,
     FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptMeta.ckptId);
    snprintf(fn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->h5SingleFileDir,
     FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptMeta.ckptId);
    if (FTI_Topo->splitRank == 0) {
        MKDIR(FTI_Conf->gTmpDir, 0777);
    }
    MPI_Barrier(FTI_COMM_WORLD);

    // NO IMPROVEMENT IN PERFORMANCE OBSERVED USING HINTS HERE
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_write", "enable");
    MPI_Info_set(info, "stripping_unit", "4194304");
    hid_t plid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plid, FTI_COMM_WORLD, info);
    fid = H5Fcreate(tmpfn, H5F_ACC_TRUNC, H5P_DEFAULT, plid);
    if (fid < 0) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS, "Unable to create '%s'", tmpfn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    H5Pclose(plid);

    int b;
    for (b = 0; b < FTI_Topo->nbApprocs; b++) {
        snprintf(lfn, FTI_BUFS, "%s/Ckpt%d-Rank%d.h5", FTI_Conf->lTmpDir,
         FTI_Exec->ckptMeta.ckptId, FTI_Topo->body[b]);

        lfid = H5Fopen(lfn, H5F_ACC_RDWR, H5P_DEFAULT);
        if (lfid < 0) {
            char errstr[FTI_BUFS];
            snprintf(errstr, FTI_BUFS, "Unable to open '%s'", lfn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        FTI_MergeObjectsSingleFile(lfid, fid);

        H5Fclose(lfid);
    }

    H5Fclose(fid);

    if (FTI_Topo->splitRank == 0) {
        int status = rename(tmpfn, fn);
        remove(FTI_Conf->gTmpDir);
        bool removeLastFile = !FTI_Conf->h5SingleFileKeep &&
         (bool)strcmp(FTI_Exec->h5SingleFileLast, "");
        if (removeLastFile) {
            status = remove(FTI_Exec->h5SingleFileLast);
            if ((status != ENOENT) && (status != 0)) {
                char errstr[FTI_BUFS];
                snprintf(errstr, FTI_BUFS,
                 "failed to remove last VPR file '%s'",
                  FTI_Exec->h5SingleFileLast);
                FTI_Print(errstr, FTI_EROR);
            } else {
                status = FTI_SCES;
            }
        }
        if (status == FTI_SCES) {
            snprintf(FTI_Exec->h5SingleFileLast, FTI_BUFS, "%s/%s-ID%08d.h5",
             FTI_Conf->h5SingleFileDir,
                    FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptMeta.ckptId);
        }
    }


    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      This function finalized the single file checkpoint
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer   FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeH5SingleFile(FTIT_execution* FTI_Exec,
     FTIT_configuration* FTI_Conf,  FTIT_topology* FTI_Topo,
     FTIT_checkpoint* FTI_Ckpt, double t) {
    char str[FTI_BUFS];
    snprintf(str, sizeof(str), "Ckpt. ID %d (Variate Processor Recovery File)"
        " (%.2f MB/proc) taken in %.2f sec.", FTI_Exec->ckptMeta.ckptId,
        FTI_Exec->ckptSize / (1024.0 * 1024.0), t);
    FTI_Print(str, FTI_INFO);
    if (!FTI_Conf->h5SingleFileIsInline) {
        FTI_Exec->activateHeads(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt,
         FTI_SCES);
    } else {
        char tmpfn[FTI_BUFS];
        char fn[FTI_BUFS];
        snprintf(tmpfn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->gTmpDir,
         FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptMeta.ckptId);
        snprintf(fn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->h5SingleFileDir,
         FTI_Conf->h5SingleFilePrefix, FTI_Exec->ckptMeta.ckptId);
        if (FTI_Topo->splitRank == 0) {
            if (rename(tmpfn, fn) == 0) {
                if (rmdir(FTI_Conf->gTmpDir) < 0) {
                    FTI_Print("cannot remove global temp directory!", FTI_EROR);
                    return FTI_NSCS;
                }
            } else {
                FTI_Print("unable to rename VPR file!", FTI_EROR);
                return FTI_NSCS;
            }
        }
    }
    return FTI_SCES;
}
#endif
