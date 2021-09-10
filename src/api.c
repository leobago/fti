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
 *  @file   api.c
 *  @date   October, 2017
 *  @brief  API functions for the FTI library.
 */

#include "./interface.h"
#include "IO/cuda-md5/md5Opt.h"

#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>
#endif

/** General configuration information used by FTI.                         */
static FTIT_configuration FTI_Conf;

/** Checkpoint information for all levels of checkpoint.                   */
static FTIT_checkpoint FTI_Ckpt[5];

/** Dynamic information for this execution.                                */
static FTIT_execution FTI_Exec;

/** Topology of the system.                                                */
static FTIT_topology FTI_Topo;

/** id map that holds metadata for protected datasets                      */
static FTIT_keymap* FTI_Data;

/** SDC injection model and all the required information.                  */
static FTIT_injection FTI_Inje;

/** MPI communicator that splits the global one into app and FTI appart.   */
MPI_Comm FTI_COMM_WORLD;

/** FTI data type for chars.                                               */
fti_id_t FTI_CHAR;
/** FTI data type for short integers.                                      */
fti_id_t FTI_SHRT;
/** FTI data type for integers.                                            */
fti_id_t FTI_INTG;
/** FTI data type for long integers.                                       */
fti_id_t FTI_LONG;
/** FTI data type for unsigned chars.                                      */
fti_id_t FTI_UCHR;
/** FTI data type for unsigned short integers.                             */
fti_id_t FTI_USHT;
/** FTI data type for unsigned integers.                                   */
fti_id_t FTI_UINT;
/** FTI data type for unsigned long integers.                              */
fti_id_t FTI_ULNG;
/** FTI data type for single floating point.                               */
fti_id_t FTI_SFLT;
/** FTI data type for double floating point.                               */
fti_id_t FTI_DBLE;
/** FTI data type for long doble floating point.                           */
fti_id_t FTI_LDBE;

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes FTI.
  @param      configFile      FTI configuration file.
  @param      globalComm      Main MPI communicator of the application.
  @return     integer         FTI_SCES if successful.

  This function initializes the FTI context and prepares the heads to wait
  for checkpoints. FTI processes should never get out of this function. In
  case of a restart, checkpoint files should be recovered and in place at the
  end of this function.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Init(const char* configFile, MPI_Comm globalComm) {
#ifdef ENABLE_FTI_FI_IO
    FTI_InitFIIO();
#endif
#ifdef ENABLE_HDF5
    H5Eset_auto2(0, 0, NULL);
#endif
    FTI_InitExecVars(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, &FTI_Inje);
    FTI_Exec.globalComm = globalComm;
    MPI_Comm_rank(FTI_Exec.globalComm, &FTI_Topo.myRank);
    MPI_Comm_size(FTI_Exec.globalComm, &FTI_Topo.nbProc);
    snprintf(FTI_Conf.cfgFile, FTI_BUFS, "%s", configFile);
    FTI_Conf.verbosity = 1;  // Temporary needed for output in FTI_LoadConf.
    FTI_Exec.initSCES = 0;
    FTI_Inje.timer = MPI_Wtime();
    FTI_COMM_WORLD = globalComm;  // Temporary before building topology.
                                  // Needed in FTI_LoadConf and FTI_Topology
                                  // to communicate.
    FTI_Topo.splitRank = FTI_Topo.myRank;  // Temporary before building
                                           // topology. Needed in FTI_Print.
    int res = FTI_Try(FTI_LoadConf(&FTI_Conf, &FTI_Exec, &FTI_Topo,
      FTI_Ckpt, &FTI_Inje), "load configuration.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    res = FTI_Try(FTI_Topology(&FTI_Conf, &FTI_Exec, &FTI_Topo),
      "build topology.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    FTI_Try(FTI_InitGroupsAndTypes(&FTI_Exec),
      "malloc arrays for groups and types.");
    if (FTI_Topo.myRank == 0) {
        int restart = (FTI_Exec.reco != 3) ? FTI_Exec.reco : 0;
        FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, restart),
          "update configuration file.");
    }
    MPI_Barrier(FTI_Exec.globalComm);  // wait for myRank == 0
                                       // process to save config file
    if (FTI_Conf.ioMode == FTI_IO_FTIFF) {
        FTIFF_InitMpiTypes();
    }
    if (FTI_Conf.stagingEnabled) {
        FTI_InitStage(&FTI_Exec, &FTI_Conf, &FTI_Topo);
    }

    if (FTI_Conf.ioMode == FTI_IO_HDF5) {
        // strcpy(FTI_Conf.suffix, "h5");
        snprintf(FTI_Conf.suffix, sizeof(FTI_Conf.suffix), "h5");
    } else {
        // strcpy(FTI_Conf.suffix, "fti");
        snprintf(FTI_Conf.suffix, sizeof(FTI_Conf.suffix), "fti");
    }

    FTI_KeyMap(&FTI_Data, sizeof(FTIT_dataset), FTI_Conf.maxVarId, true);

    FTI_Exec.initSCES = 1;

    // init metadata queue
    FTI_MetadataQueue(&FTI_Exec.mqueue);

    if (FTI_Topo.amIaHead) {  // If I am a FTI dedicated process
        if (FTI_Exec.reco) {
            res = FTI_Try(FTI_RecoverFiles(&FTI_Conf, &FTI_Exec, &FTI_Topo,
             FTI_Ckpt), "recover the checkpoint files.");
            if (res != FTI_SCES) {
                FTI_Exec.reco = 0;
                FTI_Exec.initSCES = 2;  // Could not recover all ckpt files
            }
        }
        FTI_Listen(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt);
        // infinite loop inside, can stop only by calling FTI_Finalize
        // FTI_Listen only returns if FTI_Conf.keepHeadsAlive is TRUE
        return FTI_HEAD;
    } else {  // If I am an application process
        if (FTI_Try(FTI_InitDevices(FTI_Conf.cHostBufSize),
          "Allocating resources for communication with the devices") !=
           FTI_SCES) {
            FTI_Print("Cannot Allocate defice memory\n", FTI_EROR);
        }
        if (FTI_Try(FTI_InitFunctionPointers(FTI_Conf.ioMode, &FTI_Exec),
          "Initializing IO pointers") != FTI_SCES) {
            FTI_Print("Cannot define the function pointers\n", FTI_EROR);
        }

        // call in any case. treatment for diffCkpt disabled inside initializer
        if (FTI_Conf.dcpFtiff) {
            FTI_InitDcp(&FTI_Conf, &FTI_Exec);
        }
        if (FTI_Conf.dcpPosix) {
            FTI_initMD5(FTI_Conf.dcpInfoPosix.BlockSize, 32*1024*1024,
              &FTI_Conf);
        }
        if (FTI_Exec.reco) {
            res = FTI_Try(FTI_RecoverFiles(&FTI_Conf, &FTI_Exec,
              &FTI_Topo, FTI_Ckpt), "recover the checkpoint files.");
            if (FTI_Conf.ioMode == FTI_IO_FTIFF && res == FTI_SCES) {
                res += FTI_Try(FTIFF_ReadDbFTIFF(&FTI_Conf, &FTI_Exec,
                 FTI_Ckpt, FTI_Data), "Read FTIFF meta information");
            }
            FTI_Exec.ckptCnt = FTI_Exec.ckptId;
            FTI_Exec.ckptCnt++;
            if (res != FTI_SCES) {
                FTI_Exec.reco = 0;
                FTI_Exec.initSCES = 2;  // Could not recover all ckpt files
                                        // (or failed reading meta; FTI-FF)
                FTI_Print("FTI has been initialized.", FTI_INFO);
                return FTI_NREC;
            }
            FTI_Exec.hasCkpt = (FTI_Exec.reco == 3) ? false : true;
            if (FTI_Exec.reco != 3) FTI_Try(FTI_LoadMetaDataset(&FTI_Conf,
              &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data),
              "load dataset metadata");
        }
        FTI_Print("FTI has been initialized.", FTI_INFO);
        return FTI_SCES;
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It returns the current status of the recovery flag.
  @return     integer         FTI_Exec.reco.

  This function returns the current status of the recovery flag.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Status() {
    return FTI_Exec.reco;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Registers a new data type in FTI runtime
  @param      size             The data type size in bytes.
  @return     fti_id_t         A handle to represent the new type.

  This function initalizes a data type.
  The type is treated as a black box for FTI.
  Thus, the runtime only requires information about its size.
  Types built this was are saved as byte array when using HDF5 format.

  @todo This function should replace FTI_InitType for a cleaner API.
  This function is the result of a data type refactoring in FTI 1.4.
  It is obscured from the user for API backwards compatibility.
**/
/*-------------------------------------------------------------------------*/
fti_id_t FTI_InitType_opaque(size_t size) {
    FTIT_Datatype *type;
    fti_id_t new_id = FTI_Exec.datatypes.ntypes;

    // Sanity Check
    if (!size) {
        FTI_Print("Types must have positive size", FTI_WARN);
        return FTI_NSCS;
    }
    if (new_id >= TYPES_MAX) {
        FTI_Print("Maximum number of datatypes reached", FTI_WARN);
        return FTI_NSCS;
    }
    // Type initialization
    type = &FTI_Exec.datatypes.types[new_id];
    type->id = new_id;
    type->size = size;
    type->structure = NULL;
#ifdef ENABLE_HDF5
    type->h5group = FTI_Exec.H5groups[0];
    type->h5datatype = -1;  // to mark as closed
#endif
    // Global structure update
    FTI_Exec.datatypes.ntypes++;
    return type->id;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Registers a new data type in FTI runtime
  @param      type             Output parameter for the data type handle.
  @param      size             The data type size in bytes.
  @return     int              FTI_SCES on sucess, otherwise FTI_NSCS.

  This function initalizes a data type.
  The type is treated as a black box for FTI.
  Thus, the runtime only requires information about its size.
  Types built this was are saved as byte array when using HDF5 format.
**/
/*-------------------------------------------------------------------------*/
int FTI_InitType(fti_id_t* type, int size) {
    *type = FTI_InitType_opaque(size);
    return ((*type) != FTI_NSCS)? FTI_SCES : FTI_NSCS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Obtains the FTIT_type associated to a given type handle
  @param      handle         The data type handle
  @return     FTIT_type      The internal representation of an FTI type

  Returns NULL if the handle is not associated to an initialized type.
  This function should not be used directly.
  It is meant for advanced users in need to inspect FTI data structures. 
**/
/*-------------------------------------------------------------------------*/
FTIT_Datatype* FTI_GetType(fti_id_t id) {
    if (id < 0 || id >= FTI_Exec.datatypes.ntypes)
      return NULL;
    return &FTI_Exec.datatypes.types[id];
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes an empty composite data type.
  @param      name            An optional type name
  @param      size            The total size of the complex data type
  @param      h5g             An optional H5 group identifier
  @return     fti_id_t        A handle to represent the new type.

  Creates a composite data type that can contains other data types as fields.
  The fields can be added using FTI_AddScalarField and FTI_AddVectorField.
**/
/*-------------------------------------------------------------------------*/
fti_id_t FTI_InitCompositeType(const char* name, size_t size,
 FTIT_H5Group* h5g) {
    FTIT_Datatype *type;
    FTIT_complexType *structure;
    int type_id;

    // Sanity check
    TRY_ALLOC(structure, FTIT_complexType, 1) {
        FTI_Print("Failed to allocate complex type data", FTI_WARN);
        return FTI_NSCS;
    }
    // Simple type initialization
    type_id = FTI_InitType_opaque(size);
    type = FTI_GetType(type_id);
    if (!type) {
        free(structure);
        FTI_Print("Failed to initialize complex type", FTI_WARN);
        return FTI_NSCS;
    }
    // Complex type initialization
    if (h5g)
        type->h5group = FTI_Exec.H5groups[h5g->id];
    type->structure = structure;
    FTI_CopyStringOrDefault(type->structure->name, name, "Type%d", type_id);
    return type_id;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Adds a scalar field to a composite type.
  @param      id              The composite data type handle
  @param      name            An optional field name
  @param      fid             The field data type handle
  @param      offset          Offset of the field (use offsetof)
  @return     integer         FTI_SCES when successful, FTI_NSCS otherwise

  Adds a scalar field to a complex data type at a given offset.
  @warning
  Do note that FTI does not check for memory boundaries within the data type.
  Specifying a wrong offset leads to undefined behavior.
  This can be avoided using the offsetof() macro.

 **/
/*-------------------------------------------------------------------------*/
int FTI_AddScalarField(fti_id_t id, const char* name, fti_id_t fid,
 int64_t offset) {
    FTIT_Datatype *struct_ref, *field_type;
    int field_id;
    FTIT_typeField *field;

    // Sanity Checks
    struct_ref = FTI_GetCompositeType(id);
    field_type = FTI_GetType(fid);
    if (!struct_ref) {
        FTI_Print(
          "Composite id invalid when attempting to add a field", FTI_WARN);
        return FTI_NSCS;
    }
    if (!field_type) {
        FTI_Print(
          "Composite field id invalid when attempting to add a field",
          FTI_WARN);
        return FTI_NSCS;
    }
    field_id = struct_ref->structure->length;
    if (field_id > TYPES_FIELDS_MAX) {
        FTI_Print(
          "Composite must contain at most " STR(TYPES_FIELDS_MAX) "fields.",
          FTI_WARN);
        return FTI_NSCS;
    }
    // Field Initialization
    field = &struct_ref->structure->field[field_id];
    field->id = field_id;
    field->type = field_type;
    field->offset = offset;
    FTI_CopyStringOrDefault(field->name, name, "T%d", id);
    field->rank = 1;
    field->dimLength[0] = 1;
    // Structure update
    struct_ref->structure->length++;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Adds an n-dimensional vector field to a composite data type.
  @param      id              The composite data type handle
  @param      name            The field name
  @param      fid             The field data type handle
  @param      offset          Offset of the field (use offsetof)
  @param      ndims           The number of dimensions for the field
  @param      dim_size        Array of lengths for each dimension
  @return     integer         FTI_SCES when successful, FTI_NSCS otherwise

  Adds an N-dimensional array field to a complex data type at a given offset.
  @warning
  Do note that FTI does not check for memory boundaries within the data type.
  Specifying a wrong offset leads to undefined behavior.
  This can be avoided using the offsetof() macro.

 **/
/*-------------------------------------------------------------------------*/
int FTI_AddVectorField(fti_id_t id, const char* name,
  fti_id_t tid, int64_t offset, int ndims, int* dim_sizes) {
    FTIT_complexType *type;
    FTIT_typeField *field;
    int i;

    // Sanity Check
    if (dim_sizes == NULL) {
        FTI_Print(
          "Complex type field dimension size pointer cannot be NULL.",
          FTI_WARN);
        return FTI_NSCS;
    }
    if (ndims < 1 || ndims > TYPES_DIMENSION_MAX) {
        FTI_Print(
          "Complex type field must have between 1 and "
          STR(TYPES_DIMENSION_MAX) " dimensions",
          FTI_WARN);
        return FTI_NSCS;
    }
    for (i = 0; i < ndims; i++) {
        if (dim_sizes[i] < 1) {
            FTI_Print(
              "Complex type must have positive dimension sizes.",
              FTI_WARN);
            return FTI_NSCS;
        }
    }
    // Simple Field initialization
    if (FTI_AddScalarField(id, name, tid, offset) == FTI_NSCS) {
        FTI_Print("Failed to initialize complex field data.", FTI_WARN);
        return FTI_NSCS;
    }
    type = FTI_GetCompositeType(id)->structure;
    field = &type->field[type->length-1];  // Length was inc by AddScalarField
    // Composite Field initialization
    field->rank = ndims;
    memcpy(field->dimLength, dim_sizes, ndims*sizeof(int));
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Places the FTI staging directory path into 'stageDir'.
  @param      stageDir        pointer to allocated memory region.
  @param      maxLen          size of allocated memory region in bytes.
  @return     integer         FTI_SCES if successful, FTI_NSCS else.

  This function places the FTI staging directory path in 'stageDir'. If
  allocation size is not sufficiant, no action is perfoprmed and
  FTI_NSCS is returned.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetStageDir(char* stageDir, int maxLen) {
    if (!FTI_Conf.stagingEnabled) {
        FTI_Print("'FTI_GetStageDir' -> Staging disabled,"
          " no action performed.", FTI_WARN);
        return FTI_NSCS;
    }

    if (stageDir == NULL) {
        FTI_Print("invalid value for stageDir ('nil')!", FTI_WARN);
        return FTI_NSCS;
    }

    if (maxLen < 1) {
        char errstr[FTI_BUFS];
        snprintf(errstr, FTI_BUFS, "invalid value for maxLen ('%d')!", maxLen);
        FTI_Print(errstr, FTI_WARN);
        return FTI_NSCS;
    }

    int len = strlen(FTI_Conf.stageDir);
    if (maxLen < len+1) {
        FTI_Print("insufficient buffer size (maxLen too small)!", FTI_WARN);
        return FTI_NSCS;
    }

    strncpy(stageDir, FTI_Conf.stageDir, FTI_BUFS);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns status of staging request.
  @param      ID            ID of staging request.
  @return     integer       Status of staging request on success, 
  FTI_NSCS else.

  This function returns the status of the staging request corresponding
  to ID. The ID is returned by the function 'FTI_SendFile'. The status
  may be one of the five possible statuses:

  @par
  FTI_SI_FAIL - Stage request failed
  FTI_SI_SCES - Stage request succeed
  FTI_SI_ACTV - Stage request is currently processed
  FTI_SI_PEND - Stage request is pending
  FTI_SI_NINI - There is no stage request with this ID

  @note If the status is FTI_SI_NINI, the ID is either invalid or the
  request was finished (succeeded or failed). In the latter case,
  'FTI_GetStageStatus' returns FTI_SI_FAIL or FTI_SI_SCES and frees the
  stage request ressources. In the consecutive call it will then return
  FTI_SI_NINI.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetStageStatus(int ID) {
    if (!FTI_Conf.stagingEnabled) {
        FTI_Print("'FTI_GetStageStatus' -> Staging disabled,"
          " no action performed.", FTI_WARN);
        return FTI_NSCS;
    }

    // indicator if we still need the request structure allocated
    // (i.e. send buffer not released by MPI)
    bool free_req = true;

    // get status of request
    int status;
    status = FTI_GetStatusField(&FTI_Exec, &FTI_Topo, ID, FTI_SIF_VAL,
     FTI_Topo.nodeRank);

    // check if pending
    if (status == FTI_SI_PEND) {
        int flag = 1, idx;
        // if pending check if we can free the send buffer
        if ((idx = FTI_GetRequestIdx(ID)) >= 0) {
            MPI_Test(&(FTI_SI_APTR(FTI_Exec.stageInfo->request)[idx].mpiReq),
             &flag, MPI_STATUS_IGNORE);
        }
        if (flag == 0) {
            free_req = false;
        }
    }

    if (free_req) {
        FTI_FreeStageRequest(&FTI_Exec, &FTI_Topo, ID, FTI_Topo.nodeRank);
    }

    if ((status == FTI_SI_FAIL) || (status == FTI_SI_SCES)) {
        FTI_SetStatusField(&FTI_Exec, &FTI_Topo, ID, FTI_SI_NINI, FTI_SIF_VAL,
         FTI_Topo.nodeRank);
        FTI_SetStatusField(&FTI_Exec, &FTI_Topo, ID, FTI_SI_IAVL, FTI_SIF_AVL,
         FTI_Topo.nodeRank);
    }

    return status;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Copies file asynchronously from 'lpath' to 'rpath'.
  @param      lpath           absolute path local file.
  @param      rpath           absolute path remote file.
  @return     integer         Request handle (ID) on success, FTI_NSCS else.

  This function may be used to copy a file local on the nodes via the
  FTI head process asynchronously to the PFS. The file will not be
  removed after successful transfer, however, if stored in the directory
  returned by 'FTI_GetStageDir' it will be removed during
  'FTI_Finalize'.

  @par
  If staging is enabled but no head process, the staging will be
  performed synchronously (i.e. by the calling rank).
 **/
/*-------------------------------------------------------------------------*/
int FTI_SendFile(const char* lpath, const char *rpath) {
    if (!FTI_Conf.stagingEnabled) {
        FTI_Print("'FTI_SendFile' -> Staging disabled, no action performed.",
         FTI_WARN);
        return FTI_NSCS;
    }

    int ID = FTI_NSCS;

    // discard if path is NULL
    if (lpath == NULL) {
        FTI_Print("local path field is NULL!", FTI_WARN);
        return FTI_NSCS;
    }

    if (rpath == NULL) {
        FTI_Print("remote path field is NULL!", FTI_WARN);
        return FTI_NSCS;
    }

    // asign new request ID
    // note: if ID found, FTI_Exec->stageInfo->status[ID]
    // is set to not available
    int reqID = FTI_GetRequestID(&FTI_Exec, &FTI_Topo);
    if (reqID < 0) {
        FTI_Print("Too many stage requests!", FTI_WARN);
        return FTI_NSCS;
    }
    ID = reqID;

    FTI_InitStageRequestApp(&FTI_Exec, &FTI_Topo, ID);

    if (FTI_Topo.nbHeads == 0) {
        if (FTI_SyncStage(lpath, rpath, &FTI_Exec, &FTI_Topo,
          &FTI_Conf, ID) != FTI_SCES) {
            FTI_Print("synchronous staging failed!", FTI_WARN);
            return FTI_NSCS;
        }
    }

    if (FTI_Topo.nbHeads > 0) {
        if (FTI_AsyncStage(lpath, rpath, &FTI_Conf, &FTI_Exec,
         &FTI_Topo, ID) != FTI_SCES) {
            FTI_Print("asynchronous staging failed!", FTI_WARN);
            return FTI_NSCS;
        }
    }
    return ID;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It initialize a HDF5 group
  @param      h5group         H5 group that we want to initialize
  @param      name            Name of the H5 group
  @param      parent          Parent H5 group
  @return     integer         FTI_SCES if successful.

  Initialize group defined by user. If parent is NULL this mean parent will
  be set to root group.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitGroup(FTIT_H5Group* h5group, const char* name,
 FTIT_H5Group* parent) {
    if (parent == NULL) {
        // child of root
        parent = FTI_Exec.H5groups[0];
    }
    FTIT_H5Group* parentInArray = FTI_Exec.H5groups[parent->id];
    // check if this parent has that child
    int i;
    for (i = 0; i < parentInArray->childrenNo; i++) {
        if (strcmp(FTI_Exec.H5groups[parentInArray->childrenID[i]]->name,
         name) == 0) {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS, "Group %s already has the %s child.",
             parentInArray->name, name);
            return FTI_NSCS;
        }
    }
    h5group->id = FTI_Exec.nbGroup;
    h5group->childrenNo = 0;
    strncpy(h5group->name, name, FTI_BUFS);
#ifdef ENABLE_HDF5
    h5group->h5groupID = -1;  // to mark as closed
#endif

    // set full path to group
    snprintf(h5group->fullName, FTI_BUFS, "%s/%s", parent->fullName,
     h5group->name);

    // make a clone of the group in case the user won't store pointer
    FTI_Exec.H5groups[FTI_Exec.nbGroup] = malloc(sizeof(FTIT_H5Group));
    *FTI_Exec.H5groups[FTI_Exec.nbGroup] = *h5group;

    // assign a child and increment the childrenNo
    parentInArray->childrenID[parentInArray->childrenNo] = FTI_Exec.nbGroup;
    parentInArray->childrenNo++;

    FTI_Exec.nbGroup = FTI_Exec.nbGroup + 1;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Searches in the protected variables for a name. If not found it allocates and returns the ID 
  @param      name            Name of the protected variable to search 
  @return     integer         id of the variable.

  This function searches for a given name in the protected variables and returns the respective id for it.

 **/
/*-------------------------------------------------------------------------*/
int FTI_setIDFromString(const char *name) {
    int i = 0;

    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec.nbVar) != FTI_SCES) {
        FTI_Print("failed to set ID from string", FTI_WARN);
        return FTI_NSCS;
    }

    for (i = 0 ; i < FTI_Exec.nbVar; i++) {
        if (strcmp(name, data[i].idChar) == 0) {
            return data[i].id;
        }
    }

    // initialize blank dataset
    FTIT_dataset dataAdd;
    FTI_InitDataset(&FTI_Exec, &dataAdd, i);

    // set id to i+1 and assign name
    strncpy(dataAdd.idChar, name, FTI_BUFS);

    FTI_Data->push_back(&dataAdd, i);
    FTI_Exec.nbVar++;

    return i;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Searches in the protected variables for a name. If not found it allocates and returns the ID 
  @param      name            Name of the protected variable to search 
  @return     integer         id of the variable.

  This function searches for a given name in the protected variables and returns the respective id for it.

 **/
/*-------------------------------------------------------------------------*/
int FTI_getIDFromString(const char *name) {
    // after restart and before fully recovered, nbVarStored may be
    // larger than nbVar. In that case, the idchar may be in the recovered
    // set of protected variables.
    int n = (FTI_Exec.nbVarStored > FTI_Exec.nbVar) ?
     FTI_Exec.nbVarStored : FTI_Exec.nbVar;

    FTIT_dataset* data;
    if (FTI_Data->data(&data, n) != FTI_SCES) {
        FTI_Print("failed to get ID from string", FTI_WARN);
        return FTI_NSCS;
    }

    int i = 0; for (; i < n; i++) {
        if (strcmp(name, data[i].idChar) == 0) {
            return data[i].id;
        }
    }
    return -1;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Renames a HDF5 group
  @param      h5group         H5 group that we want to rename
  @param      name            New name of the H5 group
  @return     integer         FTI_SCES if successful.

  This function renames HDF5 group defined by user.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RenameGroup(FTIT_H5Group* h5group, const char* name) {
    strncpy(FTI_Exec.H5groups[h5group->id]->name, name, FTI_BUFS);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It sets/resets the pointer and type to a protected variable.
  @param      id              ID for searches and update.
  @param      ptr             Pointer to the data structure.
  @param      count           Number of elements in the data structure.
  @param      tid             The data type handle for the variable
  @return     integer         FTI_SCES if successful.

  This function stores a pointer to a data structure, its size, its ID,
  its number of elements and the type of the elements. This list of
  structures is the data that will be stored during a checkpoint and
  loaded during a recovery. It resets the pointer to a data structure,
  its size, its number of elements and the type of the elements if the
  dataset was already previously registered.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Protect(int id, void* ptr, int64_t count, fti_id_t tid) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    char str[5*FTI_BUFS];  // For console output
    // Id out of bounds.
    if (id > FTI_Conf.maxVarId) {
        snprintf(str, FTI_BUFS, "Id out of bounds ('Basic:max_var_id = %d').",
         FTI_Conf.maxVarId);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

#ifdef GPUSUPPORT
    FTIT_ptrinfo ptrInfo;
    int res;
    if ((res = FTI_Try( FTI_get_pointer_info((const void*) ptr, &ptrInfo),
     "FTI_Protect: determine pointer type")) != FTI_SCES)
        return res;
#endif

    char memLocation[4];

    FTIT_dataset* data;
    if (FTI_Data->get(&data, id) != FTI_SCES) {
        FTI_Print("failed to protect variable", FTI_WARN);
        return FTI_NSCS;
    }

    if (data != NULL) {  // Search for dataset with given id
        int64_t prevSize = data->size;
#ifdef GPUSUPPORT
        if (ptrInfo.type == FTIT_PTRTYPE_CPU) {
            // strcpy(memLocation, "CPU");
            snprintf(memLocation, sizeof(memLocation), "CPU");
            data->isDevicePtr = false;
            data->devicePtr = NULL;
            data->ptr = ptr;
        } else if (ptrInfo.type == FTIT_PTRTYPE_GPU) {
            // strcpy(memLocation, "GPU");
            snprintf(memLocation, sizeof(memLocation), "GPU");
            data->isDevicePtr = true;
            data->devicePtr = ptr;
            data->ptr = NULL;  // (void *) malloc (type.size *count);
        } else {
            FTI_Print("ptr Should be either a device"
            " location or a cpu location\n", FTI_EROR);
            data->ptr = NULL;  // (void *) malloc (type.size *count);
            return FTI_NSCS;
        }
#else
        // strcpy(memLocation, "CPU");
        snprintf(memLocation, sizeof(memLocation), "CPU");
        data->isDevicePtr = false;
        data->devicePtr = NULL;
        data->ptr = ptr;
#endif
        data->count = count;
        data->type = FTI_GetType(tid);
        if (data->type == NULL) {
          FTI_Print("Invalid data type handle on FTI_Protect.", FTI_WARN);
          return FTI_NSCS;
        }
        data->eleSize = data->type->size;
        data->size = data->type->size * count;
        if( prevSize != data->size ) { 
          if( tid == FTI_SFLT ) {
            data->ptr_cpy = (float*) realloc( data->ptr_cpy, data->size );
          } else if( tid == FTI_DBLE ) {
            data->ptr_cpy = (double*) realloc( data->ptr_cpy, data->size );
          }
        }
        data->dimLength[0] = count;
        FTI_Exec.ckptSize = FTI_Exec.ckptSize +
        ((data->type->size * count) - prevSize);
        if (strlen(data->idChar) == 0) {
            /*sprintf(str, "Variable ID %d reseted. (Stored In %s).  
            Current ckpt. size per rank is %.2fMB.", id, memLocation, 
            (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));*/
            snprintf(str, sizeof(str), "Variable ID %d reseted. "
              "(Stored In %s).  Current ckpt. size per rank is %.2fMB.",
              id, memLocation, (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
        } else {
            /*sprintf(str, "Variable Named %s with ID %d to protect 
            (Stored in %s). Current ckpt. size per rank is %.2fMB.", 
            data->idChar, id, memLocation, 
            (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));*/
            snprintf(str, sizeof(str), "Variable Named %s with ID %d to"
            " protect (Stored in %s). Current ckpt. size per rank is %.2fMB.",
             data->idChar, id, memLocation,
              (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
        }

        FTI_Print(str, FTI_DBUG);
        if (prevSize != data->size &&  FTI_Conf.dcpPosix) {
            if (!(data->isDevicePtr)) {
                int64_t nbHashes = data->size /
                FTI_Conf.dcpInfoPosix.BlockSize +
                 (bool)(data->size %FTI_Conf.dcpInfoPosix.BlockSize);
                data->dcpInfoPosix.currentHashArray = (unsigned char*)
                 realloc(data->dcpInfoPosix.currentHashArray,
                  sizeof(unsigned char)*nbHashes*
                  FTI_Conf.dcpInfoPosix.digestWidth);
                data->dcpInfoPosix.oldHashArray = (unsigned char*)
                 realloc(data->dcpInfoPosix.oldHashArray,
                  sizeof(unsigned char)*
                  nbHashes*FTI_Conf.dcpInfoPosix.digestWidth);
            }
#ifdef GPUSUPPORT
            else {
                unsigned char *x;
                int64_t nbNewHashes = data->size /
                FTI_Conf.dcpInfoPosix.BlockSize +
                 (bool)(data->size %FTI_Conf.dcpInfoPosix.BlockSize);
                int64_t nbOldHashes = prevSize /
                FTI_Conf.dcpInfoPosix.BlockSize +
                 (bool)(data->size %FTI_Conf.dcpInfoPosix.BlockSize);
                CUDA_ERROR_CHECK(cudaMallocManaged((void**) &x,
                 (nbNewHashes * FTI_Conf.dcpInfoPosix.digestWidth),
                  cudaMemAttachGlobal));
                memcpy(x, data->dcpInfoPosix.currentHashArray,
                 MIN(nbNewHashes * FTI_Conf.dcpInfoPosix.digestWidth,
                  nbOldHashes * FTI_Conf.dcpInfoPosix.digestWidth));
                CUDA_ERROR_CHECK(cudaFree(data->dcpInfoPosix.currentHashArray));
                data->dcpInfoPosix.currentHashArray = x;

                CUDA_ERROR_CHECK(cudaMallocManaged((void **)&x,
                 nbNewHashes * FTI_Conf.dcpInfoPosix.digestWidth,
                  cudaMemAttachGlobal));
                memcpy(x, data->dcpInfoPosix.oldHashArray,
                  MIN(nbNewHashes * FTI_Conf.dcpInfoPosix.digestWidth,
                   nbOldHashes * FTI_Conf.dcpInfoPosix.digestWidth));
                CUDA_ERROR_CHECK(cudaFree(data->dcpInfoPosix.oldHashArray));
                data->dcpInfoPosix.oldHashArray = x;
            }
#endif
        }
        if (data->recovered) {
            if (strlen(data->idChar) == 0) {
                /*sprintf(str, "Variable ID %d to protect (Stored in %s). 
                Current ckpt. size per rank is %.2fMB.",
                        id,
                        memLocation,
                        (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));*/
                snprintf(str, sizeof(str), "Variable ID %d to protect "
                  "(Stored in %s). Current ckpt. size per rank is %.2fMB.",
                        id,
                        memLocation,
                        (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
            } else {
                /*sprintf(str, "Variable Named %s with ID %d to protect (Stored
                 in %s). Current ckpt. size per rank is %.2fMB.",
                        data->idChar,
                        id, memLocation,
                        (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));*/
                snprintf(str, sizeof(str), "Variable Named %s with ID %d to "
                  "protect (Stored in %s). Current ckpt. size "
                  "per rank is %.2fMB.",
                        data->idChar,
                        id, memLocation,
                        (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
            }
            FTI_Print(str, FTI_INFO);
            FTI_Exec.nbVar++;
            data->recovered = false;
        }
        return FTI_SCES;
    }
    // Id could not be found in datasets

    data = calloc(1, sizeof(FTIT_dataset));

    // Adding new variable to protect
    data->id = id;
#ifdef GPUSUPPORT
    if (ptrInfo.type == FTIT_PTRTYPE_CPU) {
        // strcpy(memLocation, "CPU");
        snprintf(memLocation, sizeof(memLocation), "CPU");
        data->isDevicePtr = false;
        data->devicePtr = NULL;
        data->ptr = ptr;
    } else if (ptrInfo.type == FTIT_PTRTYPE_GPU) {
        // strcpy(memLocation, "GPU");
        snprintf(memLocation, sizeof(memLocation), "GPU");
        data->isDevicePtr = true;
        data->devicePtr = ptr;
        data->ptr = NULL;  // (void *) malloc (type.size *count);
    } else {
        FTI_Print("ptr Should be either a device location or a cpu location\n",
         FTI_EROR);
        data->ptr = NULL;  // (void *) malloc (type.size *count);
        return FTI_NSCS;
    }
#else
    // strcpy(memLocation, "CPU");
    snprintf(memLocation, sizeof(memLocation), "CPU");
    data->isDevicePtr = false;
    data->devicePtr = NULL;
    data->ptr = ptr;
#endif
    // Important assignment, we use realloc!
    data->sharedData.dataset = NULL;
    data->count = count;
    data->type = FTI_GetType(tid);
    if (data->type == NULL) {
        FTI_Print("Invalid data type handle on FTI_Protect.", FTI_WARN);
        return FTI_NSCS;
    }
    if( tid == FTI_SFLT ) {
      data->ptr_cpy = talloc( float, count );
    } else if( tid == FTI_DBLE ) {
      data->ptr_cpy = talloc( double, count );
    }
    data->eleSize = data->type->size;
    data->size = data->type->size * count;
    data->rank = 1;
    data->dimLength[0] = data->count;
    data->h5group = FTI_Exec.H5groups[0];
    // sprintf(data->name, "Dataset_%d", id);
    snprintf(data->name, sizeof(data->name), "Dataset_%d", id);
    FTI_Exec.ckptSize = FTI_Exec.ckptSize + (data->type->size * count);

    if (FTI_Conf.dcpPosix) {
        if (!(data->isDevicePtr)) {
            int64_t nbHashes = data->size /
            FTI_Conf.dcpInfoPosix.BlockSize
             + (bool)(data->size %FTI_Conf.dcpInfoPosix.BlockSize);
            data->dcpInfoPosix.hashDataSize = 0;
            data->dcpInfoPosix.currentHashArray = (unsigned char*)
             malloc( sizeof(unsigned char)*
              nbHashes*FTI_Conf.dcpInfoPosix.digestWidth);
            data->dcpInfoPosix.oldHashArray = (unsigned char*)
             malloc( sizeof(unsigned char)*
              nbHashes*FTI_Conf.dcpInfoPosix.digestWidth);
        }
#ifdef GPUSUPPORT
        else {
            unsigned char *x;
            int64_t nbNewHashes = data->size /
            FTI_Conf.dcpInfoPosix.BlockSize +
             (bool)(data->size %FTI_Conf.dcpInfoPosix.BlockSize);
            CUDA_ERROR_CHECK(cudaMallocManaged((void**)&x,
             nbNewHashes * FTI_Conf.dcpInfoPosix.digestWidth,
              cudaMemAttachGlobal));
            data->dcpInfoPosix.currentHashArray = x;
            CUDA_ERROR_CHECK(cudaMallocManaged((void**)&x,
             nbNewHashes * FTI_Conf.dcpInfoPosix.digestWidth,
              cudaMemAttachGlobal));
            data->dcpInfoPosix.oldHashArray = x;
        }
#endif
    }

    // append dataset to protected variables
    if (FTI_Data->push_back(data, id) != FTI_SCES) {
        snprintf(str, FTI_BUFS, "failed to append variable with id = '%d' to"
        " protected variable map.", id);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }

    if (strlen(data->idChar) == 0) {
        // sprintf(str, "Variable ID %d to protect (Stored in %s). Current ckpt
        // size per rank is %.2fMB.", id, memLocation,
        // (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
        snprintf(str, sizeof(str), "Variable ID %d to protect (Stored in %s)."
          " Current ckpt. size per rank is %.2fMB.", id, memLocation,
           (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
    } else {
        // sprintf(str, "Variable Named %s with ID %d to protect (Stored in %s)
        // Current ckpt. size per rank is %.2fMB.", data->idChar, id,
        // memLocation, (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
        snprintf(str, sizeof(str), "Variable Named %s with ID %d to protect"
        " (Stored in %s). Current ckpt. size per rank is %.2fMB.",
         data->idChar, id, memLocation,
          (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
    }

    FTI_Exec.nbVar = FTI_Exec.nbVar + 1;
    FTI_Print(str, FTI_INFO);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      it allows to add descriptive attributes to a protected variable 
  @param      id              ID of the variable.
  @param      attribute       structure that holds the attributes values.
  @param      flag            flag to indicate which attributes to set.
  @return     integer         FTI_SCES if successful.

  This function allows to set a descriptive attribute to a protected variable. 
  The variable has to be protected and an ID assigned before the call. The
  flag can consist of any combination of the following flags:
    FTI_ATTRIBUTE_NAME
    FTI_ATTRIBUTE_DIM
  flags can be combined by using the bitwise or operator. The attributes will
  appear inside the meta data files when a checkpoint is taken. When setting 
  the dimension of a dataset, the first dimension is the leading dimension, 
  i.e. the dimension that is stored contiguous inside a flat matrix 
  representation. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_SetAttribute(int id, FTIT_attribute attribute,
        FTIT_attributeFlag flag) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    FTIT_dataset* data;
    if (FTI_Data->get(&data, id) != FTI_SCES) {
        FTI_Print("failed to set attribute: could not query dataset", FTI_WARN);
        return FTI_NSCS;
    }

    if ( data == NULL ) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS,
                "failed to set attribute: dataset with id=%d does not exist",
                id);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    if ( (flag & FTI_ATTRIBUTE_NAME) == FTI_ATTRIBUTE_NAME ) {
        strncpy(data->attribute.name, attribute.name, FTI_BUFS);
    }

    if ( (flag & FTI_ATTRIBUTE_DIM) == FTI_ATTRIBUTE_DIM ) {
        data->attribute.dim = attribute.dim;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Defines a global dataset (shared among application processes)
  @param      id              ID of the dataset.
  @param      rank            Rank of the dataset.
  @param      dimLength       Dimention length for each rank.
  @param      name            Name of the dataset in HDF5 file.
  @param      h5group         Group of the dataset. If Null then "/".
  @param      tid             FTI Data type handler 
  @return     integer         FTI_SCES if successful.

  This function defines a global dataset which is shared among all ranks.
  In order to assign sub sets to the dataset the user has to call the
  function 'FTI_AddSubset'. The parameter 'did' of that function, corres-
  ponds to the global dataset id define here.

 **/
/*-------------------------------------------------------------------------*/
int FTI_DefineGlobalDataset(int id, int rank, FTIT_hsize_t* dimLength,
 const char* name, FTIT_H5Group* h5group, fti_id_t tid) {
#ifdef ENABLE_HDF5
    FTIT_globalDataset* last = FTI_Exec.globalDatasets;

    if (last) {
        FTIT_globalDataset* curr = last;
        while (curr) {
            if (id == last->id) {
                char str[FTI_BUFS];
                snprintf(str, FTI_BUFS, "FTI_DefineGlobalDataset :: id '%d' "
                  "is already taken.", id);
                FTI_Print(str, FTI_EROR);
                return FTI_NSCS;
            }
            last = curr;
            curr = last->next;
        }
        last->next = talloc(FTIT_globalDataset, 1);
        last = last->next;
    } else {
        last = talloc(FTIT_globalDataset, 1);
        FTI_Exec.globalDatasets = last;
    }

    last->id = id;
    last->initialized = false;
    last->rank = rank;
    last->hid = -1;
    last->fileSpace = -1;
    last->dimension = talloc(hsize_t, rank);
    int i;
    for (i = 0; i < rank; i++) {
        last->dimension[i] = dimLength[i];
    }
    strncpy(last->name, name, FTI_BUFS);
    last->name[FTI_BUFS-1] = '\0';
    last->numSubSets = 0;
    last->varId = NULL;
    last->type = FTI_GetType(tid);
    if (last->type == NULL) {
        FTI_Print("Invalid data type handle on FTI_DefineGlobalDataset.",
          FTI_WARN);
        return FTI_NSCS;
    }
    last->location = (h5group) ?
     FTI_Exec.H5groups[h5group->id] : FTI_Exec.H5groups[0];

    // safe path to dataset
    snprintf(last->fullName, FTI_BUFS, "%s/%s", last->location->fullName,
     last->name);

    last->next = NULL;

    return FTI_SCES;
#else
    FTI_Print("'FTI_DefineGlobalDataset' is an HDF5 feature. Please enable "
      "HDF5 and recompile.", FTI_WARN);
    return FTI_NSCS;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Assigns a FTI protected variable to a global dataset
  @param      id              Corresponding variable ID.
  @param      rank            Rank of the dataset.
  @param      offset          Starting coordinates in global dataset.
  @param      count           number of elements for each coordinate.
  @param      did             Corresponding global dataset ID.
  @return     integer         FTI_SCES if successful.

  This function assigns the protected dataset with ID 'id' to a global data-
  set with ID 'did'. The parameters 'offset' and 'count' specify the selec-
  tion of the sub-set inside the global dataset ('offset' and 'count' cor-
  respond to 'start' and 'count' in the HDF5 function 'H5Sselect_hyperslab'
  For questions on what they define, please consult the HDF5 documentation.)

 **/
/*-------------------------------------------------------------------------*/
int FTI_AddSubset(int id, int rank, FTIT_hsize_t* offset,
 FTIT_hsize_t* count, int did) {
#ifdef ENABLE_HDF5

    FTIT_dataset* data;
    if (FTI_Data->get(&data, id) != FTI_SCES) {
        FTI_Print("failed to add subset", FTI_WARN);
        return FTI_NSCS;
    }

#ifdef GPUSUPPORT
    if (!data->isDevicePtr) {
#endif

        bool found = false;

        FTIT_globalDataset* dataset = FTI_Exec.globalDatasets;
        while (dataset) {
            if (dataset->id == did) {
                found = true;
                break;
            }
            dataset = dataset->next;
        }

        if (!found) {
            FTI_Print("dataset id could not be found!", FTI_EROR);
            return FTI_NSCS;
        }

        if (dataset->rank != rank) {
            FTI_Print("rank missmatch!", FTI_EROR);
            return FTI_NSCS;
        }

        dataset->numSubSets++;
        dataset->varId = (int*) realloc(dataset->varId,
         dataset->numSubSets*sizeof(int));
        dataset->varId[dataset->numSubSets-1] = id;

        data->sharedData.dataset = dataset;
        data->sharedData.offset = talloc(hsize_t, rank);
        data->sharedData.count = talloc(hsize_t, rank);
        int i = 0; for (; i < rank; i++) {
            data->sharedData.offset[i] = offset[i];
            data->sharedData.count[i] = count[i];
        }

        return FTI_SCES;
#ifdef GPUSUPPORT
    } else {
        FTI_Print("Dataset is on GPU memory. VPR does not have GPU "
          "support yet!", FTI_WARN);
        return FTI_NSCS;
    }
#endif
#else
    FTI_Print("'FTI_AddSubset' is an HDF5 feature. Please enable HDF5 and"
    " recompile.", FTI_WARN);
    return FTI_NSCS;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Updates global dataset (shared among application processes)
  @param      id              ID of the dataset.
  @param      rank            Rank of the dataset.
  @param      dimLength       Dimention length for each rank.

  updates only the rank and number of elements for each coordinate 
  direction. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_UpdateGlobalDataset(int id, int rank, FTIT_hsize_t* dimLength) {
#ifdef ENABLE_HDF5
    FTIT_globalDataset* dataset = FTI_Exec.globalDatasets;

    if (!dataset) {
        FTI_Print("there are no global datasets defined!", FTI_WARN);
        return FTI_NSCS;
    }

    bool found = false;
    while (dataset) {
        if (id == dataset->id) {
            found = true;
            break;
        }
        dataset = dataset->next;
    }

    if (!found) {
        FTI_Print("invalid dataset id!", FTI_WARN);
        return FTI_NSCS;
    }

    dataset->rank = rank;
    dataset->dimension = (hsize_t*) realloc(dataset->dimension,
     sizeof(hsize_t) * rank);
    int i;
    for (i = 0; i < rank; i++) {
        dataset->dimension[i] = dimLength[i];
    }

    return FTI_SCES;
#else
    FTI_Print("'FTI_UpdateGlobalDataset' is an HDF5 feature. Please enable "
      "HDF5 and recompile.", FTI_WARN);
    return FTI_NSCS;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Updates a FTI protected variable of a global dataset
  @param      id              Corresponding variable ID.
  @param      rank            Rank of the dataset.
  @param      offset          Starting coordinates in global dataset.
  @param      count           number of elements for each coordinate.
  @param      did             Corresponding global dataset ID.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_UpdateSubset(int id, int rank, FTIT_hsize_t* offset,
 FTIT_hsize_t* count, int did) {
#ifdef ENABLE_HDF5

    FTIT_dataset* data;
    if (FTI_Data->get(&data, id) != FTI_SCES) {
        FTI_Print("failed to update subset", FTI_WARN);
        return FTI_NSCS;
    }

#ifdef GPUSUPPORT
    if (!data->isDevicePtr) {
#endif

        FTIT_globalDataset* dataset = FTI_Exec.globalDatasets;
        while (dataset) {
            if (dataset->id == did) {
                break;
            }
            dataset = dataset->next;
        }

        if (!dataset) {
            FTI_Print("dataset id could not be found!", FTI_EROR);
            return FTI_NSCS;
        }

        if (dataset->rank != rank) {
            FTI_Print("rank missmatch!", FTI_EROR);
            return FTI_NSCS;
        }

        int i = 0; for (; i < dataset->numSubSets; i++) {
            if (dataset->varId[i] == id) {
                break;
            }
        }

        if (i == dataset->numSubSets) {
            FTI_Print("variable is not subset of dataset!", FTI_WARN);
            return FTI_NSCS;
        }

        data->sharedData.offset = (hsize_t*) realloc(data->sharedData.offset,
         sizeof(hsize_t) * rank);
        data->sharedData.count = (hsize_t*) realloc(data->sharedData.count,
         sizeof(hsize_t) * rank);
        for (i = 0; i < rank; i++) {
            data->sharedData.offset[i] = offset[i];
            data->sharedData.count[i] = count[i];
        }

        return FTI_SCES;
#ifdef GPUSUPPORT
    } else {
        FTI_Print("Dataset is on GPU memory. VPR does not have GPU support"
        " yet!", FTI_WARN);
        return FTI_NSCS;
    }
#endif
#else
    FTI_Print("'FTI_AddSubset' is an HDF5 feature. Please enable HDF5 and"
    " recompile.", FTI_WARN);
    return FTI_NSCS;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      returns rank of shared dataset
  @param      id              ID of the dataset.
  @return     integer         rank of dataset.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetDatasetRank(int did) {
#ifdef ENABLE_HDF5

    FTIT_globalDataset * dataset = FTI_Exec.globalDatasets;
    if (!dataset) {
        FTI_Print("No datasets defined!", FTI_WARN);
        return FTI_NSCS;
    }

    while (dataset) {
        if (dataset->id == did) break;
        dataset = dataset->next;
    }

    if (!dataset) {
        FTI_Print("Failed to find dataset in list!", FTI_WARN);
        return FTI_NSCS;
    }

    return dataset->rank;

#else
    FTI_Print("'FTI_GetDatasetRank' is an HDF5 feature. Please enable HDF5 and"
    " recompile.", FTI_WARN);
    return FTI_NSCS;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      returns static array of dataset dimensions 
  @param      id              ID of the dataset.
  @param      rank            Rank of the dataset.

 **/
/*-------------------------------------------------------------------------*/
FTIT_hsize_t* FTI_GetDatasetSpan(int did, int rank) {
#ifdef ENABLE_HDF5

    static hsize_t span[FTI_HDF5_MAX_DIM];

    FTIT_globalDataset * dataset = FTI_Exec.globalDatasets;
    if (!dataset) {
        FTI_Print("No datasets defined!", FTI_WARN);
        return NULL;
    }

    while (dataset) {
        if (dataset->id == did) break;
        dataset = dataset->next;
    }

    if (!dataset) {
        FTI_Print("Failed to find dataset in list!", FTI_WARN);
        return NULL;
    }

    if (rank != dataset->rank) {
        FTI_Print("Dataset rank missmatch!", FTI_WARN);
        return NULL;
    }

    memcpy(span, dataset->dimension, rank*sizeof(hsize_t));

    return span;
#else
    FTI_Print("'FTI_GetDatasetSpan' is an HDF5 feature. Please enable HDF5 and"
    " recompile.", FTI_WARN);
    return NULL;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      loads dataset dimension from ckpt file to dataset 'did'
  @param      id              ID of the dataset.
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverDatasetDimension(int did) {
#ifdef ENABLE_HDF5

    if (FTI_Exec.reco != 3) {
        FTI_Print("this is no VPR recovery!", FTI_WARN);
        return FTI_NSCS;
    }
    FTIT_globalDataset * dataset = FTI_Exec.globalDatasets;
    if (!dataset) {
        FTI_Print("No datasets defined!", FTI_WARN);
        return FTI_NSCS;
    }

    while (dataset) {
        if (dataset->id == did) break;
        dataset = dataset->next;
    }

    if (!dataset) {
        FTI_Print("Failed to find dataset in list!", FTI_WARN);
        return FTI_NSCS;
    }

    // open HDF5 file
    hid_t plid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plid, FTI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file_id = H5Fopen(FTI_Exec.h5SingleFileReco, H5F_ACC_RDONLY, plid);
    H5Pclose(plid);

    // hid_t gid = H5Gopen1( file_id, dataset->location->name );
    hid_t dataset_id = H5Dopen(file_id, dataset->fullName, H5P_DEFAULT);

    int drank = FTI_GetDatasetRankReco(dataset_id);
    if (drank != dataset->rank) {
        FTI_Print("Rank missmatch!", FTI_WARN);
        return FTI_NSCS;
    }

    hsize_t *span = talloc(hsize_t, drank);

    int status = FTI_GetDatasetSpanReco(dataset_id, span);
    if (status != FTI_SCES) {
        FTI_Print("Failed to retrieve span!", FTI_WARN);
    }

    dataset->rank = drank;
    free(dataset->dimension);
    dataset->dimension = span;

    H5Dclose(did);
    H5Fclose(file_id);

    return FTI_SCES;

#else
    FTI_Print("'FTI_RecoverDatasetDimension' is an HDF5 feature. Please enable"
    " HDF5 and recompile.", FTI_WARN);
    return FTI_NSCS;
#endif
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Defines the dataset
  @param      id              ID for searches and update.
  @param      rank            Rank of the array
  @param      dimLength       Dimention length for each rank
  @param      name            Name of the dataset in HDF5 file.
  @param      h5group         Group of the dataset. If Null then "/"
  @return     integer         FTI_SCES if successful.

  This function gives FTI all information needed by HDF5 to correctly save
  the dataset in the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_DefineDataset(int id, int rank, int* dimLength, const char* name,
 FTIT_H5Group* h5group) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if (rank > 0 && dimLength == NULL) {
        FTI_Print("If rank > 0, the dimLength cannot be NULL.", FTI_WARN);
        return FTI_NSCS;
    }
    if (rank > 32) {
        FTI_Print("Maximum rank is 32.", FTI_WARN);
        return FTI_NSCS;
    }

    char str[FTI_BUFS];  // For console output

    FTIT_dataset* data;
    if (FTI_Data->get(&data, id) != FTI_SCES) {
        FTI_Print("failed to define dataset", FTI_WARN);
        return FTI_NSCS;
    }

    if (!data) {
        // sprintf(str, "The dataset #%d not
        // initialized. Use FTI_Protect first.", id);
        snprintf(str, sizeof(str), "The dataset #%d not initialized."
          " Use FTI_Protect first.", id);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    // check if size is correct
    int64_t expectedSize = 1;
    int j;
    for (j = 0; j < rank; j++) {
        expectedSize *= dimLength[j];  // compute the number of elements
    }

    if (rank > 0) {
        if (expectedSize != data->count) {
            // sprintf(str, "Trying to define datasize: number of elements %d,
            // but the dataset count is %ld.", expectedSize, data->count);
            snprintf(str, sizeof(str), "Trying to define datasize: number of"
            " elements %lu, but the dataset count is %lu.",
             expectedSize, data->count);
            FTI_Print(str, FTI_WARN);
            return FTI_NSCS;
        }
        data->rank = rank;
        for (j = 0; j < rank; j++) {
            data->dimLength[j] = dimLength[j];
        }
    }

    if (h5group != NULL) {
        data->h5group = FTI_Exec.H5groups[h5group->id];
    }

    if (name != NULL) {
        memset(data->name, '\0', FTI_BUFS);
        strncpy(data->name, name, FTI_BUFS);
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns size saved in metadata of variable
  @param      id              Variable ID.
  @return     int64_t            Returns size of variable or 0 if size not saved.

  This function returns size of variable of given ID that is saved in metadata.
  This may be different from size of variable that is in the program. If this
  function it's called when recovery it returns size from metadata file, if it's
  called after checkpoint it returns size saved in temporary metadata. If there
  is no size saved in metadata it returns 0.
 **/
/*-------------------------------------------------------------------------*/
int64_t FTI_GetStoredSize(int id) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return 0;
    }

    FTIT_dataset* data;
    if ((FTI_Data->get(&data, id) != FTI_SCES)) {
        FTI_Print("Unable to determine the stored variable size!", FTI_WARN);
        return 0;
    }

    if (!data) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "variable id='%d' does not exist,"
          " failed to get stored size", id);
        FTI_Print(str, FTI_WARN);
        return 0;
    }

    return data->sizeStored;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Reallocates dataset to last checkpoint size.
  @param      id              Variable ID.
  @param      ptr             Pointer to the variable.
  @return     ptr             Pointer if successful, NULL otherwise
  This function loads the checkpoint data size from the metadata
  file, reallacates memory and updates data size information.
 **/
/*-------------------------------------------------------------------------*/
void* FTI_Realloc(int id, void* ptr) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return ptr;
    }

    char str[FTI_BUFS];

    FTI_Print("Trying to reallocate dataset.", FTI_DBUG);

    if (FTI_Exec.reco) {
        FTIT_dataset* data;
        if ((FTI_Data->get(&data, id) != FTI_SCES)) {
            FTI_Print("Unable to reallocate variable buffer to stored size!",
             FTI_WARN);
            return ptr;
        }

        if (!data) {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS, "variable id='%d' does not exist, failed"
            " to reallocate buffer", id);
            FTI_Print(str, FTI_WARN);
            return ptr;
        }

        if (data->sizeStored == 0) {
            // sprintf(str, "Cannot allocate 0 size.");
            snprintf(str, sizeof(str), "Cannot allocate 0 size.");
            FTI_Print(str, FTI_WARN);
            return ptr;
        }

        void* tmp = realloc(ptr, data->sizeStored);
        if (!tmp) return ptr;

        ptr = tmp;

        // sprintf(str, "Reallocated size: %ld", data->sizeStored);
        snprintf(str, sizeof(str), "Reallocated size: %lu", data->sizeStored);
        FTI_Print(str, FTI_INFO);

        FTI_Exec.ckptSize += data->sizeStored - data->size;
        data->size = data->sizeStored;
        data->ptr = ptr;
        data->count = data->size / data->eleSize;

        // sprintf(str, "Dataset #%d reallocated.", data->id);
        snprintf(str, sizeof(str), "Dataset #%d reallocated.", data->id);
        FTI_Print(str, FTI_INFO);
    } else {
        FTI_Print("This is not a recovery. Couldn't reallocate memory.",
         FTI_WARN);
    }

    return ptr;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Bit-flip injection following the injection instructions.
  @param      datasetID       ID of the dataset where to inject.
  @return     integer         FTI_SCES if successful.

  This function injects the given number of bit-flips, at the given
  frequency and in the given location (rank, dataset, bit position).

 **/
/*-------------------------------------------------------------------------*/
int FTI_BitFlip(int id) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if (FTI_Inje.rank == FTI_Topo.splitRank) {
        if (id >= FTI_Exec.nbVar) {
            return FTI_NSCS;
        }

        FTIT_dataset* data;
        if ((FTI_Data->get(&data, id) != FTI_SCES) || !data) {
            FTI_Print("Dataset id to inject BitFlip is invalid", FTI_WARN);
            return FTI_NSCS;
        }

        if (FTI_Inje.counter < FTI_Inje.number) {
            if ((MPI_Wtime() - FTI_Inje.timer) > FTI_Inje.frequency) {
                if (FTI_Inje.index < data->count) {
                    char str[FTI_BUFS];
                    if (data->type->id == 9) {  // If it is a double
                        double* target = data->ptr + FTI_Inje.index;
                        double ori = *target;
                        int res = FTI_DoubleBitFlip(target, FTI_Inje.position);
                        FTI_Inje.counter = (res == FTI_SCES) ?
                         FTI_Inje.counter + 1 : FTI_Inje.counter;
                        FTI_Inje.timer = (res == FTI_SCES) ?
                         MPI_Wtime() : FTI_Inje.timer;
                        /*sprintf(str, "Injecting bit-flip in dataset %d,
                         index %d, bit %d : %f => %f",
                                id, FTI_Inje.index, FTI_Inje.position, ori,
                                 *target);*/
                        snprintf(str, sizeof(str), "Injecting bit-flip in"
                        " dataset %d, index %d, bit %d : %f => %f",
                                id, FTI_Inje.index, FTI_Inje.position, ori,
                                 *target);
                        FTI_Print(str, FTI_WARN);
                        return res;
                    }
                    if (data->type->id == 8) {  // If it is a float
                        float* target = data->ptr + FTI_Inje.index;
                        float ori = *target;
                        int res = FTI_FloatBitFlip(target, FTI_Inje.position);
                        FTI_Inje.counter = (res == FTI_SCES) ? FTI_Inje.counter
                         + 1 : FTI_Inje.counter;
                        FTI_Inje.timer = (res == FTI_SCES) ? MPI_Wtime()
                         : FTI_Inje.timer;
                        /*sprintf(str, "Injecting bit-flip in dataset %d, 
                        index %d, bit %d : %f => %f",
                                id, FTI_Inje.index, FTI_Inje.position, ori,
                                 *target);*/
                        snprintf(str, sizeof(str), "Injecting bit-flip in "
                          "dataset %d, index %d, bit %d : %f => %f",
                                id, FTI_Inje.index, FTI_Inje.position, ori,
                                 *target);
                        FTI_Print(str, FTI_WARN);
                        return res;
                    }
                }
            }
        }
    }
    return FTI_NSCS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It takes the checkpoint and triggers the post-ckpt. work.
  @param      id              Checkpoint ID.
  @param      level           Checkpoint level.
  @return     integer         FTI_SCES if successful.

  This function starts by blocking on a receive if the previous ckpt. was
  offline. Then, it updates the ckpt. information. It writes down the ckpt.
  data, creates the metadata and the post-processing work. This function
  is complementary with the FTI_Listen function in terms of communications.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Checkpoint(int id, int level) {
    char str[FTI_BUFS];  // For console output

    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if ((level < FTI_MIN_LEVEL_ID) || (level > FTI_MAX_LEVEL_ID)) {
        FTI_Print("Invalid level id! Aborting checkpoint creation...",
         FTI_WARN);
        return FTI_NSCS;
    }
    if ((level > FTI_L4) && (level < FTI_L4_DCP)) {
        snprintf(str, FTI_BUFS, "dCP only implemented for level 4! setting to"
        " level %d...", level - 4);
        FTI_Print(str, FTI_WARN);
        level -= 4;
    }

    FTI_Exec.isPbdcp=-1;
    if((level>=FTI_L1_PBDCP)&&(level<FTI_L4_PBDCP) ){
        FTI_Print("PBDCP only implemented for L4! The checkpoint will be performed without PBDCP ",FTI_WARN);
        switch(level){
            case FTI_L1_PBDCP: level=1; break;
            case FTI_L2_PBDCP: level=2; break;
            case FTI_L3_PBDCP: level=3; break;
        }
    }

    double t1, t2;

    FTI_Exec.ckptMeta.ckptId = id;

    // reset hdf5 single file requests.
    FTI_Exec.h5SingleFile = false;
    if (level == FTI_L4_H5_SINGLE) {
        if (FTI_Conf.h5SingleFileEnable) {
            FTI_Exec.h5SingleFile = true;
            level = 4;
        }
    }

    // reset dcp requests.
    FTI_Ckpt[4].isDcp = false;
    FTI_Ckpt[4].isPbdcp = false;

    if (level == FTI_L4_DCP) {
        if ((FTI_Conf.ioMode == FTI_IO_FTIFF) ||
         (FTI_Conf.ioMode == FTI_IO_POSIX)) {
            if ( FTI_Conf.dcpFtiff || FTI_Conf.dcpPosix ) {
                FTI_Ckpt[4].isDcp = true;
            } else {
                FTI_Print("L4 dCP requested, but dCP is disabled!", FTI_WARN);
            }
        } else {
            FTI_Print("L4 dCP requested, but dCP needs FTI-FF!", FTI_WARN);
        }
        level = 4;
    }
    if (level == FTI_L4_PBDCP){
        FTI_Exec.isPbdcp=4;
        if(FTI_Conf.pbdcpEnabled){
            FTI_Ckpt[4].isDcp = true;
            FTI_Ckpt[4].isPbdcp = true;
            FTI_Exec.dcpInfoPosix.errorSum = 0;
            FTI_Exec.dcpInfoPosix.nbValues = 0;
        }
        level=4;
    }

    double t0 = MPI_Wtime();  // Start time
    if (FTI_Exec.wasLastOffline == 1) {
        // Block until previous checkpoint is done (Async. work)
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.generalTag,
         FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) {
            // Head sends level of checkpoint if post-processing succeed,
            // FTI_NSCS Otherwise

            // Store last successful post-processing checkpoint level
            FTI_Exec.ckptLvel = lastLevel;
            // sprintf(str, "LastCkptLvel received from head: %d", lastLevel);
            snprintf(str, sizeof(str), "LastCkptLvel received from head: %d",
             lastLevel);
            FTI_Print(str, FTI_DBUG);
        } else {
            FTI_Print("Head failed to do post-processing after previous"
            " checkpoint.", FTI_WARN);
        }
    }

    // Time after waiting for head to done previous post-processing
    t1 = MPI_Wtime();
    FTI_Exec.ckptMeta.level = level;  // assign to temporary metadata
    int res = FTI_Try(FTI_WriteCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt,
     FTI_Data), "write the checkpoint.");
    t2 = MPI_Wtime();  // Time after writing checkpoint
    // no postprocessing or meta data for h5 single file
    if (res == FTI_SCES && FTI_Exec.h5SingleFile) {
#ifdef ENABLE_HDF5
        return FTI_FinalizeH5SingleFile(&FTI_Exec, &FTI_Conf, &FTI_Topo,
         FTI_Ckpt, t2 - t1);
#endif
    }

    if (!FTI_Ckpt[FTI_Exec.ckptMeta.level].isInline) {
        // If postCkpt. work is Async. then send message
        FTI_Exec.activateHeads(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, res);
    } else {  // If post-processing is inline
        FTI_Exec.wasLastOffline = 0;
        if (res != FTI_SCES) {  // If Writing checkpoint failed
            // The same as head call FTI_PostCkpt with reject
            // ckptLvel if not success
            FTI_Exec.ckptMeta.level = FTI_REJW - FTI_BASE;
        }
        res = FTI_Try(FTI_PostCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt),
         "postprocess the checkpoint.");
        if (res == FTI_SCES) {
            FTI_Exec.ckptLvel = FTI_Exec.ckptMeta.level;  // Update level
        }
    }
    double t3;

    if (!FTI_Exec.hasCkpt && (FTI_Topo.splitRank == 0) && (res == FTI_SCES)) {
        // Setting recover flag to 1 (to recover from current ckpt level)
        res = FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 1),
         "update configuration file.");
        // in case FTI couldn't recover all ckpt files in FTI_Init
        FTI_Exec.initSCES = 1;
        if (res == FTI_SCES) {
            FTI_Exec.hasCkpt = true;
        }
    }

    MPI_Bcast(&FTI_Exec.hasCkpt, 1, MPI_INT, 0, FTI_COMM_WORLD);

    t3 = MPI_Wtime();  // Time after post-processing

    if (res != FTI_SCES) {
        // sprintf(str, "Checkpoint with ID %d at Level %d failed.",
        // FTI_Exec.ckptMeta.ckptId, FTI_Exec.ckptMeta.level);
        snprintf(str, sizeof(str), "Checkpoint with ID %d at Level %d failed.",
         FTI_Exec.ckptMeta.ckptId, FTI_Exec.ckptMeta.level);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    /*sprintf(str, "Ckpt. ID %d (L%d) (%.2f MB/proc) taken in %.2f sec. 
    (Wt:%.2fs, Wr:%.2fs, Ps:%.2fs)",
            FTI_Exec.ckptMeta.ckptId, FTI_Exec.ckptMeta.level, 
            FTI_Exec.ckptSize / (1024.0 * 1024.0), t3 - t0,
             t1 - t0, t2 - t1, t3 - t2);*/
    snprintf(str, sizeof(str), "Ckpt. ID %d (L%d) (%.2f MB/proc) taken in %.2f"
    " sec. (Wt:%.2fs, Wr:%.2fs, Ps:%.2fs)",
            FTI_Exec.ckptMeta.ckptId, FTI_Exec.ckptMeta.level,
             FTI_Exec.ckptSize / (1024.0 * 1024.0), t3 - t0, t1 - t0, t2 - t1,
             t3 - t2);
    FTI_Print(str, FTI_INFO);

    if ( (FTI_Conf.dcpFtiff || FTI_Conf.dcpPosix) && FTI_Ckpt[4].isDcp ) {
        FTI_PrintDcpStats(FTI_Conf, FTI_Exec, FTI_Topo);
    }

    // update stored values to allow recovery online.
    // FIXME in such a way, we don't cover the case !inline since at
    // this point we cannot know if the
    // postprocessing has been successfully.
    // One way could be to convert tmp checkpoint into
    // L1 checkpoint and update lateron.

    FTI_Exec.nbVarStored = FTI_Exec.nbVar;
    FTI_Exec.ckptId = FTI_Exec.ckptMeta.ckptId;

    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec.nbVar) != FTI_SCES) {
        FTI_Print("failed to finalize FTI", FTI_WARN);
        return FTI_NSCS;
    }

    int k = 0; for (; k < FTI_Exec.nbVar; k++) {
        data[k].sizeStored = data[k].size;
    }

    return FTI_DONE;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initialize an incremental checkpoint.
  @param      id              Checkpoint ID.
  @param      level           Checkpoint level.
  @param      activate        Boolean expression.
  @return     integer         FTI_SCES if successful.

  This function defines the environment for the incremental checkpointing
  mechanism. The iCP mechanism consists of three functions: FTI_InitICP,
  FTI_AddVarICP and FTI_FinalizeICP. The two functions FTI_InitICP and
  FTI_FinalizeICP define the iCP region within the user may write the
  protected variables in any order. The iCP region is active, when the
  expression passed through 'activate' evaluates to TRUE.

  @note This function is not blocking for POSIX, FTI-FF and HDF5, but,
  blocking for MPI-IO. This is due to the collective open call in MPI_IO
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitICP(int id, int level, bool activate) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    // only step in if activate TRUE.
    if (!activate) {
        return FTI_SCES;
    }

    FTI_Exec.h5SingleFile = false;
    if (level == FTI_L4_H5_SINGLE) {
        if (FTI_Conf.h5SingleFileEnable) {
            FTI_Exec.h5SingleFile = true;
            level = 4;
        }
    }

    // reset iCP meta info (i.e. set counter to zero etc.)
    memset( &(FTI_Exec.iCPInfo), 0x0, sizeof(FTIT_iCPInfo) );

    FTI_Exec.iCPInfo.isWritten = (bool*) calloc(FTI_Conf.maxVarId,
     sizeof(bool));

    // init iCP status with failure
    FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
    FTI_Exec.iCPInfo.result = FTI_NSCS;

    int res = FTI_NSCS;

    char str[FTI_BUFS];  // For console output

    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if ((level < FTI_MIN_LEVEL_ID) || (level > FTI_MAX_LEVEL_ID)) {
        FTI_Print("Invalid level id! Aborting checkpoint creation...",
         FTI_WARN);
        return FTI_NSCS;
    }
    if ((level > FTI_L4) && (level < FTI_L4_DCP)) {
        snprintf(str, FTI_BUFS,
         "dCP only implemented for level 4! setting to level %d...",
          level - 4);
        FTI_Print(str, FTI_WARN);
        level -= 4;
    }

    FTI_Exec.iCPInfo.lastCkptID = FTI_Exec.ckptId;
    // ckptId = 0 if first checkpoint
    FTI_Exec.iCPInfo.isFirstCp = !FTI_Exec.ckptId;
    FTI_Exec.ckptMeta.ckptId = id;

    // reset dcp requests.
    FTI_Ckpt[4].isDcp = false;
    if (level == FTI_L4_DCP) {
        if ((FTI_Conf.ioMode == FTI_IO_FTIFF) ||
         (FTI_Conf.ioMode == FTI_IO_POSIX)) {
            if (FTI_Conf.dcpFtiff || FTI_Conf.dcpPosix) {
                FTI_Ckpt[4].isDcp = true;
            } else {
                FTI_Print("L4 dCP requested, but dCP is disabled!", FTI_WARN);
            }
        } else {
            FTI_Print("L4 dCP requested, but dCP needs FTI-FF!", FTI_WARN);
        }
        level = 4;
    }

    FTI_Exec.iCPInfo.t0 = MPI_Wtime();  // Start time
    // Block until previous checkpoint is done (Async. work)
    if (FTI_Exec.wasLastOffline == 1) {
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank,
         FTI_Conf.generalTag, FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) {
            // Head sends level of checkpoint if post-processing succeed,
            // FTI_NSCS Otherwise

            // Store last successful post-processing checkpoint level
            FTI_Exec.ckptLvel = lastLevel;
            // sprintf(str, "LastCkptLvel received from head: %d", lastLevel);
            snprintf(str, sizeof(str),
             "LastCkptLvel received from head: %d", lastLevel);
            FTI_Print(str, FTI_DBUG);
        } else {
            FTI_Print("Head failed to do post-processing after previous"
            " checkpoint.", FTI_WARN);
        }
    }

    // Time after waiting for head to done previous post-processing
    FTI_Exec.iCPInfo.t1 = MPI_Wtime();
    FTI_Exec.ckptMeta.level = level;  // For FTI_WriteCkpt

    // Name of the  CKPT file.
    snprintf(FTI_Exec.ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.%s",
     FTI_Exec.ckptMeta.ckptId, FTI_Topo.myRank, FTI_Conf.suffix);

    // If checkpoint is inlin and level 4 save directly to PFS
    int offset = 2*(FTI_Conf.dcpPosix);
    if (((FTI_Ckpt[4].isInline && (FTI_Exec.ckptMeta.level == 4)) &&
     !FTI_Exec.h5SingleFile) || (FTI_Exec.h5SingleFile &&
      FTI_Conf.h5SingleFileIsInline) ) {
        if (!((FTI_Conf.dcpFtiff || FTI_Conf.dcpPosix) && FTI_Ckpt[4].isDcp)) {
            MKDIR(FTI_Conf.gTmpDir, 0777);
        } else if ( !FTI_Ckpt[4].hasDcp ) {
            MKDIR(FTI_Ckpt[4].dcpDir, 0777);
        }
        res = FTI_Exec.initICPFunc[GLOBAL](&FTI_Conf, &FTI_Exec, &FTI_Topo,
         FTI_Ckpt, FTI_Data, &ftiIO[GLOBAL+offset]);
    } else {
        if (!((FTI_Conf.dcpFtiff || FTI_Conf.dcpPosix) && FTI_Ckpt[4].isDcp)) {
            MKDIR(FTI_Conf.lTmpDir, 0777);
        } else if ( !FTI_Ckpt[4].hasDcp ) {
            MKDIR(FTI_Ckpt[1].dcpDir, 0777);
        }
        res = FTI_Exec.initICPFunc[LOCAL](&FTI_Conf, &FTI_Exec, &FTI_Topo,
         FTI_Ckpt, FTI_Data, &ftiIO[LOCAL+offset]);
    }

    if (res == FTI_SCES) {
        FTI_Exec.iCPInfo.status = FTI_ICP_ACTV;
    } else {
        FTI_Print("Could Not initialize ICP", FTI_WARN);
    }

    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Write variable into the CP file.
  @param      id              Protected variable ID.
  @return     integer         FTI_SCES if successful.

  With this function, the user may write the protected datasets in any
  order into the checkpoint file. However, before the call to
  FTI_FinalizeICP, all protected variables must have been written into
  the file.
 **/
/*-------------------------------------------------------------------------*/
int FTI_AddVarICP(int varID) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    // only step in if iCP was successfully initialized
    if ( FTI_Exec.iCPInfo.status == FTI_ICP_NINI ) {
        return FTI_SCES;
    }
    if ( FTI_Exec.iCPInfo.status == FTI_ICP_FAIL ) {
        return FTI_NSCS;
    }

    char str[FTI_BUFS];

    FTIT_dataset* data;
    if ((FTI_Data->get(&data, varID) != FTI_SCES) || !data) {
        snprintf(str, FTI_BUFS,
         "FTI_AddVarICP: dataset ID: %d is invalid!", varID);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    // check if dataset was not already written.
    int i = 0; for (; i < FTI_Exec.iCPInfo.countVar; ++i) {
        if (FTI_Exec.iCPInfo.isWritten[varID]) {
            snprintf(str, FTI_BUFS,
             "Dataset with ID: %d was already successfully written!", varID);
            FTI_Print(str, FTI_WARN);
            return FTI_NSCS;
        }
    }

    int res;
    int funcID = FTI_Ckpt[4].isInline && FTI_Exec.ckptMeta.level == 4;
    int offset = 2*(FTI_Conf.dcpPosix);
    res = FTI_Exec.writeVarICPFunc[funcID](varID, &FTI_Conf,
     &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data, &ftiIO[funcID+offset]);

    if (res == FTI_SCES) {
        FTI_Exec.iCPInfo.isWritten[varID] = true;
        FTI_Exec.iCPInfo.countVar++;
    } else {
        FTI_Print("Could not add variable to checkpoint", FTI_WARN);
    }
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalize an incremental checkpoint.
  @return     integer         FTI_SCES if successful.

  This function finalizes an incremental checkpoint. In contrast to
  InitICP, this function is collective on the communicator
  FTI_COMM_WORLD and blocking.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeICP() {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    // if iCP uninitialized, don't step in.
    if ( FTI_Exec.iCPInfo.status == FTI_ICP_NINI ) {
        return FTI_SCES;
    }

    int allRes[2];
    int locRes[2] = { (int)(FTI_Exec.iCPInfo.result == FTI_SCES),
     (int)(FTI_Exec.iCPInfo.countVar == FTI_Exec.nbVar) };
    // Check if all processes have written all the datasets failure free.
    MPI_Allreduce(locRes, allRes, 2, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes[0] != FTI_Topo.nbNodes*FTI_Topo.nbApprocs) {
        FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
        FTI_Print("Not all variables were successfully written!.", FTI_EROR);
    }
    if (allRes[1] != FTI_Topo.nbNodes*FTI_Topo.nbApprocs) {
        FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
        FTI_Print("Not all datasets were added to the CP file!.", FTI_EROR);
    }

    char str[FTI_BUFS];
    int resCP;
    int resPP = FTI_SCES;

    int funcID = FTI_Ckpt[4].isInline && FTI_Exec.ckptMeta.level == 4;
    int offset = 2*(FTI_Conf.dcpPosix);
    resCP = FTI_Exec.finalizeICPFunc[funcID](&FTI_Conf, &FTI_Exec,
     &FTI_Topo, FTI_Ckpt, FTI_Data, &ftiIO[funcID+offset]);

    // no postprocessing or meta data for h5 single file
    if (resCP == FTI_SCES && FTI_Exec.h5SingleFile) {
#ifdef ENABLE_HDF5
        return FTI_FinalizeH5SingleFile(&FTI_Exec, &FTI_Conf,
         &FTI_Topo, FTI_Ckpt, MPI_Wtime() - FTI_Exec.iCPInfo.t0);
#endif
    }

    if (resCP == FTI_SCES) {
        resCP = FTI_Try(FTI_CreateMetadata(&FTI_Conf, &FTI_Exec,
         &FTI_Topo, FTI_Ckpt, FTI_Data), "create metadata.");
    }

    if ( resCP != FTI_SCES ) {
        FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
        // sprintf(str, "Checkpoint with ID %d at Level %d failed.",
        // FTI_Exec.ckptMeta.ckptId, FTI_Exec.ckptMeta.level);
        snprintf(str, sizeof(str),
         "Checkpoint with ID %d at Level %d failed.",
          FTI_Exec.ckptMeta.ckptId, FTI_Exec.ckptMeta.level);
        FTI_Print(str, FTI_WARN);
    }

    double t2 = MPI_Wtime();  // Time after writing checkpoint

    if ((FTI_Conf.dcpFtiff || FTI_Conf.dcpPosix) && FTI_Ckpt[4].isDcp) {
        // After dCP update store total data and dCP sizes in application rank0
        int64_t *dataSize = (FTI_Conf.dcpFtiff)?(int64_t*)&
        FTI_Exec.FTIFFMeta.pureDataSize:&FTI_Exec.dcpInfoPosix.dataSize;
        int64_t *dcpSize = (FTI_Conf.dcpFtiff)?(int64_t*)&
        FTI_Exec.FTIFFMeta.dcpSize:&FTI_Exec.dcpInfoPosix.dcpSize;
        int64_t dcpStats[2];  // 0:totalDcpSize, 1:totalDataSize
        int64_t sendBuf[] = { *dcpSize, *dataSize };
        MPI_Reduce(sendBuf, dcpStats, 2, MPI_UINT64_T, MPI_SUM, 0,
         FTI_COMM_WORLD);
        if (FTI_Topo.splitRank ==  0) {
            *dcpSize = dcpStats[0];
            *dataSize = dcpStats[1];
        }
    }

    // TODO(leobago) this has to come inside postckpt on success!
    if ((FTI_Conf.dcpFtiff || FTI_Conf.keepL4Ckpt) &&
     (FTI_Topo.splitRank == 0)) {
        FTI_WriteCkptMetaData(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt);
    }

    int status = (FTI_Exec.iCPInfo.status == FTI_ICP_FAIL) ?
     FTI_NSCS : FTI_SCES;
    if (!FTI_Ckpt[FTI_Exec.ckptMeta.level].isInline) {
        // If postCkpt. work is Async. then send message
        FTI_Exec.activateHeads(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt,
         status);
    } else {  // If post-processing is inline
        FTI_Exec.wasLastOffline = 0;
        if (FTI_Exec.iCPInfo.status == FTI_ICP_FAIL) {
            // If Writing checkpoint failed
            // The same as head call FTI_PostCkpt with reject
            // ckptLvel if not success
            FTI_Exec.ckptMeta.level = FTI_REJW - FTI_BASE;
        }
        resPP = FTI_Try(FTI_PostCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt),
         "postprocess the checkpoint.");
        if (resPP == FTI_SCES) {
            // Store last successful post-processing checkpoint level
            FTI_Exec.ckptLvel = FTI_Exec.ckptMeta.level;
        }
    }

    if (!FTI_Exec.hasCkpt && (FTI_Topo.splitRank == 0) && (resPP == FTI_SCES)) {
        // Setting recover flag to 1 (to recover from current ckpt level)
        int res = FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 1),
         "update configuration file.");
        FTI_Exec.initSCES = 1;  // in case FTI couldn't recover
                                // all ckpt files in FTI_Init
        if (res == FTI_SCES) {
            FTI_Exec.hasCkpt = true;
        }
    }

    MPI_Bcast(&FTI_Exec.hasCkpt, 1, MPI_INT, 0, FTI_COMM_WORLD);

    double t3 = MPI_Wtime();  // Time after post-processing

    if (resCP == FTI_SCES) {
        FTI_Exec.ckptId = FTI_Exec.ckptMeta.ckptId;
        /*sprintf(str, "Ckpt. ID %d (L%d) (%.2f MB/proc) taken in %.2f sec.
         (Wt:%.2fs, Wr:%.2fs, Ps:%.2fs)",
                FTI_Exec.ckptId, FTI_Exec.ckptMeta.level, FTI_Exec.ckptSize /
                 (1024.0 * 1024.0), t3 - FTI_Exec.iCPInfo.t0, 
                 FTI_Exec.iCPInfo.t1 - FTI_Exec.iCPInfo.t0, t2 - 
                 FTI_Exec.iCPInfo.t1, t3 - t2);*/
        snprintf(str, sizeof(str),
         "Ckpt. ID %d (L%d) (%.2f MB/proc) taken in %.2f sec. (Wt:%.2fs,"
         " Wr:%.2fs, Ps:%.2fs)",
              FTI_Exec.ckptId, FTI_Exec.ckptMeta.level,
              FTI_Exec.ckptSize / (1024.0 * 1024.0), t3 - FTI_Exec.iCPInfo.t0,
              FTI_Exec.iCPInfo.t1 - FTI_Exec.iCPInfo.t0,
              t2 - FTI_Exec.iCPInfo.t1, t3 - t2);
        FTI_Print(str, FTI_INFO);

        if ((FTI_Conf.dcpFtiff || FTI_Conf.dcpPosix) && FTI_Ckpt[4].isDcp) {
            FTI_PrintDcpStats(FTI_Conf, FTI_Exec, FTI_Topo);
        }

        if (FTI_Exec.iCPInfo.isFirstCp && FTI_Topo.splitRank == 0) {
            // Setting recover flag to 1 (to recover from current ckpt level)
            FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 1),
             "update configuration file.");
            // in case FTI couldn't recover all ckpt files in FTI_Init
            FTI_Exec.initSCES = 1;
        }
    } else {
        FTI_Exec.ckptId = FTI_Exec.iCPInfo.lastCkptID;
    }

    free(FTI_Exec.iCPInfo.isWritten);
    FTI_Exec.iCPInfo.isWritten = NULL;

    FTI_Exec.iCPInfo.status = FTI_ICP_NINI;

    // update stored values to allow recovery online.
    // FIXME in such a way, we don't cover the case !inline since at
    // this point we cannot know if the
    // postprocessing has been successfully.
    // One way could be to convert tmp checkpoint into
    // L1 checkpoint and update lateron.

    FTI_Exec.nbVarStored = FTI_Exec.nbVar;

    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec.nbVar) != FTI_SCES) {
        FTI_Print("failed to finalize FTI", FTI_WARN);
        return FTI_NSCS;
    }

    int k=0; for (; k < FTI_Exec.nbVar; k++) {
        data[k].sizeStored = data[k].size;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It loads the checkpoint data.
  @return     integer         FTI_SCES if successful.

  This function loads the checkpoint data from the checkpoint file and
  it updates some basic checkpoint information.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Recover() {
    if ( FTI_Conf.ioMode == FTI_IO_FTIFF ) {
        int ret = FTI_Try(FTIFF_Recover(&FTI_Exec, FTI_Data, FTI_Ckpt),
         "Recovering from Checkpoint");
        return ret;
    }

    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NREC;
    }
    if (FTI_Exec.initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NREC;
    }

    int i;
    char fn[FTI_BUFS];  // Path to the checkpoint file
    char str[2*FTI_BUFS];  // For console output

    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec.nbVarStored) != FTI_SCES) {
        FTI_Print("failed to recover", FTI_WARN);
        return FTI_NREC;
    }

    // Check if number of protected variables matches
    if (FTI_Exec.h5SingleFile) {
#ifdef ENABLE_HDF5
        if (FTI_CheckDimensions(FTI_Data, &FTI_Exec) != FTI_SCES) {
            FTI_Print("Dimension missmatch in VPR file. Recovery failed!",
             FTI_WARN);
            return FTI_NREC;
        }
#else
        FTI_Print("FTI is not compiled with HDF5 support!", FTI_EROR);
        return FTI_NREC;
#endif
    } else if (!(FTI_Ckpt[FTI_Exec.ckptLvel].recoIsDcp && FTI_Conf.dcpPosix)) {
        if (FTI_Exec.nbVar != FTI_Exec.nbVarStored) {
            /*sprintf(str, "Checkpoint has %d protected variables, 
            but FTI protects %d.",
                    FTI_Exec.nbVarStored, FTI_Exec.nbVar);*/
            snprintf(str, sizeof(str), "Checkpoint has %d protected variables,"
              " but FTI protects %d.",
                    FTI_Exec.nbVarStored, FTI_Exec.nbVar);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
        // Check if sizes of protected variables matches
        for (i = 0; i < FTI_Exec.nbVarStored; i++) {
            if (data[i].size != data[i].sizeStored) {
                /*sprintf(str, "Cannot recover %ld bytes to 
                protected variable (ID %d) size: %ld",
                        data[i].sizeStored, data[i].id,
                        data[i].size);*/
                snprintf(str, sizeof(str), "Cannot recover %lu bytes to "
                  "protected variable (ID %d) size: %lu",
                        data[i].sizeStored, data[i].id,
                        data[i].size);
                FTI_Print(str, FTI_WARN);
                return FTI_NREC;
            }
        }
    } else {
        if (FTI_Exec.nbVar != FTI_Exec.dcpInfoPosix.nbVarReco) {
            /*sprintf(str, "Checkpoint has %d 
            protected variables, but FTI protects %d.",
                    FTI_Exec.dcpInfoPosix.nbVarReco, FTI_Exec.nbVar);*/
            snprintf(str, sizeof(str), "Checkpoint has %d protected variables,"
              " but FTI protects %d.",
                    FTI_Exec.dcpInfoPosix.nbVarReco, FTI_Exec.nbVar);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
        // Check if sizes of protected variables matches
        int lidx = FTI_Exec.dcpInfoPosix.nbLayerReco - 1;
        for (i = 0; i < FTI_Exec.nbVarStored; i++) {
            int varId = FTI_Exec.dcpInfoPosix.datasetInfo[lidx][i].varID;
            if ((FTI_Data->get(&data, varId) != FTI_SCES) || !data) {
                char errstr[FTI_BUFS];
                snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", varId);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NREC;
            }
            if (data->sizeStored !=
              FTI_Exec.dcpInfoPosix.datasetInfo[lidx][i].varSize) {
                /*sprintf(str, "Cannot recover %ld bytes to 
                protected variable (ID %d) size: %ld",
                        FTI_Exec.dcpInfoPosix.datasetInfo[lidx][i].varSize, 
                        FTI_Exec.dcpInfoPosix.datasetInfo[lidx][i].varID,
                        data->sizeStored);*/
                snprintf(str, sizeof(str), "Cannot recover %lu bytes to "
                  "protected variable (ID %d) size: %lu",
                        FTI_Exec.dcpInfoPosix.datasetInfo[lidx][i].varSize,
                         FTI_Exec.dcpInfoPosix.datasetInfo[lidx][i].varID,
                        data->sizeStored);
                FTI_Print(str, FTI_WARN);
                return FTI_NREC;
            }
        }
    }

#ifdef ENABLE_HDF5  // If HDF5 is installed
    if (FTI_Conf.ioMode == FTI_IO_HDF5) {
        int ret = FTI_RecoverHDF5(&FTI_Conf, &FTI_Exec, FTI_Ckpt, FTI_Data);
        return ret;
    }
#endif

    // Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec.ckptLvel == 4) {
        if (FTI_Ckpt[4].recoIsDcp && FTI_Conf.dcpPosix) {
            return FTI_RecoverDcpPosix(&FTI_Conf, &FTI_Exec, FTI_Ckpt,
             FTI_Data);
        } else {
            // Try from L1
            snprintf(fn, FTI_BUFS, "%s/Ckpt%d-Rank%d.%s", FTI_Ckpt[1].dir,
             FTI_Exec.ckptId, FTI_Topo.myRank, FTI_Conf.suffix);
            if (access(fn, R_OK) != 0) {
                // if no L1 files try from L4
                snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir,
                 FTI_Exec.ckptMeta.ckptFile);
            }
        }
    } else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec.ckptLvel].dir,
         FTI_Exec.ckptMeta.ckptFile);
    }

    // sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
    snprintf(str, sizeof(str), "Trying to load FTI checkpoint file (%s)...",
      fn);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(fn, "rb");
    if (fd == NULL) {
        // sprintf(str, "Could not open FTI checkpoint file. (%s)...", fn);
        snprintf(str, sizeof(str), "Could not open FTI checkpoint file."
          " (%s)...", fn);
        FTI_Print(str, FTI_EROR);
        return FTI_NREC;
    }

    if (FTI_Data->data(&data, FTI_Exec.nbVarStored) != FTI_SCES) {
        FTI_Print("failed to recover", FTI_WARN);
        return FTI_NREC;
    }

#ifdef GPUSUPPORT

    for (i = 0; i < FTI_Exec.nbVarStored; i++) {
        int64_t filePos = data[i].filePosStored;
        // strncpy(data[i].idChar, data[i].idChar, FTI_BUFS);
        fseek(fd, filePos, SEEK_SET);
        if (data[i].isDevicePtr)
            FTI_TransferFileToDeviceAsync(fd, data[i].devicePtr,
             data[i].sizeStored);
        else
            fread(data[i].ptr, 1, data[i].sizeStored, fd);

        if (ferror(fd)) {
            FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
            fclose(fd);
            return FTI_NREC;
        }
    }

#else
    for (i = 0; i < FTI_Exec.nbVarStored; i++) {
        size_t filePos = data[i].filePos;
        // strncpy(data[i].idChar, data[i].idChar, FTI_BUFS);
        fseek(fd, filePos, SEEK_SET);
        fread(data[i].ptr, 1, data[i].sizeStored, fd);
        if (ferror(fd)) {
            FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
            fclose(fd);
            return FTI_NREC;
        }
    }
#endif
    if (fclose(fd) != 0) {
        FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }

    FTI_Exec.reco = 0;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Takes an FTI snapshot or recovers the data if it is a restart.
  @return     integer         FTI_SCES if successful.

  This function loads the checkpoint data from the checkpoint file in case
  of restart. Otherwise, it checks if the current iteration requires
  checkpointing, if it does it checks which checkpoint level, write the
  data in the files and it communicates with the head of the node to inform
  that a checkpoint has been taken. Checkpoint ID and counters are updated.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Snapshot() {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    int i, res, level = -1;

    if (FTI_Exec.reco) {  // If this is a recovery load icheckpoint data
        res = FTI_Try(FTI_Recover(), "recover the checkpointed data.");
        if (res == FTI_NREC) {
            return FTI_NREC;
        }
    } else {  // If it is a checkpoint test
        res = FTI_SCES;
        FTI_UpdateIterTime(&FTI_Exec);
        if (FTI_Exec.ckptNext == FTI_Exec.ckptIcnt) {
            // If it is time to check for possible ckpt. (every minute)
            FTI_Print("Checking if it is time to checkpoint.", FTI_DBUG);
            if (FTI_Exec.globMeanIter > 60) {
                FTI_Exec.minuteCnt = FTI_Exec.totalIterTime/60;
            } else {
                FTI_Exec.minuteCnt++;  // Increment minute counter
            }
            for (i = 1; i < 5; i++) {  // Check ckpt. level
                if ( (FTI_Ckpt[i].ckptDcpIntv > 0)
                        && (FTI_Exec.minuteCnt/
                          (FTI_Ckpt[i].ckptDcpCnt*FTI_Ckpt[i].ckptDcpIntv)) ) {
                    // dCP level is level + 4
                    level = i + 4;
                    // counts the passed intervall times (if taken or not...)
                    FTI_Ckpt[i].ckptDcpCnt++;
                }
                if ((FTI_Ckpt[i].ckptIntv) > 0
                        && (FTI_Exec.minuteCnt/
                          (FTI_Ckpt[i].ckptCnt*FTI_Ckpt[i].ckptIntv))) {
                    level = i;
                    // counts the passed intervall times (if taken or not...)
                    FTI_Ckpt[i].ckptCnt++;
                }
            }
            if (level != -1) {
                res = FTI_Try(FTI_Checkpoint(FTI_Exec.ckptCnt, level),
                 "take checkpoint.");
                if (res == FTI_DONE) {
                    FTI_Exec.ckptCnt++;
                }
            }
            FTI_Exec.ckptLast = FTI_Exec.ckptNext;
            FTI_Exec.ckptNext = FTI_Exec.ckptNext + FTI_Exec.ckptIntv;
            FTI_Exec.iterTime = MPI_Wtime();  // Reset iteration duration timer
        }
    }
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It closes FTI properly on the application processes.
  @return     integer         FTI_SCES if successful.

  This function notifies the FTI processes that the execution is over, frees
  some data structures and it closes. If this function is not called on the
  application processes the FTI processes will never finish (deadlock).

 **/
/*-------------------------------------------------------------------------*/
int FTI_Finalize() {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }
    MPI_Barrier(FTI_COMM_WORLD);
    if (FTI_Topo.amIaHead) {
        if ( FTI_Conf.stagingEnabled ) {
            FTI_FinalizeStage(&FTI_Exec, &FTI_Topo, &FTI_Conf);
        }
        MPI_Barrier(FTI_Exec.globalComm);
        FTI_Data->clear();
        if (!FTI_Conf.keepHeadsAlive) {
            MPI_Finalize();
            exit(0);
        } else {
            return FTI_SCES;
        }
    }

    // Notice: The following code is only executed by the application procs

    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec.nbVar) != FTI_SCES) {
        FTI_Print("failed to finalize FTI", FTI_WARN);
        return FTI_NSCS;
    }

    // free hashArray memory
    if (FTI_Conf.dcpPosix) {
        int i = 0;
        for (; i < FTI_Exec.nbVar; i++) {
            if (!(data[i].isDevicePtr)) {
                free(data[i].dcpInfoPosix.currentHashArray);
                free(data[i].dcpInfoPosix.oldHashArray);
            }
#ifdef GPUSUPPORT
            else {
                cudaFree(data[i].dcpInfoPosix.currentHashArray);
                cudaFree(data[i].dcpInfoPosix.oldHashArray);
            }
#endif
        }
    }

    FTI_Try(FTI_DestroyDevices(), "Destroying accelerator allocated memory");
    if (FTI_Conf.dcpInfoPosix.cachedCkpt) {
        FTI_destroyMD5();
    }

    // If there is remaining work to do for last checkpoint
    if (FTI_Exec.wasLastOffline == 1) {
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank,
        FTI_Conf.generalTag, FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) {  // Head sends level of checkpoint if
                                      // post-processing succeed,
                                      // FTI_NSCS Otherwise
            FTI_Exec.ckptLvel = lastLevel;
        }
    }

    // Send notice to the head to stop listening
    if (FTI_Topo.nbHeads == 1) {
        int value = FTI_ENDW;
        MPI_Send(&value, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.finalTag,
         FTI_Exec.globalComm);
    }

    // for staging, we have to ensure, that the call to FTI_Clean
    // comes after the heads have written all the staging files.
    // Thus FTI_FinalizeStage is blocking on global communicator.
    if ( FTI_Conf.stagingEnabled ) {
        FTI_FinalizeStage(&FTI_Exec, &FTI_Topo, &FTI_Conf);
    }

    // If we need to keep the last checkpoint and there was a checkpoint
    if ( FTI_Conf.saveLastCkpt && FTI_Exec.hasCkpt ) {
        // if ((FTI_Conf.saveLastCkpt || FTI_Conf.keepL4Ckpt)
        // && FTI_Exec.ckptId > 0) {
        MPI_Barrier(FTI_COMM_WORLD);
        if (FTI_Exec.ckptLvel != 4) {
            FTI_Try(FTI_Flush(&FTI_Conf, &FTI_Exec, &FTI_Topo,
             FTI_Ckpt, FTI_Exec.ckptLvel), "save the last ckpt. in the PFS.");
            MPI_Barrier(FTI_COMM_WORLD);
            if (FTI_Topo.splitRank == 0) {
                if (access(FTI_Ckpt[4].dir, 0) == 0) {
                    // Delete previous L4 checkpoint
                    FTI_RmDir(FTI_Ckpt[4].dir, 1);
                }
                RENAME(FTI_Conf.gTmpDir, FTI_Ckpt[4].dir);
                if ( FTI_Conf.ioMode != FTI_IO_FTIFF ) {
                    if (access(FTI_Ckpt[4].metaDir, 0) == 0) {
                        // Delete previous L4 metadata
                        FTI_RmDir(FTI_Ckpt[4].metaDir, 1);
                    }
                    RENAME(FTI_Ckpt[FTI_Exec.ckptLvel].metaDir,
                     FTI_Ckpt[4].metaDir);
                }
            }
        }
        if (FTI_Topo.splitRank == 0) {
            // Setting recover flag to 2 (to recover from L4,
            // keeped last checkpoint)
            FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 2),
             "update configuration file to 2.");
        }
        // Cleaning only local storage
        FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 6),
         "clean local directories");
    } else if ( FTI_Conf.keepL4Ckpt ) {
        int ckptId = FTI_LoadL4CkptMetaData(&FTI_Conf, &FTI_Exec,
         &FTI_Topo, FTI_Ckpt);
        if (ckptId > 0) {
            FTI_Exec.ckptMeta.ckptIdL4 = ckptId;
            FTI_ArchiveL4Ckpt(&FTI_Conf, &FTI_Exec, FTI_Ckpt, &FTI_Topo);
            MPI_Barrier(FTI_COMM_WORLD);
            FTI_RmDir(FTI_Ckpt[4].dir, FTI_Topo.splitRank == 0);
            MPI_Barrier(FTI_COMM_WORLD);
            // Cleaning only local storage
            FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 6),
             "clean local directories");
        }
    } else {
        if (FTI_Conf.saveLastCkpt) {  // if there was no saved checkpoint
            FTI_Print("No checkpoint to keep in PFS.", FTI_INFO);
        }
        if (FTI_Topo.splitRank == 0) {
            // Setting recover flag to 0 (no checkpoint files
            // to recover from means no recovery)
            FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 0),
             "update configuration file to 0.");
        }
        // Cleaning everything
        FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 5),
         "do final clean.");
    }

    if (FTI_Conf.dcpFtiff) {
        FTI_FinalizeDcp(&FTI_Conf, &FTI_Exec);
    }

    FTI_FreeTypesAndGroups(&FTI_Exec);
    if (FTI_Conf.ioMode == FTI_IO_FTIFF) {
        FTIFF_FreeDbFTIFF(FTI_Exec.lastdb);
    }
#ifdef ENABLE_HDF5
    if (FTI_Conf.h5SingleFileEnable) {
        FTI_FreeVPRMem(&FTI_Exec, FTI_Data);
    }
#endif
    FTI_Data->clear();
    MPI_Barrier(FTI_Exec.globalComm);
    FTI_Print("FTI has been finalized.", FTI_INFO);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes recovery of variable
  @return     integer             FTI_SCES if successful.

  Initializes the I/O operations for recoverVar 
  includes implementation for all I/O modes
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarInit() {
    int res = FTI_NSCS;

    char fn[FTI_BUFS];

    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if (FTI_Exec.reco == 0) {
        /* This is not a restart: no actions performed */
        return FTI_SCES;
    }

    if (FTI_Exec.initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    // Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec.ckptLvel == 4) {
        if (FTI_Ckpt[4].recoIsDcp && FTI_Conf.dcpPosix) {
            // find ckptFile path
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec.ckptLvel].dcpDir,
             FTI_Exec.ckptMeta.ckptFile);
            res = FTI_RecoverVarDcpPosixInit();
        } else {
            snprintf(fn, FTI_BUFS, "%s/Ckpt%d-Rank%d.%s", FTI_Ckpt[1].dir,
             FTI_Exec.ckptId, FTI_Topo.myRank, FTI_Conf.suffix);
        }
    } else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec.ckptLvel].dir,
         FTI_Exec.ckptMeta.ckptFile);
    }
    // Check if sizes of protected variables matches
    // switch case
    switch (FTI_Conf.ioMode) {
        case FTI_IO_HDF5:
#ifdef ENABLE_HDF5
            res = FTI_RecoverVarInitHDF5(&FTI_Conf, &FTI_Exec, FTI_Ckpt);
#else
            FTI_Print("Selected Ckpt I/O is HDF5, but HDF5 is not enabled.",
             FTI_WARN);
#endif
            break;

#ifdef ENABLE_SIONLIB  // --> If SIONlib is installed

        case FTI_IO_SIONLIB:
            res = FTI_RecoverVarInitPOSIX(fn);
#else
            FTI_Print("Selected Ckpt I/O is SION, but SION is not enabled.",
             FTI_WARN);
#endif
            break;

        case FTI_IO_POSIX:
            res = FTI_RecoverVarInitPOSIX(fn);
            break;

        case FTI_IO_MPI:
            res = FTI_RecoverVarInitPOSIX(fn);
            break;

        case FTI_IO_FTIFF:
            res = FTIFF_RecoverVarInit(fn);
            break;

        default:
            FTI_Print("Unknown I/O mode.", FTI_EROR);
            res = FTI_NSCS;
    }
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Recovers given variable
  @param      integer         id of variable to be recovered
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVar(int id) {
    int res = FTI_NSCS;
    // Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec.ckptLvel == 4) {
        if (FTI_Ckpt[4].recoIsDcp && FTI_Conf.dcpPosix) {
            FTI_Print("about to DCP", FTI_INFO);
            res =  FTI_RecoverVarDcpPosix(&FTI_Conf, &FTI_Exec, FTI_Ckpt,
             FTI_Data, id);
            return res;
        }
    }
    switch (FTI_Conf.ioMode) {
        case FTI_IO_HDF5:
#ifdef ENABLE_HDF5  // --> If HDF5 is installed
            res = FTI_RecoverVarHDF5(&FTI_Conf, &FTI_Exec, FTI_Ckpt, FTI_Data,
             id);
#else
            FTI_Print("Selected Ckpt I/O is HDF5, but HDF5 is not enabled.",
             FTI_WARN);
#endif
            break;

#ifdef ENABLE_SIONLIB  // --> If SIONlib is installed

        case FTI_IO_SIONLIB:

            res = FTI_RecoverVarPOSIX(&FTI_Conf, &FTI_Exec, &FTI_Topo,
             FTI_Ckpt, FTI_Data, id, fileposix);
#else
            FTI_Print("Selected Ckpt I/O is SION, but SION is not enabled.",
             FTI_WARN);
#endif
            break;

        case FTI_IO_POSIX:
            res = FTI_RecoverVarPOSIX(&FTI_Conf, &FTI_Exec, &FTI_Topo,
             FTI_Ckpt, FTI_Data, id, fileposix);
            break;

        case FTI_IO_MPI:
            res = FTI_RecoverVarPOSIX(&FTI_Conf, &FTI_Exec, &FTI_Topo,
             FTI_Ckpt, FTI_Data, id, fileposix);
            break;

        case FTI_IO_FTIFF:
            res = FTIFF_RecoverVar(&FTI_Conf, &FTI_Exec, &FTI_Topo,
             FTI_Ckpt, FTI_Data, id);
            break;

        default:
            FTI_Print("Unknown I/O mode.", FTI_EROR);
            res = FTI_NSCS;
    }

    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes recovery of variable
  @return     integer             FTI_SCES if successful.

  Finalizes the I/O operations for recoverVar 
  includes implementation for all I/O modes
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarFinalize() {
    int res = FTI_SCES;

    if (FTI_Exec.ckptLvel == 4) {
        if (FTI_Ckpt[4].recoIsDcp && FTI_Conf.dcpPosix) {
            res = FTI_RecoverVarDcpPosixFinalize();
            return res;
        }
    }
    switch (FTI_Conf.ioMode) {
        case FTI_IO_HDF5:
#ifdef ENABLE_HDF5  // --> If HDF5 is installed
            res = FTI_RecoverVarFinalizeHDF5(&FTI_Conf, &FTI_Exec, FTI_Ckpt,
             FTI_Data);
#else
            FTI_Print("Selected Ckpt I/O is HDF5, but HDF5 is not enabled.",
             FTI_WARN);
#endif
            break;

#ifdef ENABLE_SIONLIB  // --> If SIONlib is installed

        case FTI_IO_SIONLIB:
            res = FTI_RecoverVarFinalizePOSIX(fileposix);
#else
            FTI_Print("Selected Ckpt I/O is SION, but SION is not enabled.",
             FTI_WARN);
#endif
            break;

        case FTI_IO_POSIX:
            res = FTI_RecoverVarFinalizePOSIX(fileposix);
            break;

        case FTI_IO_MPI:
            res = FTI_RecoverVarFinalizePOSIX(fileposix);
            break;

        case FTI_IO_FTIFF:
            res = FTIFF_RecoverVarFinalize(filemmap, filestats);
            break;

        default:
            FTI_Print("Unknown I/O mode.", FTI_EROR);
            res = FTI_NSCS;
    }

    // if( res == FTI_SCES ) FTI_Exec.reco = 0;

    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Prints FTI messages.
  @param      msg             Message to print.
  @param      priority        Priority of the message to be printed.
  @return     void

  This function prints messages depending on their priority and the
  verbosity level set by the user. DEBUG messages are printed by all
  processes with their rank. INFO messages are printed by one process.
  ERROR messages are printed with errno.

 **/
/*-------------------------------------------------------------------------*/
void FTI_Print(char* msg, int priority) {
    FILE *stream = stdout;

    // Sanity Checks
    if (priority < FTI_Conf.verbosity)
        return;
    if (msg == NULL)
      return;

    // Decide the stream
    if (priority == FTI_EROR)
      stream = stderr;

    switch (priority) {
        case FTI_EROR:
            fprintf(stream, "[ " FTI_COLOR_RED
              "FTI Error - %06d" FTI_COLOR_RESET " ] : %s : %s \n",
              FTI_Topo.myRank, msg, strerror(errno));
            break;
        case FTI_WARN:
            fprintf(stream, "[ " FTI_COLOR_ORG
              "FTI Warning %06d" FTI_COLOR_RESET " ] : %s \n",
              FTI_Topo.myRank, msg);
            break;
        case FTI_INFO:
            if (FTI_Topo.splitRank == 0) {
                fprintf(stream, "[ " FTI_COLOR_GRN
                  "FTI  Information" FTI_COLOR_RESET " ] : %s \n", msg);
            }
            break;
        case FTI_IDCP:
            if (FTI_Topo.splitRank == 0) {
                fprintf(stdout, "[ " FTI_COLOR_BLU
                  "FTI  dCP Message" FTI_COLOR_RESET " ] : %s \n", msg);
            }
            break;
        case FTI_DBUG:
            fprintf(stream, "[FTI Debug - %06d] : %s \n",
              FTI_Topo.myRank, msg);
            break;
    }
    fflush(stdout);
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns all configuration info as a structure.

  @param      configFile                    Configuration metadata.
  @param      globalComm                    MPI Global Communicator
  @return     FTIT_allConfiguration         All configuration data structure       

  This function returns all the configuration settings of the execution 
  in the form of a structure FTIT_allConfiguration
 **/
/*-------------------------------------------------------------------------*/
FTIT_allConfiguration FTI_GetConfig(const char* configFile,
 MPI_Comm globalComm) {
    FTIT_allConfiguration FTI_allconf;
      FTI_allconf.configuration = FTI_Conf;
      FTI_allconf.execution = FTI_Exec;
      FTI_allconf.topology = FTI_Topo;
      int level;
      for (level = 1; level < 5; level++) {
          FTI_allconf.checkpoint[level] = FTI_Ckpt[level];
      }
      FTI_allconf.injection = FTI_Inje;
      FTI_Print("FTI configuration returned.", FTI_INFO);
      return FTI_allconf;
}
