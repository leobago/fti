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
#include "interface.h"
#include "utility.h"
#include "api_cuda.h"

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


hsize_t FTI_calculateCountDim(size_t sizeOfElement, hsize_t maxBytes, hsize_t *count, int numOfDimensions, hsize_t *dimensions, hsize_t *sep){
    int i;
    memset(count, 0, sizeof(hsize_t)*numOfDimensions);
    size_t maxElements = maxBytes/sizeOfElement;
    hsize_t bytesToFetch;

    if (maxElements == 0 )
        maxElements = 1;

    hsize_t *dimensionSize = (hsize_t *) malloc (sizeof(hsize_t)*(numOfDimensions+1));
    dimensionSize[numOfDimensions] =1;

    //Calculate how many elements does each whole dimension holds
    for ( i = numOfDimensions - 1; i >=0 ; i--){
        dimensionSize[i] = dimensionSize[i+1] * dimensions[i];
    }

    //Find which is the maximum dimension that I can fetch continuously.
    for ( i = numOfDimensions ; i >= 0; i--){
        if ( maxElements < dimensionSize[i]){
            break;
        }
    }

    // I is =-1 when I can fetch the whole buffer
    if ( i == -1  ){
        *sep = 0;
        bytesToFetch = dimensionSize[*sep+1] * dimensions[*sep] * sizeOfElement; 
        memcpy(count, dimensions, sizeof(hsize_t)*numOfDimensions);
        return bytesToFetch;
    }


    // Calculate the maxium elements of this dimension that I can get
    // This number should be a multiple of the total dimension lenght
    // of this dimension.
    *sep= i;
    int fetchElements = 0;
    for ( i = maxElements/(dimensionSize[*sep+1]) ; i >= 1; i--){
        if ( dimensions[*sep]%i  == 0){
            fetchElements = i;
            break;
        }
    }

    //Fill in the count array and return.
    for ( i = 0 ; i < *sep ; i++ )
        count[i] = 1;

    count[*sep] = fetchElements;
    for ( i = *sep+1; i < numOfDimensions ; i++)
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

int FTI_WriteElements(hid_t dataspace, hid_t dataType, hid_t dataset, hsize_t *count, hsize_t *offset, hsize_t ranks, void *ptr){
    char str[FTI_BUFS];
    hid_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL,count, NULL);
    hsize_t *dims_in= (hsize_t*) malloc (sizeof(hsize_t)*ranks);
    memcpy(dims_in,count,ranks*sizeof(hsize_t));
    hid_t memspace = H5Screate_simple(ranks,dims_in, NULL); 
    hsize_t *offset_in = (hsize_t*) calloc (ranks,sizeof(ranks));
    status = H5Sselect_hyperslab( memspace, H5S_SELECT_SET, offset_in, NULL, count, NULL);
    status = H5Dwrite(dataset, dataType, memspace, dataspace, H5P_DEFAULT, ptr);  
    if (status < 0) {
        free(offset);
        free(count);
        sprintf(str, "Dataset could not be written");
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

int FTI_ReadElements(hid_t dataspace, hid_t dimType, hid_t dataset, hsize_t *count, hsize_t *offset, hsize_t ranks, void *ptr){
    char str[FTI_BUFS];
    hid_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL,count, NULL);
    hsize_t *dims_out= (hsize_t*) malloc (sizeof(hsize_t)*ranks);
    memcpy(dims_out,count,ranks*sizeof(hsize_t));
    hid_t memspace = H5Screate_simple(ranks,dims_out, NULL); 
    hsize_t *offset_out = (hsize_t*) calloc (ranks,sizeof(ranks));
    status = H5Sselect_hyperslab( memspace, H5S_SELECT_SET, offset_out, NULL, count, NULL);
    status = H5Dread(dataset,dimType, memspace, dataspace, H5P_DEFAULT, ptr);  
    if (status < 0) {
        free(offset);
        free(count);
        sprintf(str, "Dataset could not be written");
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

int FTI_AdvanceOffset(hsize_t sep,  hsize_t *start, hsize_t *add, hsize_t *dims, hsize_t rank){
    int i;
    hsize_t carryOut=0;
    hsize_t temp;
    temp = start[sep] + add[sep];

    if ( temp >= dims[sep] ){
        start[sep] = temp % dims[sep]; 
        carryOut=1;
    }
    else{
        start[sep] = temp;
        carryOut=0;
    }

    for ( i = sep-1; i >= 0 &&carryOut ; i--){
        temp = start[i] + carryOut;
        if ( temp >= dims[i] ){
            start[i] = temp % dims[i]; 
            carryOut=1;
        }
        else{
            start[i] = temp;
            carryOut=0;
        }
    }
    return carryOut;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Writes a  protected variable to the checkpoint file.
  @param      FTI_DataVar     The protected variable to be written. 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

  The function Write the data of a single FTI_Protect variable to the HDF5 file. 
  If the data are on the HOST CPU side, all the data are tranfered with a single call.
  If the data are on the GPU side, we use hyperslabs to slice the data and asynchronously
  move data from the GPU side to the host side and then to the filesytem.
 **/
/*-------------------------------------------------------------------------*/

int FTI_WriteHDF5Var(FTIT_dataset *FTI_DataVar){
    int j;
    hsize_t dimLength[32];
    char str[FTI_BUFS];
    int res;

    for (j = 0; j < FTI_DataVar->rank; j++) {
        dimLength[j] = FTI_DataVar->dimLength[j];
    }
  
    hid_t dataspace = H5Screate_simple( FTI_DataVar->rank, dimLength, NULL);
    hid_t dataset = H5Dcreate2 ( FTI_DataVar->h5group->h5groupID, FTI_DataVar->name,FTI_DataVar->type->h5datatype, dataspace,  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // If my data are stored in the CPU side
    // Just store the data to the file and return;
#ifdef GPUSUPPORT    
    if ( !FTI_DataVar->isDevicePtr ){
#endif
        res = H5Dwrite(dataset,FTI_DataVar->type->h5datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, FTI_DataVar->ptr);  
        if (res < 0) {
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }

        res = H5Dclose(dataset);
        if (res < 0) {
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        res = H5Sclose(dataspace);
        if (res < 0) {
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        return FTI_SCES;
#ifdef GPUSUPPORT        
    }

    // This code is only executed in the GPU case.

    hsize_t *count = (hsize_t*) malloc (sizeof(hsize_t)*FTI_DataVar->rank); 
    hsize_t *offset= (hsize_t*) calloc (FTI_DataVar->rank,sizeof(hsize_t)); 

    if ( !count|| !offset){
        sprintf(str, "Could Not allocate count and offset regions");
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }


    hsize_t seperator;
    hsize_t fetchBytes = FTI_getHostBuffSize();
    fetchBytes = FTI_calculateCountDim(FTI_DataVar->eleSize, fetchBytes ,count, FTI_DataVar->rank, dimLength, &seperator);


    FTIT_data_prefetch prefetcher;
    prefetcher.fetchSize = fetchBytes;
    prefetcher.totalBytesToFetch = FTI_DataVar->size;
    prefetcher.isDevice = FTI_DataVar->isDevicePtr;
    prefetcher.dptr = FTI_DataVar->devicePtr;
    size_t bytesToWrite;
    FTI_InitPrefetcher(&prefetcher);
    unsigned char *basePtr = NULL;


    if ( FTI_Try(FTI_getPrefetchedData(&prefetcher, &bytesToWrite, &basePtr), "Fetch next memory block from GPU to write to HDF5") !=  FTI_SCES){
        return FTI_NSCS;
    }

    while( basePtr  ){
        res = FTI_WriteElements( dataspace, FTI_DataVar->type->h5datatype, dataset, count, offset, FTI_DataVar->rank , basePtr);
        if (res != FTI_SCES ) {
            free(offset);
            free(count);
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        FTI_AdvanceOffset(seperator, offset,count, dimLength, FTI_DataVar->rank);

        if ( FTI_Try(FTI_getPrefetchedData(&prefetcher, &bytesToWrite, &basePtr), 
              "Fetch next memory block from GPU to write to HDF5") !=  FTI_SCES){
            return FTI_NSCS;
        }

    }


    res = H5Dclose(dataset);
    if (res < 0) {
        free(offset);
        free(count);
        sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    res = H5Sclose(dataspace);
    if (res < 0) {
        free(offset);
        free(count);
        sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
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
  @param      FTI_DataVar     The Var we will read from the  to the Checkpoint file 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

  The function reads the data of a single FTI_Protect variable to the HDF5 file. 
  If the data are on the HOST CPU side, all the data are tranfered with a single call.
  If the data should move to the GPU side, we use hyperslabs to slice the data and asynchronously
  move data from the File to the CPU and then to GPU side.
 **/
/*-------------------------------------------------------------------------*/
int FTI_ReadHDF5Var(FTIT_dataset *FTI_DataVar){
    char str[FTI_BUFS];
    int res;

    hid_t dataset = H5Dopen(FTI_DataVar->h5group->h5groupID, FTI_DataVar->name, H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);

    // If my data are stored in the CPU side
    // Just store the data to the file and return;
#ifdef GPUSUPPORT    
    if ( !FTI_DataVar->isDevicePtr ){
#endif
        res = H5Dread(dataset,FTI_DataVar->type->h5datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, FTI_DataVar->ptr);  
        if (res < 0) {
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }

        res = H5Dclose(dataset);
        if (res < 0) {
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        res = H5Sclose(dataspace);
        if (res < 0) {
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        return FTI_SCES;
#ifdef GPUSUPPORT        
    }

    hsize_t dimLength[32];
    int j;
    for (j = 0; j < FTI_DataVar->rank; j++) {
        dimLength[j] = FTI_DataVar->dimLength[j];
    }

    // This code is only executed in the GPU case.
    

    hsize_t *count = (hsize_t*) malloc (sizeof(hsize_t)*FTI_DataVar->rank); 
    hsize_t *offset= (hsize_t*) calloc (FTI_DataVar->rank,sizeof(hsize_t)); 

    if ( !count|| !offset){
        sprintf(str, "Could Not allocate count and offset regions");
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }


    hsize_t seperator;
    size_t fetchBytes;
    size_t hostBufSize = FTI_getHostBuffSize();
    //Calculate How many dimension I can compute each time 
    //and how bug should the HOST-GPU communication buffer should be

    fetchBytes = FTI_calculateCountDim(FTI_DataVar->eleSize, hostBufSize ,count, FTI_DataVar->rank, dimLength, &seperator);

    //If the buffer is smaller than the minimum amount 
    //then I need to allocate a bigger one.
    if (hostBufSize < fetchBytes){
        if ( FTI_Try( FTI_DestroyDevices(), "Deleting host buffers" ) != FTI_SCES){
            free(offset);
            free(count);
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }

        if ( FTI_Try (FTI_InitDevices( fetchBytes ), "Allocating host buffers")!= FTI_SCES) {
            free(offset);
            free(count);
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
    }

    unsigned char *basePtr = NULL;
    int id = 0;
    int prevId = 1;
    hsize_t totalBytes = FTI_DataVar->size;
    cudaStream_t streams[2]; 
    //Create the streams for the asynchronous data movement.
    CUDA_ERROR_CHECK(cudaStreamCreate(&(streams[0])));
    CUDA_ERROR_CHECK(cudaStreamCreate(&(streams[1])));
    unsigned char *dPtr = FTI_DataVar->devicePtr;
    // Perform the while loop until all data
    // are processed.
    while( totalBytes  ){
        basePtr = FTI_getHostBuffer(id); 
        //Read file 
        res = FTI_ReadElements( dataspace, FTI_DataVar->type->h5datatype, dataset, count, offset, FTI_DataVar->rank , basePtr);
        CUDA_ERROR_CHECK(cudaMemcpyAsync( dPtr , basePtr, fetchBytes, cudaMemcpyHostToDevice, streams[id]));
        if (res != FTI_SCES ) {
            free(offset);
            free(count);
            sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        //Increase accordingly the file offset
        FTI_AdvanceOffset(seperator, offset,count, dimLength, FTI_DataVar->rank);
        //Syncing the cuda stream.
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
        sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    res = H5Sclose(dataspace);
    if (res < 0) {
        free(offset);
        free(count);
        sprintf(str, "Dataset #%d could not be written", FTI_DataVar->id);
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
  @brief      Writes ckpt to using HDF5 file format.
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
        FTIT_dataset* FTI_Data)
{
    FTI_Print("I/O mode: HDF5.", FTI_DBUG);
    char str[FTI_BUFS], fn[FTI_BUFS];
    int level = FTI_Exec->ckptLvel;
    if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    //Creating new hdf5 file
    hid_t file_id = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        sprintf(str, "FTI checkpoint file (%s) could not be opened.", fn);
        FTI_Print(str, FTI_EROR);

        return FTI_NSCS;
    }
    FTI_Exec->H5groups[0]->h5groupID = file_id;
    FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

    int i;
    for (i = 0; i < rootGroup->childrenNo; i++) {
        FTI_CreateGroup(FTI_Exec->H5groups[rootGroup->childrenID[i]], file_id, FTI_Exec->H5groups);
    }

    // write data into ckpt file


    for (i = 0; i < FTI_Exec->nbVar; i++) {
        int toCommit = 0;
        if (FTI_Data[i].type->h5datatype < 0) {
            toCommit = 1;
        }
        sprintf(str, "Calling CreateComplexType [%d] with hid_t %ld", FTI_Data[i].type->id, FTI_Data[i].type->h5datatype);
        FTI_Print(str, FTI_DBUG);
        FTI_CreateComplexType(FTI_Data[i].type, FTI_Exec->FTI_Type);
        if (toCommit == 1) {
            char name[FTI_BUFS];
            if (FTI_Data[i].type->structure == NULL) {
                //this is the array of bytes with no name
                sprintf(name, "Type%d", FTI_Data[i].type->id);
            } else {
                strncpy(name, FTI_Data[i].type->structure->name, FTI_BUFS);
            }
            herr_t res = H5Tcommit(FTI_Data[i].type->h5group->h5groupID, name, FTI_Data[i].type->h5datatype, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (res < 0) {
                sprintf(str, "Datatype #%d could not be commited", FTI_Data[i].id);
                FTI_Print(str, FTI_EROR);
                int j;
                for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
                    FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
                }
                H5Fclose(file_id);
                return FTI_NSCS;
            }
        }
        //Write Dataset
        if ( FTI_Try(FTI_WriteHDF5Var(&FTI_Data[i]) , "Writing data to HDF5 filesystem") != FTI_SCES){
            sprintf(str, "Dataset #%d could not be written", FTI_Data[i].id);
            FTI_Print(str, FTI_EROR);
            int j;
            for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
                FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
            }
            H5Fclose(file_id);
            return FTI_NSCS;
        }
    }

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_CloseComplexType(FTI_Data[i].type, FTI_Exec->FTI_Type);
    }

    int j;
    for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
        FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
    }

    // close file
    FTI_Exec->H5groups[0]->h5groupID = -1;
    if (H5Fclose(file_id) < 0) {
        FTI_Print("FTI checkpoint file could not be closed.", FTI_EROR);
        return FTI_NSCS;
    }


    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It loads the HDF5 checkpoint data.
  @return     integer         FTI_SCES if successful.

  This function loads the checkpoint data from the checkpoint file and
  it updates checkpoint information.

 **/
/*-------------------------------------------------------------------------*/

int FTI_RecoverHDF5(FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    char str[FTI_BUFS], fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);

    sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);

    hid_t file_id = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        FTI_Print("Could not open FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }
    FTI_Exec->H5groups[0]->h5groupID = file_id;
    FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

    int i;
    for (i = 0; i < FTI_Exec->H5groups[0]->childrenNo; i++) {
        FTI_OpenGroup(FTI_Exec->H5groups[rootGroup->childrenID[i]], file_id, FTI_Exec->H5groups);
    }

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_CreateComplexType(FTI_Data[i].type, FTI_Exec->FTI_Type);
        herr_t res = FTI_ReadHDF5Var(&FTI_Data[i]);
        if (res < 0) {
            FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
            int j;
            for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
                FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
            }
            H5Fclose(file_id);
            return FTI_NREC;
        }
        FTI_CloseComplexType(FTI_Data[i].type, FTI_Exec->FTI_Type);
    }

    int j;
    for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
        FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
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
  @brief      During the restart, recovers the given variable
  @param      id              Variable to recover
  @return     int             FTI_SCES if successful.

  During a restart process, this function recovers the variable specified
  by the given id.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarHDF5(FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,

        FTIT_dataset* FTI_Data, int id)
{
    char str[FTI_BUFS], fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);

    sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);

    hid_t file_id = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        FTI_Print("Could not open FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }
    FTI_Exec->H5groups[0]->h5groupID = file_id;
    FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

    int i;
    for (i = 0; i < FTI_Exec->H5groups[0]->childrenNo; i++) {
        FTI_OpenGroup(FTI_Exec->H5groups[rootGroup->childrenID[i]], file_id, FTI_Exec->H5groups);
    }

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if (FTI_Data[i].id == id) {
            break;
        }
    }

    hid_t h5Type = FTI_Data[i].type->h5datatype;
    if (FTI_Data[i].type->id > 10 && FTI_Data[i].type->structure == NULL) {
        //if used FTI_InitType() save as binary
        h5Type = H5Tcopy(H5T_NATIVE_CHAR);
        H5Tset_size(h5Type, FTI_Data[i].size);
    }
    herr_t res = H5LTread_dataset(FTI_Data[i].h5group->h5groupID, FTI_Data[i].name, h5Type, FTI_Data[i].ptr);
    if (res < 0) {
        FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
        int j;
        for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
            FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
        }
        H5Fclose(file_id);
        return FTI_NREC;
    }

    int j;
    for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
        FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
    }
    if (H5Fclose(file_id) < 0) {
        FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }
    return FTI_SCES;
}

#endif
