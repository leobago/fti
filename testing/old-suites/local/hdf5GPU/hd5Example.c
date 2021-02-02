#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <math.h>


#define MYFILE            "myTest.h5"
#define DATASET         "DS1"
#define SIZE 30*20*10*2 


typedef struct {
    int id;
    int x;
    int y;
    int z;
}dimensions;


void initData(dimensions ptr[], int sizex, int sizey, int sizez){
    int i,j,k, index ;
    index = 0;
    for ( i = 0 ; i < sizex; i++){
        for ( j = 0 ; j < sizey ; j++ ){
            for ( k = 0 ; k < sizez ; k++) {
                ptr[index].id = index;
                ptr[index].x = index;
                ptr[index].y = index;
                ptr[index].z = index;
                index++;
            }
        }
    }
}

void printDimensions(dimensions *ptr, int size){
    int i;
    for ( i = 0 ; i < size ; i ++)
        printf("ID %d (%d %d %d)\n",ptr[i].id, ptr[i].x, ptr[i].y, ptr[i].z); 
}

dimensions ***allocateLinearMemory(hsize_t x, hsize_t y, hsize_t z)
{
    dimensions *p = (dimensions*) malloc(x * y * z * sizeof(dimensions));
    dimensions ***q = (dimensions***) malloc(x * sizeof(dimensions**));
    int i;
    for (i = 0; i < x; i++)
    {
        q[i] = (dimensions **) malloc(y * sizeof(dimensions *));
        int j;
        for (j = 0; j < y; j++)
        {
            int idx = x*j + x*y*i;
            q[i][j] = &p[idx];
        }
    }
    return q;
} 

void deallocateLinearMemory(hsize_t x, dimensions ***q)
{
    free(q[0][0]);
    for(hsize_t i = 0; i < x; i++)
    {
        free(q[i]);
    }
    free(q);    
}

hsize_t calculateCountDim(size_t sizeOfElement, hsize_t maxBytes, hsize_t *count, int numOfDimensions, hsize_t *dimensions, hsize_t *sep){
    int i;
    memset(count, 0, sizeof(hsize_t)*numOfDimensions);
    size_t maxElements = maxBytes/sizeOfElement;
    hsize_t bytesToFetch;
    if (maxElements == 0 )
        maxElements = 1;
    hsize_t *dimensionSize = (hsize_t *) malloc (sizeof(hsize_t)*(numOfDimensions+1));
    // The last dimension is actually the element itself
    dimensionSize[numOfDimensions] =1;

    for ( i = numOfDimensions - 1; i >=0 ; i--){
        dimensionSize[i] = dimensionSize[i+1] * dimensions[i];
        printf("%d %d\n",i, dimensionSize[i]);
    }
    
    for ( i = numOfDimensions ; i >= 0; i--){
        if ( maxElements < dimensionSize[i]){
            break;
        }
    }

    if ( i == -1  ){
        *sep = 0;
        bytesToFetch = dimensionSize[*sep+1] * dimensions[*sep] * sizeOfElement; 
        memcpy(count, dimensions, sizeof(hsize_t)*numOfDimensions);
            return bytesToFetch;
    }
    *sep= i;

    int fetchElements = 0;
    for ( i = maxElements/(dimensionSize[*sep+1]) ; i >= 1; i--){
        if ( dimensions[*sep]%i  == 0){
            fetchElements = i;
            break;
        }
    }


    for ( i = 0 ; i < *sep ; i++ )
        count[i] = 1;

    count[*sep] = fetchElements;
    for ( i = *sep+1; i < numOfDimensions ; i++)
        count[i] = dimensions[i];
    
    bytesToFetch = dimensionSize[*sep+1] * count[*sep] * sizeOfElement; 
    return bytesToFetch;

}

int calculateCountDimOld(size_t sizeOfElement, hsize_t *maxBytes, hsize_t *count, int numOfDimensions, hsize_t *dimensions){
    int i;
    // Initialize entire buffer equal to 1, In the worst case scenario I will 
    // read at least one element
    for ( i = 0; i < numOfDimensions; i++){
        count[i] = 1;
    }
    //Calculate In bytes size of each dimension.
    hsize_t *dimensionSize = (hsize_t *) malloc (sizeof(hsize_t)*(numOfDimensions+1));
    dimensionSize[numOfDimensions] = sizeOfElement; //dimensions[numOfDimensions-1]; 
    i =numOfDimensions;


    for ( i = numOfDimensions-1; i >=0 ; i--){
        dimensionSize[i] = dimensionSize[i+1]*dimensions[i];
    }

    hsize_t currentDimension = 0;
    hsize_t remainElements = (*maxBytes); 
    while ( currentDimension <= numOfDimensions ){
        if ( (*maxBytes) >= dimensionSize[currentDimension] )
            count[currentDimension] = dimensions[currentDimension];
        currentDimension++;           
    }
    count[currentDimension] = (*maxBytes)/dimensionSize[currentDimension+1];
    *maxBytes -= count[currentDimension] * dimensionSize[currentDimension+1];

    return currentDimension;

}

void readElements(hid_t dataspace, hid_t dimType, hid_t dataset, hsize_t *count, hsize_t *offset, hsize_t ranks, void *readData, hsize_t elementsRead ){
    hid_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL,count, NULL);
    hsize_t *dims_out = (hsize_t*) malloc (sizeof(hsize_t)*ranks);
    memcpy(dims_out,count,ranks*sizeof(hsize_t));
    hid_t memspace = H5Screate_simple(ranks,dims_out, NULL); 

    hsize_t *offset_out = (hsize_t*) calloc (ranks,sizeof(ranks));

    status = H5Sselect_hyperslab( memspace, H5S_SELECT_SET, offset_out, NULL, count, NULL);
    status = H5Dread(dataset,dimType, memspace, dataspace, H5P_DEFAULT, readData);  

    printDimensions(readData,elementsRead );

    free(offset_out);
    free(dims_out);

}

int advanceOffset(hsize_t sep,  hsize_t *start, hsize_t *add, hsize_t *dims, hsize_t rank){
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


int main( int argc, char *argv){
    herr_t status;
    hid_t file = H5Fcreate(  MYFILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dimType = H5Tcreate( H5T_COMPOUND, sizeof(dimensions));
    status = H5Tinsert(dimType, "id", HOFFSET(dimensions,id),  H5T_NATIVE_INT);
    status = H5Tinsert(dimType, "x", HOFFSET(dimensions,x),  H5T_NATIVE_INT);
    status = H5Tinsert(dimType, "y", HOFFSET(dimensions,y),  H5T_NATIVE_INT);
    status = H5Tinsert(dimType, "z", HOFFSET(dimensions,z),  H5T_NATIVE_INT);

    hsize_t i, j;
    srand(23);
    hsize_t ranks = 3;
    hsize_t *dims = (hsize_t*) malloc (sizeof(hsize_t)*ranks);
    for ( i = 0 ; i < ranks; i++ ){
        dims[i] = (i+1)*10;
    }

    hsize_t sizeInBytes = sizeof(dimensions);
    for ( i = 0 ; i < ranks; i++){
        sizeInBytes *= dims[i]; 
    }


    dimensions ***data = allocateLinearMemory(dims[0],dims[1],dims[2]);
    initData(&data[0][0][0],dims[0],dims[1],dims[2]);
    //printDimensions(&data[0][0][0],dims[0]*dims[1]*dims[2]);

    hid_t dataspace = H5Screate_simple( ranks , dims, NULL);
    hid_t dataset = H5Dcreate2 ( file, "firstArray", dimType, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset, dimType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0][0][0]);

    status = H5Dclose(dataset);
    status= H5Fclose(file);
    status = H5Sclose(dataspace);


    // Open FILE again
    file = H5Fopen(MYFILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, "firstArray", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);

    dimensions readData[SIZE];
    hsize_t count[3];
    hsize_t offset[3] = {0,0,0};
    hsize_t preferedFetchSize = SIZE * sizeof(dimensions);


    hsize_t totalBytes = sizeof(dimensions)*dims[0]*dims[1]*dims[2];

    hsize_t elementsRead = 1;


    hsize_t seperator;
    hsize_t fetchBytes = calculateCountDim(sizeof(dimensions), preferedFetchSize ,count, 3, dims, &seperator);
    printf("Remaining bytes are  %lld\n", totalBytes );
    dataset = H5Dopen(file, "firstArray", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    while ( totalBytes > 0 ){
        printf("=========================\n");
        printf("Start Before    %lld %lld %lld\n", offset[0],offset[1],offset[2]);
        printf("Count Is        %lld %lld %lld\n", count[0], count[1], count[2]);
        readElements ( dataspace, dimType, dataset, count, offset, 3, readData, fetchBytes/sizeof(dimensions));
        advanceOffset(seperator, offset,count, dims, 3);
        totalBytes -=fetchBytes;
        printf("Start After     %lld %lld %lld\n", offset[0],offset[1],offset[2]);
        printf("Remaining bytes are %lld\n", totalBytes );
    }

    status = H5Sclose(dataspace);
    status = H5Fclose(file);
    status = H5Dclose(dataset);
    deallocateLinearMemory(dims[0],data);
    free(dims);


}

