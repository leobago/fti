#include "interface.h"

// allocate static memory
FTIFF_MPITypeInfo FTIFF_MPITypes[FTIFF_NUM_MPI_TYPES];

void FTI_FreeDbFTIFF(FTIFF_db* last)
{
    if (last) {
        FTIFF_db *current = last;
        FTIFF_db *previous;
        while( current ) {
            previous = current->previous;
            free(current->dbvars);
            free(current);
            current = previous;
        }
    }
}

int FTIFF_Checksum(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data, char* checksum)
{       
    
    MD5_CTX mdContext;
    MD5_Init (&mdContext);
    FTIFF_db *currentdb = FTI_Exec->firstdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *dptr;
    int dbvar_idx, pvar_idx, dbcounter=0;

    int isnextdb;

    //MD5_Update (&mdContext, &(FTI_Exec->FTIFFMeta), sizeof(FTIFF_metaInfo));

    do {

        isnextdb = 0;

        MD5_Update (&mdContext, &(currentdb->numvars), sizeof(int));
        MD5_Update (&mdContext, &(currentdb->dbsize), sizeof(long));

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            MD5_Update (&mdContext, currentdbvar, sizeof(FTIFF_dbvar));

        }

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            dptr = (char*)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
            MD5_Update (&mdContext, dptr, currentdbvar->chunksize);

        }

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_Final (hash, &mdContext);
    int ii = 0, i;
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&checksum[ii], "%02x", hash[i]);
        ii += 2;
    }
}

int FTIFF_InitMpiTypes() 
{
    
    // define MPI datatypes
    // headInfo
    DECL_MPI_TYPES( headInfo );
    MBR_CNT( headInfo ) =  6;
    MBR_BLK_LEN( headInfo ) = { 1, 1, FTI_BUFS, 1, 1, 1 };
    MBR_TYPES( headInfo ) = { MPI_INT, MPI_INT, MPI_CHAR, MPI_LONG, MPI_LONG, MPI_LONG };
    MBR_DISP( headInfo ) = {  
        offsetof( FTIFF_headInfo, exists), 
        offsetof( FTIFF_headInfo, nbVar), 
        offsetof( FTIFF_headInfo, ckptFile), 
        offsetof( FTIFF_headInfo, maxFs), 
        offsetof( FTIFF_headInfo, fs), 
        offsetof( FTIFF_headInfo, pfs) 
    };
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrCnt = headInfo_mbrCnt;
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrBlkLen = headInfo_mbrBlkLen;
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrTypes = headInfo_mbrTypes;
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrDisp = headInfo_mbrDisp;
   
    // commit MPI types
    int i;
    for(i=0; i<FTIFF_NUM_MPI_TYPES; i++) {
        MPI_Type_create_struct( 
                FTIFF_MPITypes[i].mbrCnt, 
                FTIFF_MPITypes[i].mbrBlkLen, 
                FTIFF_MPITypes[i].mbrDisp, 
                FTIFF_MPITypes[i].mbrTypes, 
                &FTIFF_MPITypes[i].raw );
        MPI_Type_get_extent( 
                FTIFF_MPITypes[i].raw, 
                &lb, 
                &extent );
        MPI_Type_create_resized( 
                FTIFF_MPITypes[i].raw, 
                lb, 
                extent, 
                &FTIFF_MPITypes[i].final);
        MPI_Type_commit( &FTIFF_MPITypes[i].final );
    }
 

}
