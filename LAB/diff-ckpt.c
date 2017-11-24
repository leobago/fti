#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdint.h>
#include "../deps/md5/md5.h"
#define CHUNK_SIZE 131072    /**< MD5 algorithm chunk size.      */
#include "fti.h"

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

#include "../deps/jerasure/include/galois.h"
#include "../deps/jerasure/include/jerasure.h"

#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <limits.h>

// INSTEAD OF FLAG IN DATABLOCK, WRITE ONLY ZEROS IN THE END

/*
 * declarations of structures and functions
 */

typedef struct pvar_struct_raw {
    int id;
    void *buffer;
    long disp;
    long size;
} pvar_struct; // protected variable

typedef struct dbvar_struct_raw {
    int id;
    int idx;   // index to corresponding id in pvar array
    long dptr;  // data pointer
    long fptr;  // file pointer
    long chunksize;
} dbvar_struct; // defines variable chunk in datablock

typedef struct db_struct_raw {
    int numvars;
    long dbsize;
    dbvar_struct *dbvars;
    struct db_struct_raw *previous;
    struct db_struct_raw *next;
} db_struct; // defines datablock

void update_metadata();
void recover();
void finalize();
void checkpoint(); // perform checkpoint
void protect_var(int id, void *buffer, long size, long disp); // protect variable
void edit_pvar(int id, void *buffer, long new_size); // edit protect variable

/*
 * global variables
 */

long size1=10000;
long size2=20000;
long size3=30000;
long size4=40000;
long size5=50000;
long size22=60000;
long size32=70000;

// meta data of protected variables 
pvar_struct *pvars = NULL;
int num_pvars = 0;
int num_pvars_old = 0;

// datablock size in file
const int dbstructsize 
    = sizeof(int)  /* numvars */ 
    + sizeof(long); /* dbsize */ 

// var info element size in file
const int dbvarstructsize 
    = sizeof(int)  /* id */ 
    + sizeof(int)  /* idx */ 
    + sizeof(long)  /* dptr */ 
    + sizeof(long)  /* fptr */ 
    + sizeof(long); /* chunksize */ 

// filename
char fn[256];
int rank;

// init datablock list
db_struct *firstdb = NULL;
db_struct *lastdb = NULL;

int main() {

    MPI_Init(NULL,NULL);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    sprintf(fn, "testfile-rank-%i.file", rank);

    int i;
    int arr1[size1];
    for(i=0;i<size1;i++){arr1[i]=i+1;}

    int *arr2 = (int*) malloc( size2 * sizeof(int) );
    for(i=0;i<size2;i++){arr2[i]=i+1;}
    
    int *arr3 = (int*) malloc( size3 * sizeof(int) );
    for(i=0;i<size3;i++){arr3[i]=i+1;}
    
    int arr4[size4];
    for(i=0;i<size4;i++){arr4[i]=i+1;}
    
    int arr5[size5];
    for(i=0;i<size5;i++){arr5[i]=i+1;}

    protect_var(1, arr1, size1, sizeof(int));  
    protect_var(2, arr2, size2, sizeof(int));  
    protect_var(3, arr3, size3, sizeof(int)); 
    
    checkpoint();
    
    protect_var(4, arr4, size4, sizeof(int)); 
    
    checkpoint();

    arr2 = (int*) realloc(arr2, sizeof(int) * size22);
    for(i=size2;i<size22;i++){arr2[i]=i+1;}
    
    arr3 = (int*) realloc(arr3, sizeof(int) * size32);
    for(i=size3;i<size32;i++){arr3[i]=i+1;}
    
    edit_pvar(2, arr2, size22);
    edit_pvar(3, arr3, size32);
    
    checkpoint();
    
    protect_var(5, arr5, size5, sizeof(int)); 
    
    checkpoint();

    //printf("expected value arr1[67] is 67. we have -> %i\n",arr1[67]);

    finalize();
    
    for(i=0;i<size1;i++){arr1[i]=0;}

    for(i=0;i<size22;i++){arr2[i]=0;}
    
    for(i=0;i<size32;i++){arr3[i]=0;}
    
    for(i=0;i<size4;i++){arr4[i]=0;}
    
    for(i=0;i<size5;i++){arr5[i]=0;}

    recover();

    //printf("expected value arr1[67] is 67. we have -> %i\n",arr1[67]);

    if (pvars) {
        free(pvars);
    }

    long a1_ex = (size1/2)*(size1+1);
    long a2_ex = (size22/2)*(size22+1);
    long a3_ex = (size32/2)*(size32+1);
    long a4_ex = (size4/2)*(size4+1);
    long a5_ex = (size5/2)*(size5+1);

    long a1=0, a2=0, a3=0, a4=0, a5=0;

    for(i=0;i<size1;i++){a1+=arr1[i];}

    for(i=0;i<size22;i++){a2+=arr2[i];}
    
    for(i=0;i<size32;i++){a3+=arr3[i];}
    
    for(i=0;i<size4;i++){a4+=arr4[i];}
    
    for(i=0;i<size5;i++){a5+=arr5[i];}
   
    printf(
            "rank -> %i\n"
            "-----------------------\n"
            "a1_ex: %ld, a1: %ld\n"
            "a2_ex: %ld, a2: %ld\n"
            "a3_ex: %ld, a3: %ld\n"
            "a4_ex: %ld, a4: %ld\n"
            "a5_ex: %ld, a5: %ld\n"
            "-----------------------\n",
            rank,
            a1_ex, a1,
            a2_ex, a2,
            a3_ex, a3,
            a4_ex, a4,
            a5_ex, a5
          );

    free(arr2);
    free(arr3);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

void recover() {
    //sprintf(fn, "Ckpt4-Rank%i.fti", rank);
    FILE *ckpt_file = fopen(fn, "rb");
    
    db_struct *currentdb, *nextdb;
    dbvar_struct *currentdbvar = NULL;
    char *dptr;
    int dbvar_idx, pvar_idx, dbcounter=0;
    char *zeros = (char*) calloc(1, dbstructsize); 

    long endoffile = 0, mdoffset;
    
    int isnextdb;
    
    currentdb = (db_struct*) malloc( sizeof(db_struct) );
    firstdb = currentdb;
    long read;

    do {
        
        nextdb = (db_struct*) malloc( sizeof(db_struct) );

        isnextdb = 0;
        
        mdoffset = endoffile;
        //printf("[RECOVER] db->mdoffset: %" PRIu64 "\n", mdoffset);
        
        fseek( ckpt_file, mdoffset, SEEK_SET );
        fread( currentdb, dbstructsize, 1, ckpt_file );
        mdoffset += dbstructsize;
        //printf("[RECOVER] db->numvars: %" PRIu32 " db->dbsize: %" PRIu64 "\n", currentdb->numvars, currentdb->dbsize);
        
        currentdb->dbvars = (dbvar_struct*) malloc( sizeof(dbvar_struct) * currentdb->numvars );
        
        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {
            
            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            fseek( ckpt_file, mdoffset, SEEK_SET );
            fread( currentdbvar, dbvarstructsize, 1, ckpt_file );
            ////printf("datablock-id: %i, var-id: %i, mdoffset: %" PRIu64 "\n",
            //        dbcounter, currentdbvar->id, mdoffset);
            mdoffset += dbvarstructsize;
            dptr =(void*)((uintptr_t)pvars[currentdbvar->idx].buffer + currentdbvar->dptr);
            //printf("[RECOVER] dbvar->id: %" PRIu32 " dbvar->fstart: %" PRIu32 ", dbvar->fend: %" PRIu64 ", write to buffer at 0x%" PRIxPTR " compare: 0x%" PRIxPTR "\n", 
            //        currentdbvar->id, currentdbvar->fptr, currentdbvar->fptr+currentdbvar->chunksize, pvars[currentdbvar->idx].buffer, dptr);
            fseek( ckpt_file, currentdbvar->fptr, SEEK_SET );
            
            read = fread( dptr, currentdbvar->chunksize, 1, ckpt_file );
            //printf("read: %ld\n", read);
        
        }
        
        endoffile += currentdb->dbsize;
        fseek( ckpt_file, endoffile, SEEK_SET );
        fread( nextdb, dbstructsize, 1, ckpt_file );
        
        //if ( memcmp( nextdb, zeros, dbstructsize ) != 0 ) {
        if ( !feof(ckpt_file) ) {
            currentdb->next = nextdb;
            nextdb->previous = currentdb;
            currentdb = nextdb;
            isnextdb = 1;
        }

        dbcounter++;
    
    } while( isnextdb );

    lastdb = currentdb;

}

void protect_var(int id, void *buffer, long size, long disp) {
    pvars = (pvar_struct*)realloc(pvars, (num_pvars+1)*sizeof(pvar_struct));
    pvars[num_pvars].buffer = buffer;
    //printf("[PROTECT] id: %i, buffer address: 0x%" PRIxPTR "\n", id, buffer);
    pvars[num_pvars].id = id;
    pvars[num_pvars].size = size;
    pvars[num_pvars].disp = disp;
    num_pvars++;
}

void edit_pvar(int id, void *buffer, long new_size) {
    int i;
    for(i=0;i<num_pvars;i++){
        if(pvars[i].id == id) {
            pvars[i].size = new_size;
            pvars[i].buffer = buffer;
        }
    }
}

void checkpoint() {
    
    //printf("< METADATA BEGIN >\n\n");
    update_metadata();
    //printf("\n< METADATA END >\n");

    //printf("< CHECKPOINT BEGIN >\n\n");
   
    FILE *ckpt_file;

    long mdoffset;
    long endoffile = 0;
    char *zeros = (char*) calloc(1, dbstructsize); 

    if(firstdb) {
        
        ckpt_file = fopen(fn, "wb+");
        db_struct *currentdb = firstdb;
        dbvar_struct *currentdbvar = NULL;
        char *dptr;
        int dbvar_idx, pvar_idx, dbcounter=0;

        int isnextdb;

        do {
            
            isnextdb = 0;
            
            mdoffset = endoffile;

            fseek( ckpt_file, mdoffset, SEEK_SET );
            fwrite( currentdb, dbstructsize, 1, ckpt_file );
            mdoffset += dbstructsize;
            
            for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {
                
                currentdbvar = &(currentdb->dbvars[dbvar_idx]);
                fseek( ckpt_file, mdoffset, SEEK_SET );
                fwrite( currentdbvar, dbvarstructsize, 1, ckpt_file );
                //printf("datablock-id: %i, var-id: %i, mdoffset: %" PRIu64 "\n",
                //        dbcounter, currentdbvar->id, mdoffset);
                mdoffset += dbvarstructsize;
                dptr = (char*)(pvars[currentdbvar->idx].buffer) + currentdb->dbvars[dbvar_idx].dptr;
                fseek( ckpt_file, currentdbvar->fptr, SEEK_SET );
                fwrite( dptr, currentdbvar->chunksize, 1, ckpt_file );
            
            }
            
            endoffile += currentdb->dbsize;
            
            if (currentdb->next) {
                currentdb = currentdb->next;
                isnextdb = 1;
            }

            dbcounter++;
        
        } while( isnextdb );

        //fseek( ckpt_file, endoffile, SEEK_SET );
        //fwrite( zeros, dbstructsize, 1, ckpt_file );
        fflush( ckpt_file );
        fclose( ckpt_file );

    }
    //printf("\n< CHECKPOINT END >\n");
}

void update_metadata() {
    
    int dbvar_idx, pvar_idx, num_edit_pvars = 0;
    int *editflags = (int*) calloc( num_pvars, sizeof(int) ); // 0 -> nothing changed, 1 -> new pvar, 2 -> size changed
    dbvar_struct *dbvars = NULL;
    int isnextdb;
    long offset = 0, chunksize;
    long *pvar_oldsizes, dbsize;
    // first call, init first datablock
    if(!firstdb) { // init file info
        dbsize = dbstructsize + dbvarstructsize * num_pvars;
        db_struct *dblock = (db_struct*) malloc( sizeof(db_struct) );
        dbvars = (dbvar_struct*) malloc( sizeof(dbvar_struct) * num_pvars );
        dblock->previous = NULL;
        dblock->next = NULL;
        dblock->numvars = num_pvars;
        dblock->dbvars = dbvars;
        for(dbvar_idx=0;dbvar_idx<dblock->numvars;dbvar_idx++) {
            dbvars[dbvar_idx].fptr = dbsize;
            dbvars[dbvar_idx].dptr = 0;
            dbvars[dbvar_idx].id = pvars[dbvar_idx].id;
            dbvars[dbvar_idx].idx = dbvar_idx;
            dbvars[dbvar_idx].chunksize = (pvars[dbvar_idx].size * pvars[dbvar_idx].disp);
            dbsize += dbvars[dbvar_idx].chunksize; 
            //printf("var-id: %i, fstart: %" PRIu64 " fend: %" PRIu64 ", dstart: %" PRIu64 " dend: %" PRIu64 "\n", 
            //        dbvars[dbvar_idx].id, dbvars[dbvar_idx].fptr, dbvars[dbvar_idx].fptr + dbvars[dbvar_idx].chunksize, 
            //        dbvars[dbvar_idx].dptr, dbvars[dbvar_idx].dptr + dbvars[dbvar_idx].chunksize);
        }
        num_pvars_old = num_pvars;
        dblock->dbsize = dbsize;
        
        // set as first datablock
        firstdb = dblock;
        lastdb = dblock;
    
    } else {
       
        /*
         *  - check if protected variable is in file info
         *  - check if size has changed
         */
        
        pvar_oldsizes = (long*) calloc( num_pvars_old, sizeof(long) );
        lastdb = firstdb;
        // iterate though datablock list
        do {
            isnextdb = 0;
            for(dbvar_idx=0;dbvar_idx<lastdb->numvars;dbvar_idx++) {
                for(pvar_idx=0;pvar_idx<num_pvars_old;pvar_idx++) {
                    if(lastdb->dbvars[dbvar_idx].id == pvars[pvar_idx].id) {
                        chunksize = lastdb->dbvars[dbvar_idx].chunksize;
                        pvar_oldsizes[pvar_idx] += chunksize;
                    }
                }
            }
            offset += lastdb->dbsize;
            if (lastdb->next) {
                lastdb = lastdb->next;
                isnextdb = 1;
            }
        } while( isnextdb );
        
        //printf("offset: %" PRIu64 "\n", offset);

        // check for new protected variables
        for(pvar_idx=num_pvars_old;pvar_idx<num_pvars;pvar_idx++) {
            editflags[pvar_idx] = 1;
            num_edit_pvars++;
        }
        
        // check if size changed
        for(pvar_idx=0;pvar_idx<num_pvars_old;pvar_idx++) {
            if(pvar_oldsizes[pvar_idx] != (pvars[pvar_idx].size*pvars[pvar_idx].disp)) {
                editflags[pvar_idx] = 2;
                num_edit_pvars++;
            }
        }
                
        // check for edit flags
        for(pvar_idx=0;pvar_idx<num_pvars;pvar_idx++) {
            //printf("%i,",editflags[pvar_idx]);
        }
        //printf("\n");

        // if size changed or we have new variables to protect, create new block.
        
        dbsize = dbstructsize + dbvarstructsize * num_edit_pvars;
       
        int evar_idx = 0;
        if( num_edit_pvars ) {
            for(pvar_idx=0; pvar_idx<num_pvars; pvar_idx++) {
                switch(editflags[pvar_idx]) {

                    case 1:
                        // add new protected variable in next datablock
                        dbvars = (dbvar_struct*) realloc( dbvars, (evar_idx+1) * sizeof(dbvar_struct) );
                        dbvars[evar_idx].fptr = offset + dbsize;
                        dbvars[evar_idx].dptr = 0;
                        dbvars[evar_idx].id = pvars[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbvars[evar_idx].chunksize = (pvars[pvar_idx].size * pvars[pvar_idx].disp);
                        dbsize += dbvars[evar_idx].chunksize; 
                        //printf("var-id: %i, fstart: %" PRIu64 " fend: %" PRIu64 ", dstart: %" PRIu64 " dend: %" PRIu64 "\n", 
                        //        dbvars[evar_idx].id, dbvars[evar_idx].fptr, dbvars[evar_idx].fptr + dbvars[evar_idx].chunksize, 
                        //        dbvars[evar_idx].dptr, dbvars[evar_idx].dptr + dbvars[evar_idx].chunksize);
                        evar_idx++;

                        break;

                    case 2:
                        
                        // create data chunk info
                        dbvars = (dbvar_struct*) realloc( dbvars, (evar_idx+1) * sizeof(dbvar_struct) );
                        dbvars[evar_idx].fptr = offset + dbsize;
                        dbvars[evar_idx].dptr = pvar_oldsizes[pvar_idx];
                        dbvars[evar_idx].id = pvars[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbvars[evar_idx].chunksize = pvars[pvar_idx].size * pvars[pvar_idx].disp - pvar_oldsizes[pvar_idx];
                        dbsize += dbvars[evar_idx].chunksize; 
                        //printf("var-id: %i, fstart: %" PRIu64 " fend: %" PRIu64 ", dstart: %" PRIu64 " dend: %" PRIu64 "\n", 
                        //        dbvars[evar_idx].id, dbvars[evar_idx].fptr, dbvars[evar_idx].fptr + dbvars[evar_idx].chunksize, 
                        //        dbvars[evar_idx].dptr, dbvars[evar_idx].dptr + dbvars[evar_idx].chunksize);
                        evar_idx++;

                        break;

                }

            }

            db_struct  *dblock = (db_struct*) malloc( sizeof(db_struct) );
            lastdb->next = dblock;
            dblock->previous = lastdb;
            dblock->next = NULL;
            dblock->numvars = num_edit_pvars;
            dblock->dbsize = dbsize;
            dblock->dbvars = dbvars;
            lastdb = dblock;
        
        }

        num_pvars_old = num_pvars;
        
        free(pvar_oldsizes);
    
    }

    free(editflags);

}
    
void finalize() {

    int ispreviousdb, dbvar_idx, pvar_idx, counter=0;

    if(firstdb) {
        do {
            ispreviousdb = 0;
            //printf("%i\n",counter);
            counter++;
            free(lastdb->dbvars);

            if (lastdb->previous) {
                lastdb = lastdb->previous;
                free(lastdb->next);
                ispreviousdb = 1;
            } else {
                free(lastdb);
            }
        } while( ispreviousdb );
    }

    //if (pvars) {
    //    free(pvars);
    //}


}
                



        
        
        

















