#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

// INSTEAD OF FLAG IN DATABLOCK, WRITE ONLY ZEROS IN THE END

/*
 * declarations of structures and functions
 */

typedef struct pvar_struct_raw {
    uint32_t id;
    void *buffer;
    uint64_t disp;
    uint64_t size;
} pvar_struct; // protected variable

typedef struct dbvar_struct_raw {
    uint32_t id;
    uint32_t idx;   // index to corresponding id in pvar array
    uint64_t dptr;  // data pointer
    uint64_t fptr;  // file pointer
    uint64_t chunksize;
} dbvar_struct; // defines variable chunk in datablock

typedef struct db_struct_raw {
    uint32_t numvars;
    uint64_t dbsize;
    dbvar_struct *dbvars;
    struct db_struct_raw *previous;
    struct db_struct_raw *next;
} db_struct; // defines datablock

void update_metadata();
void finalize();
void checkpoint(); // perform checkpoint
void protect_var(uint32_t id, void *buffer, uint64_t size, uint64_t disp); // protect variable
void edit_pvar(uint32_t id, uint64_t new_size); // edit protect variable

/*
 * global variables
 */

// meta data of protected variables 
pvar_struct *pvars = NULL;
int num_pvars = 0;
int num_pvars_old = 0;

// datablock size in file
const uint8_t dbstructsize 
    = sizeof(uint32_t)  /* numvars */ 
    + sizeof(uint64_t); /* dbsize */ 

// var info element size in file
const uint8_t dbvarstructsize 
    = sizeof(uint32_t)  /* id */ 
    + sizeof(uint32_t)  /* idx */ 
    + sizeof(uint64_t)  /* dptr */ 
    + sizeof(uint64_t)  /* fptr */ 
    + sizeof(uint64_t); /* chunksize */ 

// init datablock list
db_struct *firstdb = NULL;
db_struct *lastdb = NULL;

int main() {
    
    int arr1[100];
    int *arr2 = (int*) malloc( 200 * sizeof(int) );
    int *arr3 = (int*) malloc( 300 * sizeof(int) );
    int arr4[400];
    int arr5[700];

    protect_var(1, arr1, 100, sizeof(int));  
    protect_var(2, arr2, 200, sizeof(int));  
    protect_var(3, arr3, 300, sizeof(int)); 
    checkpoint();
    protect_var(4, arr4, 400, sizeof(int)); 
    checkpoint();

    arr2 = (int*) realloc(arr2, sizeof(int) * 500);
    arr3 = (int*) realloc(arr3, sizeof(int) * 600);
    edit_pvar(2, 500);
    edit_pvar(3, 600);
    checkpoint();
    protect_var(5, arr5, 700, sizeof(int)); 
    checkpoint();

    finalize();

    free(arr2);
    free(arr3);
}

void protect_var(uint32_t id, void *buffer, uint64_t size, uint64_t disp) {
    pvars = (pvar_struct*)realloc(pvars, (num_pvars+1)*sizeof(pvar_struct));
    pvars[num_pvars].buffer = buffer;
    pvars[num_pvars].id = id;
    pvars[num_pvars].size = size;
    pvars[num_pvars].disp = disp;
    num_pvars++;
}

void edit_pvar(uint32_t id, uint64_t new_size) {
    int i;
    for(i=0;i<num_pvars;i++){
        if(pvars[i].id == id) {
            pvars[i].size = new_size;
        }
    }
}

void checkpoint() {
    
    printf("< METADATA BEGIN >\n\n");
    update_metadata();
    printf("\n< METADATA END >\n");

    printf("< CHECKPOINT BEGIN >\n\n");
   
    FILE *ckpt_file = fopen("test.ckpt", "wb+");

    uint64_t mdoffset = 0;
    if(firstdb) {

        db_struct *currentdb = firstdb;
        dbvar_struct *currentdbvar = NULL;
        char *dptr;
        int dbvar_idx, pvar_idx, dbcounter=0;

        uint8_t isnextdb;

        do {
            
            isnextdb = 0;
            
            fseek( ckpt_file, mdoffset, SEEK_SET );
            fwrite( currentdb, dbstructsize, 1, ckpt_file );
            mdoffset += dbstructsize;
            
            for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {
                
                currentdbvar = &(currentdb->dbvars[dbvar_idx]);
                fseek( ckpt_file, mdoffset, SEEK_SET );
                fwrite( currentdbvar, dbvarstructsize, 1, ckpt_file );
                printf("datablock-id: %i, var-id: %i, mdoffset: %" PRIu64 "\n",
                        dbcounter, currentdbvar->id, mdoffset);
                mdoffset += dbvarstructsize;
                dptr = (char*)&(pvars[currentdbvar->idx]) + currentdb->dbvars[dbvar_idx].dptr;
                fseek( ckpt_file, currentdbvar->fptr, SEEK_SET );
                fwrite( dptr, currentdbvar->chunksize, 1, ckpt_file );
            
            }
            
            if (currentdb->next) {
                currentdb = currentdb->next;
                isnextdb = 1;
            }

            dbcounter++;
        
        } while( isnextdb );
    
    }
    printf("\n< CHECKPOINT END >\n");
}

void update_metadata() {
    
    int dbvar_idx, pvar_idx, num_edit_pvars = 0;
    int *editflags = (int*) calloc( num_pvars, sizeof(int) ); // 0 -> nothing changed, 1 -> new pvar, 2 -> size changed
    dbvar_struct *dbvars = NULL;
    uint8_t isnextdb;
    uint64_t offset = 0, chunksize;
    uint64_t *pvar_oldsizes, dbsize;
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
            printf("var-id: %i, fstart: %" PRIu64 " fend: %" PRIu64 ", dstart: %" PRIu64 " dend: %" PRIu64 "\n", 
                    dbvars[dbvar_idx].id, dbvars[dbvar_idx].fptr, dbvars[dbvar_idx].fptr + dbvars[dbvar_idx].chunksize, 
                    dbvars[dbvar_idx].dptr, dbvars[dbvar_idx].dptr + dbvars[dbvar_idx].chunksize);
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
        
        pvar_oldsizes = (uint64_t*) calloc( num_pvars_old, sizeof(uint64_t) );
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
        
        printf("offset: %" PRIu64 "\n", offset);

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
            printf("%i,",editflags[pvar_idx]);
        }
        printf("\n");

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
                        printf("var-id: %i, fstart: %" PRIu64 " fend: %" PRIu64 ", dstart: %" PRIu64 " dend: %" PRIu64 "\n", 
                                dbvars[evar_idx].id, dbvars[evar_idx].fptr, dbvars[evar_idx].fptr + dbvars[evar_idx].chunksize, 
                                dbvars[evar_idx].dptr, dbvars[evar_idx].dptr + dbvars[evar_idx].chunksize);
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
                        printf("var-id: %i, fstart: %" PRIu64 " fend: %" PRIu64 ", dstart: %" PRIu64 " dend: %" PRIu64 "\n", 
                                dbvars[evar_idx].id, dbvars[evar_idx].fptr, dbvars[evar_idx].fptr + dbvars[evar_idx].chunksize, 
                                dbvars[evar_idx].dptr, dbvars[evar_idx].dptr + dbvars[evar_idx].chunksize);
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
            printf("%i\n",counter);
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

    if (pvars) {
        free(pvars);
    }


}
                



        
        
        

















