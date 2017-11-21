#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

// INSTEAD OF FLAG IN DATABLOCK, WRITE ONLY ZEROS IN THE END

/*
 * declarations of structures and functions
 */

typedef struct pvar_struct_raw {
    uint32_t id;
    uint64_t disp;
    uint64_t size;
} pvar_struct; // protected variable

typedef struct dbvar_struct_raw {
    uint32_t id;
    uint32_t idx;
    uint64_t start;
    uint64_t end;
} dbvar_struct; // defines variable chunk in datablock

typedef struct db_struct_raw {
    struct db_struct_raw *previous;
    struct db_struct_raw *next;
    uint32_t numvars;
    uint64_t dbsize;
    dbvar_struct *dbvars;
} db_struct; // defines datablock

void finalize();
void checkpoint(); // perform checkpoint
void protect_var(uint32_t id, uint64_t size, uint64_t disp); // protect variable
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
    + sizeof(uint64_t)  /* start */ 
    + sizeof(uint64_t); /* end */ 

// init datablock list
db_struct *firstdb = NULL;
db_struct *lastdb = NULL;

int main() {
    protect_var(1, 100, 8);  
    protect_var(2, 200, 8);  
    protect_var(3, 300, 8); 
    checkpoint();
    protect_var(4, 400, 8); 
    checkpoint();
    edit_pvar(2, 500);
    edit_pvar(3, 600);
    checkpoint();
    protect_var(5, 700, 8); 
    checkpoint();

    finalize();
}

void protect_var(uint32_t id, uint64_t size, uint64_t disp) {
    pvars = (pvar_struct*)realloc(pvars, (num_pvars+1)*sizeof(pvar_struct));
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
    printf("< CHECKPOINT BEGIN >\n\n");
    // variable definitions
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
            dbvars[dbvar_idx].start = dbsize;
            dbvars[dbvar_idx].id = pvars[dbvar_idx].id;
            dbvars[dbvar_idx].idx = dbvar_idx;
            dbsize += pvars[dbvar_idx].size;
            dbvars[dbvar_idx].end = dbsize;
            printf("var-id: %dbvar_idx, start: %" PRIu64 " end: %" PRIu64 "\n", dbvars[dbvar_idx].id, dbvars[dbvar_idx].start, dbvars[dbvar_idx].end);
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
                        chunksize = lastdb->dbvars[dbvar_idx].end - lastdb->dbvars[dbvar_idx].start;
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
            if(pvar_oldsizes[pvar_idx] != pvars[pvar_idx].size) {
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
                        dbvars[evar_idx].start = offset + dbsize;
                        dbvars[evar_idx].id = pvars[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbsize += pvars[pvar_idx].size;
                        dbvars[evar_idx].end = offset + dbsize;
                        printf("var-id: %i, start: %" PRIu64 " end: %" PRIu64 "\n", 
                                dbvars[evar_idx].id, dbvars[evar_idx].start, dbvars[evar_idx].end);
                        evar_idx++;

                        break;

                    case 2:
                        
                        // create data chunk info
                        dbvars = (dbvar_struct*) realloc( dbvars, (evar_idx+1) * sizeof(dbvar_struct) );
                        dbvars[evar_idx].start = offset + dbsize;
                        dbvars[evar_idx].id = pvars[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbsize += pvars[pvar_idx].size - pvar_oldsizes[pvar_idx];
                        dbvars[evar_idx].end = offset + dbsize;
                        printf("var-id: %i, start: %" PRIu64 " end: %" PRIu64 "\n", 
                                dbvars[evar_idx].id, dbvars[evar_idx].start, dbvars[evar_idx].end);
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
    
    printf("\n< CHECKPOINT END >\n");
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
                



        
        
        

















