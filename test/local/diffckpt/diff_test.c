#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <fti.h>

typedef struct _dcp_info dcp_info_t;

/*
 * init a random amount of buffers with random data.
 * Allocate not more then 'alloc_size' in bytes.
*/
void init_buffers( dcp_info_t * info );

/*
 * change 'share' percentage (integer) of data in buffer with 'id' 
 * and return size of changed data in bytes.
*/
unsigned long change_data( int share, dcp_info_t *info, int id );

int main() {

}
