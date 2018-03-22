\page ftiff FTI File Format (FTI-FF)
## What for?
FTI writes meta data related to the checkpoint files in separate text files. The meta data is needed to perform a restart from the last checkpoint. FTI-FF includes the meta data in the checkpoint files, thus reduces the amount of files on the PFS. This can be beneficial for executions with several thousands of processes.

But it can also be interesting for restarting from others then the last checkpoint files. The current implementation (V1.1) does not implement this in an automatic fashion, hence the checkpoint files must be copied by hand in order to be available for an alternative restart.

The motivation, however, for the new file format is to facilitate the future implementation of differential and incremental checkpointing. FTI-FF provides information within the file where every chunk of data is stored, thus enables to write changed data blocks to the corresponding position. 

## Structure

The file format basic structure, consists of a meta block (`FB`) and a data block (`VB`):
  
```
    +--------------+ +------------------------+
    |              | |                        |
    | FB           | | VB                     |
    |              | |                        |
    +--------------+ +------------------------+
```
  
The `FB` (file block) holds meta data related to the file whereas the `VB` (variable block) holds meta and actual data of the variables protected by FTI.
  
The `FB` has the following structure (`FTIFF_metaInfo`):
  
```C
    typedef struct FTIFF_metaInfo {                    
        char checksum[MD5_DIGEST_STRING_LENGTH];   // Hash of the VB block in hex representation (33 bytes)        
        unsigned char myHash[MD5_DIGEST_LENGTH];   // Hash of FB without 'myHash' in unsigned char (16 bytes)                 
        long ckptSize;                             // Size of actual data stored in file                                   
        long fs;                                   // Size of FB + VB                                               
        long maxFs;                                // Maximum size of FB + VB in group
        long ptFs;                                 // Size of FB + VB of partner process                        
        long timestamp;                            // Time in ns of FB block creation                       
    } FTIFF_metaInfo;
```

The `VB` contains the sub structures `VCB_i` (variable chunk blocks), which consist of the variable chunks (`VC_ij`) stored in the current `VCB_i` and the corresponding variable chunk  meta data (`VMB_i`):
  
```
    |<-------------------------------------------------- VB -------------------------------------------------->|
    #                                                                                                          #
    |<------------- VCB_1 --------------------------->|      |<------------- VCB_n --------------------------->|
    #                                                 #      #                                                 #       
    +-------------------------------------------------+      +-------------------------------------------------+
    | +--------++-------------+      +--------------+ |      | +--------++-------------+      +--------------+ |
    | |        ||             |      |              | |      | |        ||             |      |              | |
    | | VMB_1  || VC_11       | ---- | VC_1k        | | ---- | | VMB_n  || VC_n1       | ---- | VC_nl        | |
    | |        ||             |      |              | |      | |        ||             |      |              | |
    | +--------++-------------+      +--------------+ |      | +--------++-------------+      +--------------+ |
    +-------------------------------------------------+      +-------------------------------------------------+
```

The number of data chunks (e.g. `k` and `l` in the sketch), generally differs. To which protected variable the chunk `VC_ij` belongs is kept in the corresponding data structure (`FTIFF_dbvar`) which is part of the `VMB_i`.   

The `VMB_i` have the following sub structure:
  
```
   |<-------------- VMB_i ------------->|
   #                                    #
   +-------++---------+      +----------+
   |       ||         |      |          |
   | BMD_i || VMD_i1  | ---- | VMD_ij   |
   |       ||         |      |          |
   +-------++---------+      +----------+
```
  
Where the `BMD_i` (block meta data) keep information related to the variable chunk block and possess the following structure (`FTIFF_db`):
  
```C
    typedef struct FTIFF_db {
        int numvars;                               // Size of entire block VCB_i (meta + actual data)       
        long dbsize;                               // Number of variable chunks in data block        
        FTIFF_dbvar *dbvars;                       // pointer to related VMD_ij array
        struct FTIFF_db *previous;                 // link to BMD_(i-1) or NULL if first block (FTI_Exec->firstdb)
        struct FTIFF_db *next;                     // link to BMD_(i+1) or NULL if last block (FTI_Exec->lastdb)
    } FTIFF_db;
```
  
The `VMD_ij` have the following structure (`FTIFF_dbvar`):
  
```C
    typedef struct FTIFF_dbvar {
        int id;                                   // Id of protected variable the data chunk belongs to                             
        int idx;                                   // Index of corresponding element in FTI_Data      
        int containerid;                           // Id of this container                                   
        bool hascontent;                           // Boolean value indicating if container holds data or not                                    
        long dptr;                                 // offset of chunk in runtime-data (i.e. virtual address ptr = FTI_Data[idx].ptr + dptr)                                               
        long fptr;                                 // offset of chunk in file                                                           
        long chunksize;                            // Size of chunk stored in container                                                   
        long containersize;                        // Total size of container                                                           
        unsigned char hash[MD5_DIGEST_LENGTH];     // Hash of variable chunk 'VC_ij'                                                                      
    } FTIFF_dbvar;
```

## Example

The container size is fixed once an additional container is created. A container is created if the size of a protected variable increases between two invocations of `FTI_Checkpoint()` and if between invocations an additional variable is protected by a call to `FTI_Protect()`.
  
The following example shows the status of the data structures just before the invocations of `FTI_Checkpoint()`. To achieve the correct values for `fptr` one has to consider the size of the `FB` part at the beginning of the file. Thus the base offset is `sizeof(FTIFF_infoMeta)` which is in the example here equal to 96 bytes.

### Checkpoint 1

```C
    size1=1000000;
    size2=2000000;
    size3=3000000;

    FTI_Protect(1, arr1, size1, FTI_INTG);  
    FTI_Protect(2, arr2, size2, FTI_INTG);  
    FTI_Protect(3, arr3, size3, FTI_INTG); 

    FTI_Checkpoint(1,level);
```

Output:

```
------------------- DATASTRUCTURE BEGIN -------------------

    DataBase-id: 0
                 numvars: 3
                 dbsize: 24000204
                 *dbvars: 0xe4b4b0
                 *previous: (nil)
                 *next: (nil)
                 [size metadata: 204]

         Var-id: 0
                 id: 1
                 idx: 0
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 300
                 chunksize: 4000000
                 containersize: 4000000
                 [hash not printed]

         Var-id: 1
                 id: 2
                 idx: 1
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 4000300
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 2
                 id: 3
                 idx: 2
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 12000300
                 chunksize: 12000000
                 containersize: 12000000
                 [hash not printed]


------------------- DATASTRUCTURE END ---------------------
```

### Checkpoint 2

```C
    size4=4000000;

    FTI_Protect(4, arr4, size4, FTI_INTG); 

    FTI_Checkpoint(2,level);
```

Output:

```
------------------- DATASTRUCTURE BEGIN -------------------

    DataBase-id: 0
                 numvars: 3
                 dbsize: 24000204
                 *dbvars: 0xe4b4b0
                 *previous: (nil)
                 *next: 0xe78f30
                 [size metadata: 204]

         Var-id: 0
                 id: 1
                 idx: 0
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 300
                 chunksize: 4000000
                 containersize: 4000000
                 [hash not printed]

         Var-id: 1
                 id: 2
                 idx: 1
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 4000300
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 2
                 id: 3
                 idx: 2
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 12000300
                 chunksize: 12000000
                 containersize: 12000000
                 [hash not printed]

    DataBase-id: 1
                 numvars: 1
                 dbsize: 16000076
                 *dbvars: 0xe77470
                 *previous: 0xe4fea0
                 *next: (nil)
                 [size metadata: 76]

         Var-id: 0
                 id: 4
                 idx: 3
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 24000376
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]


------------------- DATASTRUCTURE END ---------------------
```

### Checkpoint 3

```C
    size2=6000000;
    size3=7000000;

    FTI_Protect(2, arr2, size2, FTI_INTG);  
    FTI_Protect(3, arr3, size3, FTI_INTG); 

    FTI_Checkpoint(3,level);
```

Output:

```
------------------- DATASTRUCTURE BEGIN -------------------

    DataBase-id: 0
                 numvars: 3
                 dbsize: 24000204
                 *dbvars: 0xe4b4b0
                 *previous: (nil)
                 *next: 0xe78f30
                 [size metadata: 204]

         Var-id: 0
                 id: 1
                 idx: 0
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 300
                 chunksize: 4000000
                 containersize: 4000000
                 [hash not printed]

         Var-id: 1
                 id: 2
                 idx: 1
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 4000300
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 2
                 id: 3
                 idx: 2
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 12000300
                 chunksize: 12000000
                 containersize: 12000000
                 [hash not printed]

    DataBase-id: 1
                 numvars: 1
                 dbsize: 16000076
                 *dbvars: 0xe77470
                 *previous: 0xe4fea0
                 *next: 0xe50620
                 [size metadata: 76]

         Var-id: 0
                 id: 4
                 idx: 3
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 24000376
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 2
                 numvars: 2
                 dbsize: 32000140
                 *dbvars: 0xe79260
                 *previous: 0xe78f30
                 *next: (nil)
                 [size metadata: 140]

         Var-id: 0
                 id: 2
                 idx: 1
                 containerid: 1
                 hascontent: true
                 dptr: 8000000
                 fptr: 40000516
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

         Var-id: 1
                 id: 3
                 idx: 2
                 containerid: 1
                 hascontent: true
                 dptr: 12000000
                 fptr: 56000516
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]


------------------- DATASTRUCTURE END ---------------------
```

### Checkpoint 4

```C
    size5=5000000;

    FTI_Protect(5, arr5, size5, FTI_INTG); 

    FTI_Checkpoint(4,level);
```

Output:

```
------------------- DATASTRUCTURE BEGIN -------------------

    DataBase-id: 0
                 numvars: 3
                 dbsize: 24000204
                 *dbvars: 0xe4b4b0
                 *previous: (nil)
                 *next: 0xe78f30
                 [size metadata: 204]

         Var-id: 0
                 id: 1
                 idx: 0
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 300
                 chunksize: 4000000
                 containersize: 4000000
                 [hash not printed]

         Var-id: 1
                 id: 2
                 idx: 1
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 4000300
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 2
                 id: 3
                 idx: 2
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 12000300
                 chunksize: 12000000
                 containersize: 12000000
                 [hash not printed]

    DataBase-id: 1
                 numvars: 1
                 dbsize: 16000076
                 *dbvars: 0xe77470
                 *previous: 0xe4fea0
                 *next: 0xe50620
                 [size metadata: 76]

         Var-id: 0
                 id: 4
                 idx: 3
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 24000376
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 2
                 numvars: 2
                 dbsize: 32000140
                 *dbvars: 0xe79260
                 *previous: 0xe78f30
                 *next: 0xe79230
                 [size metadata: 140]

         Var-id: 0
                 id: 2
                 idx: 1
                 containerid: 1
                 hascontent: true
                 dptr: 8000000
                 fptr: 40000516
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

         Var-id: 1
                 id: 3
                 idx: 2
                 containerid: 1
                 hascontent: true
                 dptr: 12000000
                 fptr: 56000516
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 3
                 numvars: 1
                 dbsize: 20000076
                 *dbvars: 0xe79170
                 *previous: 0xe50620
                 *next: (nil)
                 [size metadata: 76]

         Var-id: 0
                 id: 5
                 idx: 4
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 72000592
                 chunksize: 20000000
                 containersize: 20000000
                 [hash not printed]


------------------- DATASTRUCTURE END ---------------------
```

### Checkpoint 5

```C
    size2=5000000;
    size3=6000000;

    FTI_Protect(2, arr2, size2, FTI_INTG);  
    FTI_Protect(3, arr3, size3, FTI_INTG); 

    FTI_Checkpoint(1,level);
```

Output:

```
------------------- DATASTRUCTURE BEGIN -------------------

    DataBase-id: 0
                 numvars: 3
                 dbsize: 24000204
                 *dbvars: 0xe734b0
                 *previous: (nil)
                 *next: 0xea0f30
                 [size metadata: 204]

         Var-id: 0
                 id: 1
                 idx: 0
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 300
                 chunksize: 4000000
                 containersize: 4000000
                 [hash not printed]

         Var-id: 1
                 id: 2
                 idx: 1
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 4000300
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 2
                 id: 3
                 idx: 2
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 12000300
                 chunksize: 12000000
                 containersize: 12000000
                 [hash not printed]

    DataBase-id: 1
                 numvars: 1
                 dbsize: 16000076
                 *dbvars: 0xe9f470
                 *previous: 0xe77ea0
                 *next: 0xe78620
                 [size metadata: 76]

         Var-id: 0
                 id: 4
                 idx: 3
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 24000376
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 2
                 numvars: 2
                 dbsize: 32000140
                 *dbvars: 0xea1260
                 *previous: 0xea0f30
                 *next: 0xea1230
                 [size metadata: 140]

         Var-id: 0
                 id: 2
                 idx: 1
                 containerid: 1
                 hascontent: true
                 dptr: 8000000
                 fptr: 40000516
                 chunksize: 12000000
                 containersize: 16000000
                 [hash not printed]

         Var-id: 1
                 id: 3
                 idx: 2
                 containerid: 1
                 hascontent: true
                 dptr: 12000000
                 fptr: 56000516
                 chunksize: 12000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 3
                 numvars: 1
                 dbsize: 20000076
                 *dbvars: 0xea1170
                 *previous: 0xe78620
                 *next: (nil)
                 [size metadata: 76]

         Var-id: 0
                 id: 5
                 idx: 4
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 72000592
                 chunksize: 20000000
                 containersize: 20000000
                 [hash not printed]


------------------- DATASTRUCTURE END ---------------------

```

### Checkpoint 6

```C
    size2=8000000;
    size3=9000000;

    FTI_Protect(2, arr2, size2, FTI_INTG);  
    FTI_Protect(3, arr3, size3, FTI_INTG); 

    FTI_Checkpoint(6,level);
```

Output:

```
------------------- DATASTRUCTURE BEGIN -------------------

    DataBase-id: 0
                 numvars: 3
                 dbsize: 24000204
                 *dbvars: 0xe734b0
                 *previous: (nil)
                 *next: 0xea0f30
                 [size metadata: 204]

         Var-id: 0
                 id: 1
                 idx: 0
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 300
                 chunksize: 4000000
                 containersize: 4000000
                 [hash not printed]

         Var-id: 1
                 id: 2
                 idx: 1
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 4000300
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 2
                 id: 3
                 idx: 2
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 12000300
                 chunksize: 12000000
                 containersize: 12000000
                 [hash not printed]

    DataBase-id: 1
                 numvars: 1
                 dbsize: 16000076
                 *dbvars: 0xe9f470
                 *previous: 0xe77ea0
                 *next: 0xe78620
                 [size metadata: 76]

         Var-id: 0
                 id: 4
                 idx: 3
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 24000376
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 2
                 numvars: 2
                 dbsize: 32000140
                 *dbvars: 0xea1260
                 *previous: 0xea0f30
                 *next: 0xea1230
                 [size metadata: 140]

         Var-id: 0
                 id: 2
                 idx: 1
                 containerid: 1
                 hascontent: true
                 dptr: 8000000
                 fptr: 40000516
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

         Var-id: 1
                 id: 3
                 idx: 2
                 containerid: 1
                 hascontent: true
                 dptr: 12000000
                 fptr: 56000516
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 3
                 numvars: 1
                 dbsize: 20000076
                 *dbvars: 0xea1170
                 *previous: 0xe78620
                 *next: 0xea0cf0
                 [size metadata: 76]

         Var-id: 0
                 id: 5
                 idx: 4
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 72000592
                 chunksize: 20000000
                 containersize: 20000000
                 [hash not printed]

    DataBase-id: 4
                 numvars: 2
                 dbsize: 16000140
                 *dbvars: 0xea0dc0
                 *previous: 0xea1230
                 *next: (nil)
                 [size metadata: 140]

         Var-id: 0
                 id: 2
                 idx: 1
                 containerid: 2
                 hascontent: true
                 dptr: 24000000
                 fptr: 92000732
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 1
                 id: 3
                 idx: 2
                 containerid: 2
                 hascontent: true
                 dptr: 28000000
                 fptr: 100000732
                 chunksize: 8000000
                 containersize: 8000000
                 [hash not printed]


------------------- DATASTRUCTURE END ---------------------

```

### Checkpoint 7

```C
    size2 = 1000000;
    size3 = 2000000;

    FTI_Protect(2, arr2, size2, FTI_INTG);  
    FTI_Protect(3, arr3, size3, FTI_INTG); 

    FTI_Checkpoint(7,level);
```

Output:

```
------------------- DATASTRUCTURE BEGIN -------------------

    DataBase-id: 0
                 numvars: 3
                 dbsize: 24000204
                 *dbvars: 0xe734b0
                 *previous: (nil)
                 *next: 0xea0f30
                 [size metadata: 204]

         Var-id: 0
                 id: 1
                 idx: 0
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 300
                 chunksize: 4000000
                 containersize: 4000000
                 [hash not printed]

         Var-id: 1
                 id: 2
                 idx: 1
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 4000300
                 chunksize: 4000000
                 containersize: 8000000
                 [hash not printed]

         Var-id: 2
                 id: 3
                 idx: 2
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 12000300
                 chunksize: 8000000
                 containersize: 12000000
                 [hash not printed]

    DataBase-id: 1
                 numvars: 1
                 dbsize: 16000076
                 *dbvars: 0xe9f470
                 *previous: 0xe77ea0
                 *next: 0xe78620
                 [size metadata: 76]

         Var-id: 0
                 id: 4
                 idx: 3
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 24000376
                 chunksize: 16000000
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 2
                 numvars: 2
                 dbsize: 32000140
                 *dbvars: 0xea1260
                 *previous: 0xea0f30
                 *next: 0xea1230
                 [size metadata: 140]

         Var-id: 0
                 id: 2
                 idx: 1
                 containerid: 1
                 hascontent: false
                 dptr: 8000000
                 fptr: 40000516
                 chunksize: 0
                 containersize: 16000000
                 [hash not printed]

         Var-id: 1
                 id: 3
                 idx: 2
                 containerid: 1
                 hascontent: false
                 dptr: 12000000
                 fptr: 56000516
                 chunksize: 0
                 containersize: 16000000
                 [hash not printed]

    DataBase-id: 3
                 numvars: 1
                 dbsize: 20000076
                 *dbvars: 0xea1170
                 *previous: 0xe78620
                 *next: 0xea0cf0
                 [size metadata: 76]

         Var-id: 0
                 id: 5
                 idx: 4
                 containerid: 0
                 hascontent: true
                 dptr: 0
                 fptr: 72000592
                 chunksize: 20000000
                 containersize: 20000000
                 [hash not printed]

    DataBase-id: 4
                 numvars: 2
                 dbsize: 16000140
                 *dbvars: 0xea0dc0
                 *previous: 0xea1230
                 *next: (nil)
                 [size metadata: 140]

         Var-id: 0
                 id: 2
                 idx: 1
                 containerid: 2
                 hascontent: false
                 dptr: 24000000
                 fptr: 92000732
                 chunksize: 0
                 containersize: 8000000
                 [hash not printed]

         Var-id: 1
                 id: 3
                 idx: 2
                 containerid: 2
                 hascontent: false
                 dptr: 28000000
                 fptr: 100000732
                 chunksize: 0
                 containersize: 8000000
                 [hash not printed]


------------------- DATASTRUCTURE END ---------------------
```
