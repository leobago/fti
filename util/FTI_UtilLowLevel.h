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
 *  @file   FTI_UtiLowLevel.c
 *  @author konstantinos Parasyris (koparasy)
 *  @date   23 January, 2020
 *  @brief  FTI API for lists.
 */


#ifndef __TOOL_API__
#define __TOOL_API__

#define BUFF_SIZE 1000
#define MAX_BUFF (16*1024*1024)
#define MD5_DIGEST_STRING_LENGTH 33
#define MD5_DIGEST_LENGTH 16


    /** @typedef   FTIInfo 
     *  @brief     Information of the config file passed as argument on fti.
     *  
     * This structure contains all necessary information of the configuration file
     *  of FTI.
     *  
     */

typedef struct FTI_Info{
    char *execDir;              /**< This is the path in which I started executing my application*/
    char *configFileDir;        /**< Where my configuration path is */
    char *configName;           /**< Configuration file name*/

    char *metaDir;              /**< Where the metadata dir is*/
    char *globalDir;            /**< Where the global dir is*/

    char *execId;               /**< execution id*/

    int localTest;              /**<Was this a local or a cluster rank      */
    int groupSize;              /**< Size of the group                      */
    int head;                   /**< Did I use FTI managers                 */
    int nodeSize;               /**< Number of ranks in node                */
    int userRanks;              /**< Number of ranks executing user code    */
    int numCheckpoints;         /**< Number of checkpoints in this execution*/

}FTIInfo;

    /** @typedef  FTIDataVar 
     *  @brief     Contains all information of a checkpointed variable.
     *  
     * This structure contains all necessary information of a checkpointed 
     *  variable.
     *  
     */
typedef struct FTI_DataVar{
    char *name;                 /**< Name of the variable (given by user)                   */
    int id;                     /**< Id of the variable (given by user)                     */
    size_t size;                /**< size of the variable                                   */
    size_t pos;                 /**< where variable is stored in ckpt file                  */
    unsigned char *buf;         /**< Data of the variable                                   */
}FTIDataVar;

    /** @typedef  FTIDataVar 
     *  @brief     Contains all information of a checkpoint file
     *  
     * This structure contains all necessary information of a checkpoint 
     * file.
     *  
     */
typedef struct FTI_ckptFile{
    char *name;                 /**< Name of the checkpoint file                             */
    char *md5hash;              /**< MD5 hash of the ckpt file                               */
    FTIDataVar *variables;     /**< pointer to variables stored in ckpt file                */
    int numVars;                /**< Number of variables in ckpt file                        */
    int globalRank;             /**< Which rank wrote this ckpt file in original exec.       */
    int applicationRank;        /**< Rank in FTI_COMM_WORLD                                  */
    int verified;               /**< Whether I have already checked if data are correct      */
    char *pathToFile;           /**< path to ckpt file                                       */
    FILE *fd;                   /**< file Descriptor of ckpt                                 */
}FTICkptFile;


    /** @typedef  FTIDataVar 
     *  @brief     Contains all information checkpoints performed collectively by 
     *             an FTI_Checkpoint call
     *
     *  
     *      This structure wraps all the information of the checkpoiints. Currently the utility
     *      supports only POSIX checkpoints. Therefore a single checkpoint consists of multiple 
     *       checkpoint files. Therefore the structure is called "Collection"
     *  
     */

typedef struct FTI_collection{
    FTICkptFile *files;        /**< All files of a checkpoint                              */
    int numCkpts;               /**< Number of checkpoitns                                  */
    int ckptId;                 /**< Id of the checkpoint                                   */
}FTICollection;




int FTI_LLInitEnvironment(char *configFile);
int FTI_LLGetNumCheckpoints();
int FTI_LLGetCkptID(int *ckptIds);
int FTI_LLFinalizeUtil();
int FTI_LLGetNumUserRanks();
int FTI_LLverifyCkpt( int ckptId, int rank);
int FTI_LLGetNumVars(int ckptId, int rank);
int FTI_LLreadVariable(int varIndex, int ckptId, int rank, char **varName,int *varId,  unsigned char **buf, size_t *size);

#endif
