#ifndef STAGE_DEFS_H
#define STAGE_DEFS_H

#define FTI_SI_NAVL 0x1
#define FTI_SI_IAVL 0x0

#define FTI_SI_IALL 0x1
#define FTI_SI_NALL 0x0

#define FTI_SI_APTR( ptr ) ((FTIT_StageAppInfo*)ptr)
#define FTI_SI_HPTR( ptr ) ((FTIT_StageHeadInfo*)ptr)

#define FTI_SI_MAX_ID (0x7ffff)

#define FTI_DISABLE_STAGING do{*enableStagingPtr = false;} while(0)
#define FTI_SI_ENABLED (*(bool*)enableStagingPtr)

/** status 'failed' for stage requests                                     */
#define FTI_SI_FAIL 0x4
/** status 'succeed' for stage requests                                    */
#define FTI_SI_SCES 0x3
/** status 'active' for stage requests                                     */
#define FTI_SI_ACTV 0x2
/** status 'pending' for stage requests                                    */
#define FTI_SI_PEND 0x1
/** status 'not initialized' for stage requests                            */
#define FTI_SI_NINI 0x0

/** Maximum amount of concurrent active staging requests                   
  @note leads to 2.5MB for the application processes as minimum memory
  allocated
 **/
#define FTI_SI_MAX_NUM (512L*1024L) 

#endif // STAGE_DEFS_H
