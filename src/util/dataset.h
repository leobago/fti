#ifndef FTI_DATASET_H
#define FTI_DATASET_H

/**--------------------------------------------------------------------------
  
  
  @brief        Initializes instance of FTIT_dataset.

  This function initializes the dataset to 0 and all the members with non-zero
  default values to the respective ones.

  @param        FTI_Exec[in]    <b> FTIT_execution* </b>    FTI execution
  metadata.
  @param        data[out]       <b> FTIT_dataset* </b>      Pointer to dataset
  to be initialized.
  @param        id[in]          <b> int </b>                id of dataset to
  be initialized.
  
  @return                       \ref FTI_SCES  
 

--------------------------------------------------------------------------**/
int FTI_InitDataset( FTIT_execution* FTI_Exec, FTIT_dataset* data , int id );

#endif
