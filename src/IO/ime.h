#ifndef __IME_H__
#define __IME_H__
#ifdef __cplusplus
extern "C"
{
#endif
void* FTI_InitIME(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);

#ifdef __cplusplus
}
#endif
#endif // __IME_H__
