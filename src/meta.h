#ifndef __META_H__
#define __META_H__

int FTI_GetChecksums( char* checksum, char* ptnerChecksum, char* rsChecksum);
int FTI_WriteRSedChecksum( int rank, char* checksum);
int FTI_LoadTmpMeta(void);
int FTI_LoadMeta(void);
int FTI_WriteMetadata( long* fs, long mfs, char* fnl, char* checksums, 
int* allVarIDs, long* allVarSizes, unsigned long* allLayerSizes,
char* allLayerHashes , long *allVarPositions);
int FTI_CreateMetadata(void);
int FTI_WriteCkptMetaData(void);
int FTI_LoadCkptMetaData(void);
int FTI_LoadL4CkptMetaData(void);

#endif // __META_H__
