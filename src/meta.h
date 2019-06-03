#ifndef __META_H__
#define __META_H__

int FTI_GetChecksums( char* checksum, char* ptnerChecksum, char* rsChecksum);
int FTI_WriteRSedChecksum( int rank, char* checksum);
int FTI_LoadTmpMeta();
int FTI_LoadMeta();
int FTI_WriteMetadata( long* fs, long mfs, char* fnl, char* checksums, 
int* allVarIDs, long* allVarSizes, unsigned long* allLayerSizes,
char* allLayerHashes , long *allVarPositions);
int FTI_CreateMetadata();
int FTI_WriteCkptMetaData();
int FTI_LoadCkptMetaData();
int FTI_LoadL4CkptMetaData();

#endif // __META_H__
