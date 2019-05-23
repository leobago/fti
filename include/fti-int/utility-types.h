#ifndef UTILITY_TYPE_H
#define UTILITY_TYPE_H

typedef struct
{
  FTIT_configuration* FTI_Conf;
  MPI_File pfh;
  MPI_Offset offset;
  int err;
} WriteMPIInfo_t;

#endif // UTILITY_TYPE_H
