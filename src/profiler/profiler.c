#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "profiler.h"

profileData *pfData;
int totalData = -1;
int currentData =  -1;
char *outDir= NULL;

void initProfiler(int numEvents, char *outputDir){
  pfData = (profileData*) malloc ( numEvents * sizeof(profileData)); 
  totalData = numEvents;
  currentData = 0;
  int len = strlen(outputDir)+1;
  outDir = (char *) malloc (sizeof(char)*len);
  strcpy(outDir, outputDir);
}

int addEvent(char *name){
  int i;
  for ( i = 0 ; i < currentData ; i++){
    if (strcmp(name, pfData[i].name ) == 0)
      return i;
  }
  
  if (currentData == totalData-1){
    pfData = realloc(pfData, totalData * 2 *sizeof(profileData));
    if ( !pfData ) {
      fprintf(stderr,"Could not reallocate array for profiler data");
    }
    totalData = 2*totalData;
  }

  int l = strlen(name)+1;
  currentData ++;
  pfData[i].name = (char *) malloc (sizeof(char)*l);
  strcpy(pfData[i].name, name);
  pfData[i].start = 0.0;
  pfData[i].end = 0.0;
  pfData[i].totalDuration = 0.0;
  pfData[i].timesCalled = 0;
  return i;
}

void startCount(char *name){
  int i;
  i = addEvent(name);
  pfData[i].start = MPI_Wtime();
}


void stopCount(char *name){
  int i;
  for ( i = 0 ; i < currentData ; i++)
    if (strcmp(name, pfData[i].name ) == 0)
      break;
  if ( i == currentData ){
    return ;
  }
  pfData[i].end= MPI_Wtime();
  pfData[i].totalDuration += (pfData[i].end - pfData[i].start);
  pfData[i].timesCalled ++;
  return;
}

void finalizeProfiler(){
  char procName[MPI_MAX_PROCESSOR_NAME];  
  int len;
  int i;
  char outName[200];
  MPI_Get_processor_name(procName, &len);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
 
  sprintf(outName,"%s%s_%d.prof",outDir,procName,world_rank);
  printf("Out dir is %s\n",outName);


  FILE *fd = fopen(outName,"w+");

  for ( i = 0 ; i < currentData; i++){
    fprintf(fd, "%s: %g %ld\n",pfData[i].name, pfData[i].totalDuration, pfData[i].timesCalled);
    free(pfData[i].name);
    pfData[i].name = NULL;
  }

  free(pfData);
  free(outDir);
  pfData = NULL;
  outDir= NULL;
  fclose(fd);
}
