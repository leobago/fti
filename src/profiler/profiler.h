#ifndef __PROFILER__
#define __PROFILER__

typedef struct{
  double start, end;
  double totalDuration;
  long timesCalled;
  char *name;
}profileData;


void initProfiler(int numEvents, char *outputDir);
void startCount(char *name);
void stopCount(char *name);
void finalizeProfiler();
#endif
