#!/bin/groovy

def executeSteps_one( arg1, arg2 ) {
  env.PATHA = arg1 
  env.PATHB = arg2 
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH0I1.fti LEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I1.fti LEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I0.fti LEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH0I1.fti LEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I1.fti LEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I0.fti LEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
}

def executeSteps_two( arg1, arg2 ) {
  env.PATHA = arg1 
  env.PATHB = arg2
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH0I1.fti LEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I1.fti LEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I0.fti LEVEL=3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.3 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.4 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.5 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.6 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.7 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.8 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.9 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH0I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=4 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH0I1.fti LEVEL=4 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=4 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=4 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I1.fti LEVEL=4 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=addInArray CONFIG=configH1I0.fti LEVEL=4 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=heatdis CONFIG=configH0I1.fti ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=heatdis CONFIG=configH1I1.fti ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=heatdis CONFIG=configH1I0.fti ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=nodeFlag CONFIG=configH0I1.fti ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=nodeFlag CONFIG=configH1I1.fti ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=nodeFlag CONFIG=configH1I0.fti ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=syncIntv CONFIG=configH1I0.fti ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=hdf5 ./test/tests.sh
      '''
  }
  catchError {
    sh '''
      export PATH=$PATHA:$PATHB:$PATH
      echo $PATH
      cd build; TEST=cornerCases ./test/tests.sh
      '''
  }
}

versions = [ '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9' ]

def cmakesteps(list) {
  for (int i = 0; i < list.size(); i++) {
    env.CMAKE = "/opt/cmake/${list[i]}/bin/cmake"
    sh '''
      mkdir build; cd build
      $CMAKE --version
      $CMAKE -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE ..
      make -j 16 all install
      '''
    catchError {
      sh 'cd build; TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh'
    }
    sh 'rm -rf build'
  }
}

pipeline {
  agent none
    stages {
      stage('Cmake Versions Test') {
        agent {
          docker {
            image 'kellekai/archlinuxopenmpi1.10'
          }
        }
        steps {
          cmakesteps(versions)
        }
      }
      stage('GCC Compiler Tests (1/2)') {
        agent {
          docker {
            image 'kellekai/archlinuxopenmpi1.10:stable'
          }
        }
        steps {
          sh '''
            mkdir build; cd build
            cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            make -j 16 all install
          '''
          executeSteps_one( '', '' )
        }
      }
      stage('GCC Compiler Tests (2/2)') {
        agent {
          docker {
            image 'kellekai/archlinuxopenmpi1.10:stable'
          }
        }
        steps {
          sh '''
            mkdir build; cd build
            cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            make -j 16 all install
          '''
          executeSteps_two( '', '' )
        }
      }
      stage('Clang Compiler Tests (1/2)') {
        agent {
          docker {
            image 'kellekai/archlinuxopenmpi1.10:stable'
          }
        }
        steps {
          sh '''
            mkdir build; cd build
            export OMPI_MPICC=clang
            export OMPI_CXX=clang++
            CC=clang FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            VERBOSE=1 make -j 16 all install
          '''
          executeSteps_one( '', '' )
        }
      }
      stage('Clang Compiler Tests (2/2)') {
        agent {
          docker {
            image 'kellekai/archlinuxopenmpi1.10:stable'
          }
        }
        steps {
          sh '''
            mkdir build; cd build
            export OMPI_MPICC=clang
            export OMPI_CXX=clang++
            CC=clang FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            VERBOSE=1 make -j 16 all install
          '''
          executeSteps_two( '', '' )
        }
      }
      stage('PGI Compiler Tests (1/2)') {
        agent {
          docker {
            image 'kellekai/archlinuxpgi18:stable'
          }
        }
        environment {
          PGICC = '/opt/pgi/linux86-64/18.4/bin/'
          PGIMPICC = '/opt/pgi/linux86-64/2018/mpi/openmpi-2.1.2/bin/'
          LM_LICENSE_FILE = '$PGI/license.dat'
          LD_LIBRARY_PATH = '/opt/pgi/linux86-64/18.4/lib'
        }
        steps {
          sh '''
            export PATH=$PGICC:$PGIMPICC:$PATH
            echo $PATH
            ls /opt/pgi/
            mkdir build; cd build
            CC=pgcc FC=pgfortran cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            make -j 16 all install
          '''
          executeSteps_one( '/opt/pgi/linux86-64/18.4/bin/', '/opt/pgi/linux86-64/2018/mpi/openmpi-2.1.2/bin/' )
        }
      }
      stage('PGI Compiler Tests (2/2)') {
        agent {
          docker {
            image 'kellekai/archlinuxpgi18:stable'
          }
        }
        environment {
          PGICC = '/opt/pgi/linux86-64/18.4/bin/'
          PGIMPICC = '/opt/pgi/linux86-64/2018/mpi/openmpi-2.1.2/bin/'
          LM_LICENSE_FILE = '$PGI/license.dat'
          LD_LIBRARY_PATH = '/opt/pgi/linux86-64/18.4/lib'
        }
        steps {
          sh '''
            export PATH=$PGICC:$PGIMPICC:$PATH
            echo $PATH
            ls /opt/pgi/
            mkdir build; cd build
            CC=pgcc FC=pgfortran cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            make -j 16 all install
          '''
          executeSteps_two( '/opt/pgi/linux86-64/18.4/bin/', '/opt/pgi/linux86-64/2018/mpi/openmpi-2.1.2/bin/' )
        }
      }
      stage('Intel Compiler Tests (1/2)') {
        agent {
          docker {
            image 'kellekai/archlinuximpi18:stable'
          }
        }
        environment {
          CFLAGS_FIX = '-D__PURE_INTEL_C99_HEADERS__ -D_Float32=float -D_Float64=double -D_Float32x=_Float64 -D_Float64x=_Float128'
          ICCPATH = '/opt/intel/compilers_and_libraries_2018.3.222/linux/bin'
          MPICCPATH = '/opt/intel/compilers_and_libraries_2018.3.222/linux/mpi/intel64/bin'
        }
        steps {
          sh '''
            mkdir build; cd build
            . $ICCPATH/compilervars.sh intel64
            . $MPICCPATH/mpivars.sh
            CFLAGS=$CFLAGS_FIX cmake -C ../intel.cmake cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            make -j 16 all install
          '''
          executeSteps_one( '/opt/intel/compilers_and_libraries_2018.3.222/linux/mpi/intel64/bin', '' )
        }
      }
      stage('Intel Compiler Tests (2/2)') {
        agent {
          docker {
            image 'kellekai/archlinuximpi18:stable'
          }
        }
        environment {
          CFLAGS_FIX = '-D__PURE_INTEL_C99_HEADERS__ -D_Float32=float -D_Float64=double -D_Float32x=_Float64 -D_Float64x=_Float128'
          ICCPATH = '/opt/intel/compilers_and_libraries_2018.3.222/linux/bin'
          MPICCPATH = '/opt/intel/compilers_and_libraries_2018.3.222/linux/mpi/intel64/bin'
        }
        steps {
          sh '''
            mkdir build; cd build
            . $ICCPATH/compilervars.sh intel64
            . $MPICCPATH/mpivars.sh
            CFLAGS=$CFLAGS_FIX cmake -C ../intel.cmake cmake -DCMAKE_INSTALL_PREFIX=`pwd`/RELEASE -DENABLE_HDF5=ON ..
            make -j 16 all install
          '''
          executeSteps_two( '/opt/intel/compilers_and_libraries_2018.3.222/linux/mpi/intel64/bin', '' )
        }
      }
    }
}
