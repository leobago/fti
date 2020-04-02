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
      stage('ITF Tests') {
      agent { docker { image 'kellekai/archlinuxopenmpi1.10:stable' } }
      
      stages {
        stage('Build') {
          steps { sh 'scripts/install.sh -DENABLE_HDF5=1' }
        }

        stage('Local Tests') {
          steps { sh 'cd build/test && ci/localtests.sh' }
        }
      }
    }
  }
}