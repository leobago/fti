#!/bin/groovy

pipeline {
  agent none
  stages {
    stage('Local Tests') {
      agent {
        docker { image 'kellekai/archlinuxopenmpi1.10:stable' }
      }
      
      steps {
        sh 'scripts/install.sh && cd build/test'
        
        catchError { sh 'itf/testrunner local/standard' }
        catchError { sh 'itf/testrunner local/standard-disrupt' }

        catchError { sh 'itf/testrunner local/staging/staging' }

        catchError { sh 'itf/testrunner local/diffckpt/dCP-standard' }
        catchError { sh 'itf/testrunner local/diffckpt/dCP-corrupt' }

        catchError { sh 'itf/testrunner local/keepL4Ckpt/keepl4' }

        catchError { sh 'itf/testrunner local/recoverVar/recovervar' }
      }
    }
  }
}
