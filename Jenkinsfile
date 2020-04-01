#!/bin/groovy

ITFLocalFixtures = [
  'local/standard',
  'local/standard-disrupt',
  'local/staging/staging',
  'local/diffckpt/dCP-standard',
  'local/diffckpt/dCP-corrupt',
  'local/keepL4Ckpt/keepl4',
  'local/recoverVar/recovervar',
  'local/recoverVar/recovername']

pipeline {
  agent none
  stages {
    stage('Local Tests') {
      agent {
        docker { image 'kellekai/archlinuxopenmpi1.10:stable' }
      }
      
      steps {
        sh 'scripts/install.sh && cd build/test'
        
        for (fixture in ITFLocalFixtures) {
          catchError { sh "itf/testrunner $fixture" }
        }
      }
    }
  }
}
