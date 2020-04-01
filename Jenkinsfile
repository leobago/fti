#!/bin/groovy

ITFLocalFixtures = [
]

pipeline {
  agent none
  stages {
    stage('ITF Tests') {
      agent { docker { image 'kellekai/archlinuxopenmpi1.10:stable' } }
      
      stages {
        stage('Build') {
          steps { sh 'scripts/install.sh' }
        }

        stage('Local Tests') {
          steps { sh 'cd build/test && ci/localtests.sh' }
        }
      }
    }
  }
}