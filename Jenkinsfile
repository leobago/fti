#!/bin/groovy

def itf_suite(stage) {
  tests = labelledShell (
    label: "List ${stage} suites",
    script: "testing/tools/ci/testdriver --find ${stage}",
    returnStdout: true
  ).trim()
  
  for( String test : tests.split('\n'))
    catchError {
      labelledShell (
        label: "Suite: ${test}",
        script: "testing/tools/ci/testdriver --run ${test}"
      )
    }
}

def compiler_checks(compilerName) {  
  stage('Compilation') {
    labelledShell (label:'Clean Folder', script:"rm -rf build/ install/")
    labelledShell (
      label:'Build FTI',
      script:"testing/tools/ci/build.sh ${compilerName}"
    )
  }
  stage('Core behavior checks') { itf_suite('core') }
  stage('Feature checks') { itf_suite('features') }
}

pipeline {
agent none

stages {
  stage('Complation checks') {
    agent {
      docker {
        image 'ftibsc/ci:latest'
        args '--volume cmake-versions:/opt/cmake'
      }
    }
    steps { itf_suite('compilation') }
  }

  stage('GCC') {
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps { script { compiler_checks('GCC') } }
    post {
      always {
        labelledShell (
          label:'Generate coverage reports',
          script:"gcovr --xml -r . -o coverage.xml")
        cobertura coberturaReportFile: 'coverage.xml'
      }
    }
  }

  stage('Intel') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
        args '--volume intel-compiler:/opt/intel'
      }
    }
    steps { script { compiler_checks('Intel') } }
  }

  stage('CLang') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps { script { compiler_checks('Clang') } }
  }

  stage('PGI') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker { 
        image 'ftibsc/ci:latest'
        args '--volume pgi-compiler:/opt/pgi'
      }
    }
    steps { script { compiler_checks('PGI') } }
  }
}}
