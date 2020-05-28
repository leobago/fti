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
  stage('CMake checks') { itf_suite('cmake') }
  stage('Compilation') {
    labelledShell (label:'Clean Folder', script:"rm -rf build/ install/")
    labelledShell (
      label:'Build FTI',
      script:"testing/tools/ci/build.sh ${compilerName}"
    )
  }
  stage('Local checks') { itf_suite('local') }
  stage('Integration checks') { itf_suite('integration') }
}

pipeline {
agent none

stages {
  stage('GCC') {
    when { expression { return env.BRANCH_NAME == 'master' } }
    agent { docker { image 'kellekai/archlinuxopenmpi1.10:stable' } }
    steps { script { compiler_checks('GCC') } }
  }
  stage('Intel') {
    when { expression { return env.BRANCH_NAME == 'master' } }
    agent { docker { image 'kellekai/archlinuximpi18:stable' } }
    steps { script { compiler_checks('Intel') } }
  }
  stage('CLang') {
    when { expression { return env.BRANCH_NAME == 'master' } }
    agent { docker { image 'kellekai/archlinuxopenmpi1.10:stable' } }
    steps { script { compiler_checks('Clang') } }
  }
  stage('PGI') {
    when { expression { return env.BRANCH_NAME == 'master' } }
    agent { docker { image 'kellekai/archlinuxpgi18:stable' } }
    steps { script { compiler_checks('PGI') } }
  }
}}
