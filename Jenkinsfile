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

def standard_checks(compilerName) {  
  step('Compilation') {
    labelledShell (label:'Clean Folder', script:"rm -rf build/ install/")
    labelledShell (
      label:'Build FTI',
      script:"testing/tools/ci/build.sh ${compilerName}"
    )
  }
  step('Standard behavior checks') { itf_suite('standard') }
}

def diffsizes_checks(compilerName) {  
  step('Compilation') {
    labelledShell (label:'Clean Folder', script:"rm -rf build/ install/")
    labelledShell (
      label:'Build FTI',
      script:"testing/tools/ci/build.sh ${compilerName}"
    )
  }
  step('DiffSizes behavior checks') { itf_suite('diffsizes') }
}

def feature_checks(compilerName) {  
  step('Compilation') {
    labelledShell (label:'Clean Folder', script:"rm -rf build/ install/")
    labelledShell (
      label:'Build FTI',
      script:"testing/tools/ci/build.sh ${compilerName}"
    )
  }
  step('Feature checks') { itf_suite('features') }
}

pipeline {
agent none

stages {
  stage('Compilation checks') {
    agent {
      docker {
        image 'ftibsc/ci:latest'
        args '--volume cmake-versions:/opt/cmake'
      }
    }
    steps { itf_suite('compilation') }
  }

  stage('GCC-Standard') {
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { standard_checks('GCC') }
    }
    post {
      always {
        labelledShell (
          label:'Generate coverage reports',
          script:"gcovr --xml -r . -o coverage.xml")
        cobertura coberturaReportFile: 'coverage.xml'
      }
    }
  }

  stage('GCC-DiffSizes') {
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { diffsizes_checks('GCC') }
    }
    post {
      always {
        labelledShell (
          label:'Generate coverage reports',
          script:"gcovr --xml -r . -o coverage.xml")
        cobertura coberturaReportFile: 'coverage.xml'
      }
    }
  }

  stage('GCC-Features') {
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { feature_checks('GCC') }
    }
    post {
      always {
        labelledShell (
          label:'Generate coverage reports',
          script:"gcovr --xml -r . -o coverage.xml")
        cobertura coberturaReportFile: 'coverage.xml'
      }
    }
  }

  /*stage('GCC') {
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { standard_checks('GCC') }
     script { diffsizes_checks('GCC') }
     script { feature_checks('GCC') }
    }
    post {
      always {
        labelledShell (
          label:'Generate coverage reports',
          script:"gcovr --xml -r . -o coverage.xml")
        cobertura coberturaReportFile: 'coverage.xml'
      }
    }
  }*/

  stage('Intel') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
        args '--volume intel-compiler:/opt/intel'
      }
    }
    steps {
     script { standard_checks('Intel') }
     script { diffsizes_checks('Intel') }
     script { feature_checks('Intel') }
    }
    /*steps { script { compiler_checks('Intel') } }*/
  }

  stage('CLang') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { standard_checks() }
     script { diffsizes_checks() }
     script { feature_checks() }
    }
    /*steps { script { compiler_checks('Clang') } }*/
  }

  stage('PGI') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker { 
        image 'ftibsc/ci:latest'
        args '--volume pgi-compiler:/opt/pgi'
      }
    }
    steps {
     script { standard_checks() }
     script { diffsizes_checks() }
     script { feature_checks() }
    }
    /*steps { script { compiler_checks('PGI') } }*/
  }
}}
