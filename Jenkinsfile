#!/bin/groovy

def itf_suite_compilation(stage) {
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

def itf_suite(stage, compilerName) {
  labelledShell (label:'Clean Folder', script:"rm -rf build/ install/")
  labelledShell (
    label:'Build FTI',
    script:"testing/tools/ci/build.sh ${compilerName}"
  )
  tests = labelledShell (
    label: "List ${stage} suites",
    script: "testing/tools/ci/testdriver --find ${stage}",
    returnStdout: true
  ).trim()
  
  for (String test : tests.split('\n'))
    catchError {
      labelledShell (
        label: "Suite: ${test}",
        script: "testing/tools/ci/testdriver --run ${test}"
      )
    }
}

def standard_checks(compilerName) {  
  stage('Standard behavior checks') { itf_suite('standard', compilerName) }
}

def diffsizes_checks(compilerName) {  
  stage('DiffSizes behavior checks') { itf_suite('diffsizes', compilerName) }
}

def feature_checks(compilerName) {  
  stage('Feature checks') { itf_suite('features', compilerName) }
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
    steps { itf_suite_compilation('compilation') }
  }

  //GCC

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

  //INTEL

  stage('Intel-Standard') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
        args '--volume intel-compiler:/opt/intel'
      }
    }
    steps {
     script { standard_checks('Intel') }
    }
  }

  stage('Intel-DiffSizes') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
        args '--volume intel-compiler:/opt/intel'
      }
    }
    steps {
     script { diffsizes_checks('Intel') }
    }
  }

  stage('Intel-Features') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
        args '--volume intel-compiler:/opt/intel'
      }
    }
    steps {
     script { feature_checks('Intel') }
    }
  }

  //CLANG

  stage('CLang-Standard') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { standard_checks('Clang') }
    }
  }

  stage('CLang-DiffSizes') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { diffsizes_checks('Clang') }
    }
  }

  stage('CLang-Features') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker {
        image 'ftibsc/ci:latest'
      }
    }
    steps {
     script { feature_checks('Clang') }
    }
  }

  //PGI

  stage('PGI-Standard') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker { 
        image 'ftibsc/ci:latest'
        args '--volume nvidia-compiler:/opt/nvidia'
      }
    }
    steps {
     script { standard_checks('PGI') }
    }
  }

  stage('PGI-DiffSizes') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker { 
        image 'ftibsc/ci:latest'
        args '--volume nvidia-compiler:/opt/nvidia'
      }
    }
    steps {
     script { diffsizes_checks('PGI') }
    }
  }

  stage('PGI-Features') {
    when { expression { return env.BRANCH_NAME == 'develop' } }
    agent {
      docker { 
        image 'ftibsc/ci:latest'
        args '--volume nvidia-compiler:/opt/nvidia'
      }
    }
    steps {
     script { feature_checks('PGI') }
    }
  }
}}
