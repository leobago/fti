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
  labelledShell (label:"Clean Folder", script:"rm -rf build/ install/")
  labelledShell (
    label:"Build FTI",
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
        script: "./run testing/tools/ci/testdriver --run ${test}"
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

// FIXME workaround until hdf5 checks fixed
def itf_suite_intel(stage, compilerName) {
  labelledShell (label:"Clean Folder", script:"rm -rf build/ install/")
  labelledShell (
    label:"Build FTI",
    script:"testing/tools/ci/build.sh ${compilerName}"
  )
  tests = labelledShell (
    label: "List ${stage} suites",
    script: "testing/tools/ci/testdriver --find ${stage} | egrep -v hdf5",
    returnStdout: true
  ).trim()
  
  for (String test : tests.split('\n'))
    catchError {
      labelledShell (
        label: "Suite: ${test}",
        script: "./run testing/tools/ci/testdriver --run ${test}"
      )
    }
}

def feature_checks_intel(compilerName) {  
  stage('Feature checks') { itf_suite_intel('features', compilerName) }
}

pipeline {
agent none

stages {
  //stage('Compilation checks') {
  //  agent {
  //    docker {
  //      image 'ftibsc/ci:latest'
  //      args '--volume cmake-versions:/opt/cmake'
  //    }
  //  }
  //  steps { itf_suite_compilation('compilation') }
  //}

  ////GCC

  //stage('GCC-Standard') {
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-gnu-openmpi:/opt/gnu-openmpi --env MPIRUN_ARGS=--oversubscribe --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { standard_checks('GCC') }
  //  }
  //  post {
  //    always {
  //      labelledShell (
  //        label:'Generate coverage reports',
  //        script:"gcovr --xml -r . -o coverage.xml")
  //      cobertura coberturaReportFile: 'coverage.xml'
  //    }
  //  }
  //}

  //stage('GCC-DiffSizes') {
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-gnu-openmpi:/opt/gnu-openmpi --env MPIRUN_ARGS=--oversubscribe --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { diffsizes_checks('GCC') }
  //  }
  //  post {
  //    always {
  //      labelledShell (
  //        label:'Generate coverage reports',
  //        script:"gcovr --xml -r . -o coverage.xml")
  //      cobertura coberturaReportFile: 'coverage.xml'
  //    }
  //  }
  //}

  //stage('GCC-Features') {
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-gnu-openmpi:/opt/gnu-openmpi --env MPIRUN_ARGS=--oversubscribe --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { feature_checks('GCC') }
  //  }
  //  post {
  //    always {
  //      labelledShell (
  //        label:'Generate coverage reports',
  //        script:"gcovr --xml -r . -o coverage.xml")
  //      cobertura coberturaReportFile: 'coverage.xml'
  //    }
  //  }
  //}

  //PGI

  stage('PGI-Standard') {
    when { expression: { return env.CHANGE_TARGET == 'master' } beforeAgent: true }
    agent {
      docker {
        image 'ftibsc/debian-stable-slim-dev:latest'
        args '--volume ci-pgi-openmpi:/opt/pgi-openmpi --env MPIRUN_ARGS="--oversubscribe --mca mpi_cuda_support 0" --shm-size=4G'
      }
    }
    steps {
     script { standard_checks('PGI') }
    }
  }

  //stage('PGI-DiffSizes') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-pgi-openmpi:/opt/pgi-openmpi --env MPIRUN_ARGS="--oversubscribe --mca mpi_cuda_support 0" --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { diffsizes_checks('PGI') }
  //  }
  //}

  //stage('PGI-Features') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-pgi-openmpi:/opt/pgi-openmpi --env MPIRUN_ARGS="--oversubscribe --mca mpi_cuda_support 0" --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { feature_checks('PGI') }
  //  }
  //}

  ////LLVM
  //
  //stage('LLVM-Standard') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-llvm-openmpi:/opt/llvm-openmpi --env MPIRUN_ARGS=--oversubscribe --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { standard_checks('LLVM') }
  //  }
  //}

  //stage('LLVM-DiffSizes') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-llvm-openmpi:/opt/llvm-openmpi --env MPIRUN_ARGS=--oversubscribe --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { diffsizes_checks('LLVM') }
  //  }
  //}

  //stage('LLVM-Features') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-llvm-openmpi:/opt/llvm-openmpi --env MPIRUN_ARGS=--oversubscribe --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { feature_checks('LLVM') }
  //  }
  //}

  //// Intel 

  //stage('Intel-Standard') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-intel-impi:/opt/intel-impi --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { standard_checks('Intel') }
  //  }
  //}

  //stage('Intel-DiffSizes') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-intel-impi:/opt/intel-impi --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { diffsizes_checks('Intel') }
  //  }
  //}

  //stage('Intel-Features') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-intel-impi:/opt/intel-impi --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { feature_checks_intel('Intel') }
  //  }
  //}
  //
  ////MPICH

  //stage('MPICH-Standard') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-gnu-mpich:/opt/gnu-mpich --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { standard_checks('MPICH') }
  //  }
  //}

  //stage('MPICH-DiffSizes') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-gnu-mpich:/opt/gnu-mpich --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { diffsizes_checks('MPICH') }
  //  }
  //}

  //stage('MPICH-Features') {
  //  when { expression { return env.CHANGE_TARGET == 'master' } }
  //  agent {
  //    docker {
  //      image 'ftibsc/debian-stable-slim-dev:latest'
  //      args '--volume ci-gnu-mpich:/opt/gnu-mpich --shm-size=4G'
  //    }
  //  }
  //  steps {
  //   script { feature_checks('MPICH') }
  //  }
  //}


}}
