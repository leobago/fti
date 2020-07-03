.. Fault Tolerance Library documentation ci

FTI Continuous Integration Environment
===================================================

FTI follows the guidelines of `Continuous Integration <https://en.wikipedia.org/wiki/Continuous_integration>`_ (CI) development process.
In this scheme, small new additions are included in Pull Requests (PR) to the *develop* branch on github.
The additions must be automatically validated through the :doc:`test suites <test_suites>` prior to integration.
This is achieved using `Jenkins <https://www.jenkins.io/>`_ to define a development pipeline.

Jenkins pipeline is defined in the **JenkinsFile** in FTI root folder.
This pipeline is executed by a Jenkins server configured to test every PR sent to FTI's github repository.
In most cases, Jenkins will compile FTI with the GCC compiler and all IO libraries and execute all test suites.
Furthermore, if the PR targets the master branch, this process is repeated for these additional compilers:
(i) PGI;
(ii) CLang and
(iii) Intel.

The FTI CI environment is composed of all libraries and software tools involved in the build and testing pipeline.
The CI environment is contained in a Docker image so that these software pieces can be easily managed and duplicated.
Docker is a software to create a sandbox to execute software in the form of containers.
The current image used in FTI is named **alexadrelimasssantana/fti-ci:latest**.
Jenkins uses this image to instantiate a container and run the pipeline without conflicting with the host machine software.
To donwload FTI docker image, make sure you have docker installed and perform the following command.

.. code-block:: bash
   docker pull alexadrelimasssantana/fti-ci:latest


Docker Image software stack
-----------

The FTI Docker image is based on the official x86_64 `ArchLinux image <https://hub.docker.com/_/archlinux/>`_.
The image is extended to contain the majority of software required to: 
(i) build FTI with all supported libraries;
(ii) execute all FTI test suites and
(iii) generate code coverage reports.
A non-exhaustive list of software in the Docker image, annotated with its installation procedure, is as follows.

**Libraries**
- OpenMPI-4.0.3 (pacman)
- HDF5-1.12 (pacman)
- SIONLib-1.7.6 (compiled from source and installed in /opt/sionlib)

**Compilers**
- CLang-10 (pacman)
- gcc-10.1 (pacman)
- gccfortran-10.1 (pacman)

**Tools**
- CMake-3.17 (pacman)
- DiffUtils (pacman)
- make-4.3 (pacman)
- python-3.8 (pacman)
- gcovr-4.2 (pacman)
- git-2.27 (pacman)

We recommend getting to know a bit about Docker for further details on the CI environment.
A good place to start is the Docker official `tutorial page <https://docs.docker.com/get-started/overview/>`_.
Moreover, most commands used in this guide will be shortly explained.
Granted that docker is already installed, the following command can be used to run the Docker image.

.. code-block:: bash
   docker run -d -t --name fti alexadrelimasssantana/fti-ci:latest
  
The command will create and run a container named **fti** based on FTI DockerHub image.
The container will be executed in daemon mode and be available for connections.
To connect to the container, issue the following command.

.. code-block:: bash
   docker exec -it fti /bin/bash

This command will connect to the **fti** container and execute the bash application.
The *-it* flag informs docker that this is an interative session.
Once inside the container, you can clone the fti repository using git and checkout to any specific branch.

.. code-block:: bash
   git clone https://github.com/leobago/fti
   cd fti
   git checkout develop

It is possible to replicate the GCC and CLang stages of the CI pipeline with this current setup.
This can be done by performing the same commands as depicted in those stages.
For the sake of simplicity, we added the command in the snippet below for the GCC compiler stage.
Those steps are not required to be executed, they are included here for demonstration only.

.. code-block:: bash
   testing/tools/ci/build.sh gcc
   testing/tools/ci/testdriver --run $(testing/tools/ci/testdriver --find core)
   testing/tools/ci/testdriver --run $(testing/tools/ci/testdriver --find features)

As mentioned earlier, this environment is not able yet to run the **Intel**, **PGI** and **Compilation Checks** CI stages.
This is due to the fact that these compilers are proprietary and require licenses to use.
As such, it is not possible to redistribute them within containers.
For the compilation checks, the stage lacks additional CMake installations.
We chose to let those out of the docker as to minimize its size.
These software pieces must be incorporated into the containers using `Docker volumes <https://docs.docker.com/storage/volumes/>`_.

Docker image: required volumes
-----------

Docker volumes are, in essence, filesystem bindings between the host machine and the container.
Volumes can store persistent data in the host machine and allow the container to use it.
FTI employs three volumes to fulfill its CI pipeline:
(i) cmake-versions;
(ii) pgi-compiler and
(iii) intel-compiler.

In order to fully replicate the CI environment, these three volumes must be mounted in the docker container.
We will go over the process of creating these volumes using your own licenses and CMake installations.

cmake-versions
~~~~~~~

The *cmake-versions* volume is a requirement of the **Compilation Checks** CI stage.
This volume contains multiple CMake version installations that are used in the *cmake_versions* test suite.
The CI pipeline expects this volume to be a filesystem with one folder for each CMake version that is supported by the test suite.
Each folders must be named under the CMake version preppended with a 'v' character (e.g v3.10, v3.3.2).
As of now, these are the CMake versions that must be present in the cmake-versions volume folders:
(i) 3.3.2;
(ii) 3.4;
(iii) 3.5
(iv) 3.7;
(v) 3.8;
(vi) 3.9;
(vii) 3.10;
(viii) 3.11;
(ix) 3.12;
(x) 3.13;
(xi) 3.14;
(xii) 3.15 and
(xiii) 3.16.

Docker volumes are managed by the Docker application.
As such, they need to be populated by a container.
To create the cmake-versions volume, run the FTI container with the following parameters.

.. code-block:: bash
   docker run -d -t --name make-cmake --volume cmake-versions:/opt/cmake alexandrelimassantana/fti-ci:latest

This command will create a new container, **make-cmake**, and a new volume **cmake-versions** mounted in */opt/cmake*.
Now, all we must do is to populate the container with the CMake installations.
For that, you will need to connect to the container which can be done with the following command.

.. code-block:: bash
   docker exec -it make-cmake /bin/bash

Now that you are inside the container, verify if the volume is mounted in */opt/cmake* with the *ls* command.
You should see an empty folder, for now.
Now, to populate the volume, we need to build and install the multiple CMake versions there.
One of the ways to do this, is to clone the CMake github directory and build all versions from the source.
Fortunately, the build part can be using CMake which is already installed in the Docker Image.
The following script will install the first two required CMake versions (i.e 3.3.2 and 3.4) in the mounted volume.

.. code-block:: bash
   git clone https://github.com/Kitware/CMake
   cd CMake;

   versions=('3.3.2' '3.4')
   for v in ${versions[@]}; do
     mkdir build
     git checkout v$v
     cd build
     cmake .. -DCMAKE_INSTALL_PREFIX==/opt/cmake/v$v
     make install -j
     cd ..
     rm -rf build
   done

The aforementioned script can be used to build and install all versions.
To do that, simply append the other versions into the **versions** bash array.
After running this script for all versions, the volume should be ready to use.
To check if everything is in order, you can manually run the **cmake_versions** test case with the following command.

.. code-block:: bash
   cd path/to/fti/local/git;
   testing/tools/ci/testdriver --run $(testing/tools/ci/testdriver --find compilation)
  
This script will run the compilation suite which will only succeed if all CMake versions where installed correctly.
If everything went well, you can exit the container and the volume will persist in the host machine.
It is important to remember that you need to launch the container with the volume mounted everytime you need to run this stage.
The following command will do this.

.. code-block:: bash
   docker run -d -t --volume cmake-versions:/opt/cmake alexandrelimassantana/fti-ci:latest

pgi-compiler
~~~~~~~

The *pgi-compiler* volume is a requirement of the **PGI** CI stage.
This volume should contain an installation of the PGI community edition compiler and license.
To build this volume, you first need to download the PGI compiler in this `link <https://www.pgroup.com/index.htm>`_.
After this step, get a docker running with the following command.

.. code-block:: bash
   docker run -d -t --name make-pgi --volume pgi-compiler:/opt/pgi alexandrelimassantana/fti-ci:latest

The command will create an image named **make-pgi** and a new volume, **pgi-compiler** in */opt/pgi*.
We need to install the PGI compiler inside the volume just as we did with the CMake versions.
However, this time we downloaded the compressed compiler in the host machine.

.. note::  The PGI website does not provide a link to download the compiler with wget.

To copy the tar file into the container, we can use the *docker cp* command.
The following snippet exemplifies this.

.. code-block:: bash
   docker cp path/to/pgi.tar.gz make-pgi:/home/ftidev

Now we can connect to the docker, unpack the compiler and run its install script.
Using the default options should be enough, just make sure to install the compiler at the volume in */opt/pgi*.
To verify if the installation went out correctly, try to build FTI with PGI using the following command.

.. code-block:: bash
   cd path/to/fti/git/local/repo
   git checkout develop
   testing/tools/ci/build.sh pgi

This should build FTI using the pgi compiler found in /opt/pgi.
If the compilation fails, check if the paths in the *build.sh* matches the PGI compiler version.
As of now, the script will rely on version *19.10* of the compiler.
Once you are done, you can leave the container and the volume will persist in the host machine.

intel-compiler
~~~~~~~

.. warning:: not yet in the docker image.