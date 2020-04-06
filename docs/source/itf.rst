.. Fault Tolerance Library documentation ITF file

FTI Integrated Tests Framework (FTI-ITF)
===================================================

The FTI Integrated Test Framework (ITF) is, as the name suggests, a tool to create integration tests for FTI.
ITF provides a small subset of testing features similar to those found in libraries like CUnit, JUnit, googletets and others.
ITF is tailored for defining black-box integration tests for FTI using MPI C test applications and bash scripts.


Executing ITF test cases
-----------------

FTI integration tests are composed of four components: (i) test application; (ii) test fixture; (iii) test suite and (iv) ITF test driver.
The test application is responsible to mock an use case for an FTI feature.
It will most likely simulate a healthy execution and failures given a set of parameters.
The test fixture is a set of bash functions to prepare, lauch and verify the output of the FTI test application.
The test suite is a collection of the individual parameters the fixture can receive to cover different cases of the feature being tested.
Finally, the ITF test driver is a bash program that composes everything together, executes the test and provides a summary of the integration tests.

ITF is used to assist the construction of the fixture and suite components.
It also executing the tests through the `testdriver` script.
In a nutshell, ITF creates a standard for how to build integration test cases for FTI.
ITF provides a set of commonly used functions for testing FTI which are built upon these standards.
It enhances code reusability for tests in FTI and provides abstractions for testing procedures.

ITF source files can be found in the `testing/itf` folder.
They are processed during compilation by CMake and are later installed in the same folder within the build directory.
ITF has four files with the following definitions:

- variables: a bash script file that defines all variables present in ITF;
- api: a set of bash functions for test developers to call within a fixture;
- engine: a set of ITF private bash functions to assist the testrunner;
- testdriver: a bash program to execute ITF tests;
- fti_template.cfg: a template FTI configuration file set with default values for the fields

If you intend to develop tests through ITF, a good place to start is with the `variables` and `api` files.
If you intend to execute ITF tests, checking how the `testdriver` works is the first step.
You should only worry about the engine file if you intend to improve ITF or hunt for bugs in the engine.
The remainder of this guide will help you get familiar with the basics of testing with ITF.


ITF Fixtures with the default suite
-----------------

An ITF test case is composed of a **fixture** and its required parameters.
ITF provides a **testdriver** script to execute it's test cases.
By default, ITF fixtures have the `.fixture` file extension.
ITF also expose sets of test cases with suite files with the `.suite` extension.

The testdriver script takes as a parameter the name of a fixture file without its extension.
By default, the program will try to find a suite with the same name (i.e. mytest.fixture and mytest.suite).
If it can find these files, it will run the fixture with each set of parameters provided by the suite file.

To illustrate that, let's run the recoverVar ITF tests.
First, build FTI and then navigate to the main testing directory.
You can do so by calling the handy install script in the root folder.

.. code-block:: bash

    ./install.sh --enable-testing
    cd build/testing


Now, lets call the ITF test runner with the recovervar default suite.

.. code-block:: bash

    itf/testdriver local/recoverVar/recovervar


This command will run all the individual test cases for the recovervar fixture.
Each of the individual cases are depicted in the suite file in `local/recoverVar/recovervar.suite`.
If your tests failed, it might be because of the MPI configurations.
FTI uses MPI applications to run its tests and the default number of ranks is 16.
If your machine has fewer ranks than that, you might have to oversubscribe to run ITF tests.
ITF uses the `MPIRUN_ARGS` variable to pass additional parameters to its MPI tests.
If your MPI distribution is OpenMPI, you can issue this command to oversubscribe your machine.

.. code-block:: bash

    export MPIRUN_ARGS=--oversubscribe



ITF fixture with a single test case
-----------------

It might be the case that only a specific test case is of the interest in a fixture.
ITF can execute a single test case paired with a fixture by having the `--custom-params` or `-c` flag passed to the testdriver.
This flag will make the testdriver not look for a matching **suite** file.
Instead, it will get the parameters from the command line arguments passed after the fixture name.
The parameters have to be passed with the following format `--varname value`.

For instance, if we want to run the test that checks the standard behavior of FTI, we can use the standard fixture located in local/standard folder.
It requires the following parameters: (i) iolib; (ii) level; (iii) icp; (iv) diffsize; (v) head and (vi) keep.
These map to the IO library used, the checkpoint level tested, if FTI will use iCP or not, if checkpoint sizes differ, if FTI uses a dedicated process for checkpointing, and if it should keep the last checkpoint file respectively.
Try running a custom test case for this fixture by issuing the following command.

.. code-block:: bash
    itf/testdriver --custom-params local/standard/standard --iolib 1 --level 1 --icp 0 --diffsize 0 --head 0 --keep 0


This will run only the test case for the appropriate parameters set.
You can try running with different parameters to see how these are launched in ITF.
Non-expected parameters will make the testdriver fail, you can verify this with the following.

.. code-block:: bash
    itf/testdriver --custom-params local/standard/standard --iolib 1 --level 1 --icp 0 --diffsize 0 --head 0 --keep 0 --notexpected ohno

A failure is also expected if a parameter registered by the fixture is missing.
If a parameter is registered in the fixture, it needs to be provided.
This is true for when declaring a suite or executing with custom parameters.
As an example, the next command has the keep parameter missing.

.. code-block:: bash
    itf/testdriver --custom-params local/standard/standard --iolib 1 --level 1 --icp 0 --diffsize 0 --head 0



ITF fixture with a custom suite
-----------------


ITF also supports the execution of sets of test cases not tied with the default suite (i.e a `.suite` file with the same name as the fixture).
To execute a custom suite of test cases, pass the `-s` or `--suite` flag to the test driver.
This will associate the fixture and the custom suite and run every test defined in it.

As an example, let's imagine you only wants to run the dCP checks for the POSIX IO.
The dCP checks are defined in the `local/diffckpt/dCP-standard.fixture` file.
It has a default suite with the same name where the FTI-FF and POSIX IOs are tested.
Copy the POSIX parameters into another file in the same folder, the `dCP-POSIX.suite`.
The contents of the file should be as follows.

.. code-block::

    --iolib 1 --head 0 --mode NOICP
    --iolib 1 --head 0 --mode ICP
    --iolib 1 --head 1 --mode NOICP
    --iolib 1 --head 1 --mode ICP


Do not forget to add a newline feed after the last line.
Otherwise, the testdriver will not run the last test case.
With the file ready, run the following command.

.. code-block:: bash

    itf/testdriver --suite=local/diffckpt/dCP-POSIX.suite local/diffckpt/dCP-standard


You can also run the following equivalent command for this.

.. code-block:: bash

    itf/testdriver -s local/diffckpt/dCP-POSIX.suite local/diffckpt/dCP-standard


Both commands should execute the 4 test cases defined in the suite.
Note that this is equivalent to running the four following commands.

.. code-block:: bash

    itf/testdriver -c local/diffckpt/dCP-standard --iolib 1 --head 0 --mode NOICP
    itf/testdriver -c local/diffckpt/dCP-standard --iolib 1 --head 0 --mode ICP
    itf/testdriver -c local/diffckpt/dCP-standard --iolib 1 --head 1 --mode NOICP
    itf/testdriver -c local/diffckpt/dCP-standard --iolib 1 --head 1 --mode ICP


**Multiple ITF fixtures with their default suites**

ITF supports the execution of multiple fixtures with their default suites.
In this mode, ITF will aggregate the results of all suites under a single execution.
This is useful for obtaining a complete summary for checks spanning multiple FTI features.
For running multiple fixtures, use the default execution flags but append more suite names into the testdriver command.

As an example, run the recovervar and recovername features with the following command.

.. code-block:: bash

    itf/testdriver local/recoverVar/recovervar local/recoverName/recovername


As expected, this command will run the recovervar fixture with all the test cases in its suite.
Then, it will procceed to the recovername tests and performe it's default suite.


Understanding ITF 
-----------------


ITF display information about its API function calls for providing summarized and real-time feedback.
The test application standard output is supressed by default.
However, ITF buffers the test output into a file, and saves it to a log, in case of a test failure.

The `--verbose` family of ITF flags controls what is shown in the terminal.
The folllowing options are available:

- `--quiet`: will supress all ITF output besides the test case parameters and result;
- `--verbose`: will output the test application into the terminal in real time;
- `--verbose=Integer`: can be set to the values 0, 1 and 2.

A verbose value of 0 is equivalent to the quiet flag.
A verbose value of 2 is equivalent to the verbose flag.
A verbose value of 1 is equivalent to the default ITF configuration.

When a message with the format of `fti_config_set` is displayed, it means that a configuration file had it's value changed prior to the execution.
The template FTI configuration file is int `itf/fti_template.cfg`.
Any runtime changes are not persistent and are valid only for the test case in question.

A message with the format of `app_run` informs that a test MPI application has been triggered.
The message is followed by the mpirun command, the application name and it's parameters.
After the application execution, ITF outputs `returns x` where x is the return from the main function.
This information allows for a quick inspection of the application without having to read through the whole application output.


Understanding ITF logs
-----------------


ITF has three types of logs: (i) failed tests log; (ii) all tests log; and (iii) failed test names log.
The log with the failed names is created when a fixture executes a test case which failed.
When this happens, at the end of the fixture run, ITF will display the following message in bold.

.. code-block:: 
    
    Failed tests stdout recorded: logname.log


The logname will contain the path to the log file, which is the fixture name appended with `-failed.log`.
This log will contains the same ITF output as if the testdriver executed with `--verbose`.
However, it will contain only the standard output for the test cases that failed.
This file is the go-to log when debugging your FTI feature executing through an extensive suite.


ITF can also create a logfile for all the tests it executes regardless of their outcome.
This can be done by passing the `--maintain-logs` to the testdriver.
The flag will trigger the creation of another log file with the standard output for all test cases.
Again, the contents are the same as if executing ITF with the `--verbose` flag.
The log name is outputed per-fixture and is exposed after running the fixture with a message like the following.

.. code-block:: 

    All tests stdout recorded: logname.log


The last log is called `itf.log`.
This log is generated if at least one test case failed in the testdriver command.
The log will contain the fixture names where at least one test failed.
After the fixture name, a list with all the test cases is displayed, identified by their parameters.
An example of this log looks like the following.

.. code-block:: 

    dCP-standard
    --iolib 1 --head 0 --mode NOICP
    --iolib 1 --head 0 --mode ICP


This indicates that the dCP-standard fixture failed on two test cases.
You can rerun these tests using the `--custom-params` flag.
Also, it is possible to copy the test cases and create a custom suite for dCP-standard.
As of now, there is no way to re-run automatically all the test cases that failed.
You can contribute with that :)


**Creating a new ITF fixture**

A typical fixture requires a setup, runtest and teardown functions. These functions' definitions are customized depending on the test scanario and the parameters it runs with. Below is a simplistic example from the recover-var fixture's definition:

.. code-block:: bash

    setup() {
        head=0
        keep=0
        param_register 'iolib' 'level'
    }
    runtest() {
        local app='.../build/testing/local/recoverVar/recoverVar.exe'
        app_run_success $app $itf_cfgfile 1 $level 1
        app_run_success $app $itf_cfgfile 0 $level 1
        pass
    }
    teardown() {
        unset head keep
    }


**Declaring test constants and dependencies**

setup function serves to define FTI's constant parameters that will be passed to the configuration file used in all the test cases. It also serves to register any additional/optional variable names that are required in this test. The values passed to these optional variables are specified in the corresponding suite. 

**Preparing the FTI configfile**

prepare_fti function serves to prepare the environment where FTI's code will run. This includes the configuration file and its variables.

**Encapsulating test behavior and checks**

runtest function serves to define the behavior of the test as run by ITF. This includes the test application name, path and parameters if any, additional scripting code that describes the test scenario, and an assertion statement to validate the output of the test. 

**Cleaning up**

teardown function unsets all the variables related to the test. The purpose is to reinitialize ITF's enviroment, preparing it for another test or for a shutdown. 

**Enhancing a fixture with ITF functions**

ITF's APIs are to be found in: testing/itf/api

**Argument-parsing API**

param_register registers names of the arguments that the fixture relies on for its execution. In the case of FTI's default configuration variables (head, failure, io, etc), itf_set_default_variables_to_config function in testing/itf/engine is used instead.

**Configuration file manipulation API**

fti_param_set_inline function sets FTI to perform the all checkpoints inline.

fti_config_set_ckpts function sets the checkpoint intervals of FTI. It takes the required checkpoint intervals as parameters.


**Running an FTI test application**

api_run function runs the MPI application necessary for the test. It takes the application's executable's path and the application's parameters as arguments. Upon execution of the application, it appends the output in the itf_log file. 

app_run_success makes use of api_run but only continues execution if the application succeeds. 

**Checkpoint file disruption API**

ckpt_erase_last function erases the checkpoint objects from the last ITF execution. It expects the name of the object to erase and the node_id from which the object will be erased. 

ckpt_corrupt_last function corrupts the checkpoint objects from the last ITF execution. It expects the name of the object to corrupt and the node_id from which the object will be erased. 

Below is an example from the standard-disrupt fixture:

.. code-block:: bash

    setup() {
        param_register 'iolib' 'level' 'icp' 'diffsize' 'head' 'keep'
        param_register 'disrupt' 'target' 'consecutive' 'expected'

            write_dir='checks'
            mkdir -p $write_dir
        } 
    runtest() {
        local app='.../build/testing/local/standard/check.exe'
        local _crash=0
        if [ $keep -eq 0 ]; then
            # Simulate a crash when not keeping last checkpoint
            local _crash=1
        fi
        app_run_success $app $itf_cfgfile $_crash $level $diffsize $icp $write_dir
        case $consecutive in
        true)
            ckpt_${disrupt}_last "$target" 'node0' 'node1'
            ;;
        false)
            ckpt_${disrupt}_last "$target" 'node0' 'node2'
            ;;
        *)
            ckpt_${disrupt}_last "$target"
            ;;
        esac
        app_run $app $itf_cfgfile 0 $level $diffsize $icp $write_dir
        local retv=$?
        
        if [ $expected == 'success' ]; then
            assert_equals $retv 0 'FTI should succeed in the informed scenario'
        else
            assert_not_equals $retv 0 'FTI should fail in the informed scenario'
        fi
    }


**Assertions API**

ITF's API defines a set of fixture assertions to be called at the end of the test application's execution. This step concludes whether the test passes or fails. These functions are split into three categories: 

-Functions to evaluate the exit code of the application (pass and fail functions)
-Functions to evaluate variables on which the test outcome bases its status by yeilding a fail (check_equals, check_not_equals, check_is_zero, check_non_zero, check_file_exists)
-Functions to evaluate variables and finalize the test by yeilding a pass (assert_equals, assert_not_equals, assert_file_exists). 


**ITF public variables**

ITF's variables file is to be found in: testing/itf/variables

**Extending ITF**

ITF's engine is to be found in: testing/itf/engine

ITF's testdriver is to be found in: testing/itf/testdriver

This file encompasses the general variables needed for the tests to run: FTI variables, MPI applications settings, ITF logs, etc. Any global variables and constants needed by the IT-Framework would be declared here. 

Inside the Engine, test_case function prepares FTI's variables and configuration files, runs the fixture of the given test and cleans up the environment afterwards. 
