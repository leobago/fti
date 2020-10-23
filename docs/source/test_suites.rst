.. Fault Tolerance Library documentation test suites

Test Suites
===================================================

FTI is bundled with a set of testing programs and scripts to execute them.
The combination of script and a testing program is called a **test suite**.
The test suites validate, by design, a single FTI feature under different conditions.
Moreover, FTI contains several test suites to achieve higher software quality.

The **test suites** scripts are developed using the **FTI Integration Test Framework (ITF)**.
In this format, **test functions** define how to validate different aspects of the target feature.
This is done by creating multiple **test cases** by variating parameters that might affect the feature's behavior.
For more information about how to implement these concepts, we recommend reading the :doc:`ITF documentation <itf>`.
This page is devoted to explaining which test suites exist and what they validate.

We divide the test suite in three **test categories**: 
(i) core functionalities; 
(ii) additional features and 
(iii) compilation/build. 
Indeed, the suites are grouped by the type of feature they validate.
This organization is expressed in the testing folder hierarchy.
Hence, the suites are bundled in hierarchical directories with the root in *testing/suites*.
A brief description of every suite in each category can be found in the table below. The latter groups these tests by their
corresponding Jenkins pipeline step. It lists the tests (sub-suites) executed within each suite, and it mentions the checkpoint interval
used for tests where checkpoints are taken using FTI_Snapshot(). The checkpoint interval is not applicable for tests that rely on FTI_Checkpoint().
Note that the latter are respective to [L1, L2, L3, L4] levels. 

+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
|   **Suite Name**   |         **Focus on testing**        | **Suite Cases** |    **Sub-suites**         |  **Ckpt intervals**         |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
|                          **Standard pipeline step**                                                                                  |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **multiLevelCkpt** | Multi-level checkpointing           |       760       | ckpt_disrupt:440          |  N/A                        |
|                    |                                     |                 | ckpt_disrupt:320          |                             |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
|                          **DiffSizes pipeline step**                                                                                 |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **ckptDiffSizes**  | Different checkpoint sizes per rank |       96        | verify_log_disrupt:84     |  N/A                        |
|                    |                                     |                 | verify_log:12             |                             |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
|                        **Features pipeline step**                                                                                    |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **cornerCases**    | Supported corner cases              |       56        | ckpt_consistency:12       |  N/A                        |
|                    |                                     |                 | keep_last_consistency:12  |                             |
|                    |                                     |                 | double_fti_init:8         |                             |
|                    |                                     |                 | subsequent_checkpoints:15 |                             |
|                    |                                     |                 | subsequent_ckpts_restart:9|                             |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **useCases**       | Simplified use-case scenarios       |       18        | nodeflag:3                |[1,2,3,4] with fast_forward=5|
|                    |                                     |                 | simulated_use_cases:15    |                             |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **syncIntv**       | Runtime-aware checkpoint interval   |       1         |                           |[1,2,3,4] with fast_forward=5|
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **keepL4Ckpt**     | Archive checkpoint in PFS           |       4         |                           |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **dCP**            | Differential Checkpointing          |       10        | corrupt_check:2           |  N/A                        |
|                    |                                     |                 | standard:8                |                             |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **vpr**            | Variate Processor Restart           |       8         |                           |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **recoverName**    | Recover Variable per Name           |       16        |                           |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **recoverVar**     | Recover Variable per Id             |       20        |                           |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **staging**        | Staging Feature                     |       2         |                           |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **getConfig**      | Manipulate FTI configurations       |       5         |                           |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **hdf5**           | HDF5 support and sanity checks      |       12        |                           |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **datatypes**      | Support for Fortran datatypes       |       10        | fortran_to_c_map:5        |  N/A                        |
|                    |                                     |                 | fortran_complex:5         |                             |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
|                         **Compilation and Build Suite**                                                                              |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+
| **cmake_versions** | Build with different CMake versions |       13        | No                        |  N/A                        |
+--------------------+-------------------------------------+-----------------+---------------------------+-----------------------------+



Core functionality test category
-----------------

The *core* category is attributed to test suites that validate the main FTI behavior.
In other words, this category applies mainly to the multi-level checkpoint feature.
Furthermore, we added into this category other FTI behaviors that support the multi-level functionality.

Multi-level checkpoint
~~~~~~~~~~~~~~~~~

The multi-level checkpoint suite is located in the *testing/suites/core/multiLevelCkpt* folder.
The ITF suite file is declared under the name *standard.itf*.
The suite is composed of two testing functions: (i) **normal_run** and (ii) **ckpt_disruption**.

The *normal_run* function checks the expected behavior of FTI under normal circumstances.
The function simulates an application crashing and re-starting to verify FTI checkpoint/restart behavior.

The *ckpt_disruption* function checks the expected behavior of FTI when the checkpoint files are disrupted (i.e erased or corrupted).
It follows the same flow as *normal_run* but the checkpoint files are disrupted before the second application run.
This function simulates scenarios where FTI is supposed to both fail and succeed in recovering the application state.

.. warning::  The tests which cause FTI to fail are currently disabled in the CI environment due to unexpected MPI hanging.

FTI corner cases
~~~~~~~~~~~~~~~~~

The corner cases suite is located in the *testing/suites/core/cornerCases* folder.
The ITF suite file is declared under the name *corner_cases.itf*.
The suite is composed of corner case scenarios regarding the consistency and hierarchy of checkpoint files.
There are three scenarios regarding the consistency aspect represented as the test functions: 
(i) *ckpt_consistency*; 
(ii) *keep_last_consistency* and 
(iii) *double_fti_init*.

The *ckpt_consistency* tests check if FTI creates consistent checkpoint **and** partner files.
The checks validate if FTI can recover from one group of files when the other is corrupted.
Then, a new set of checkpoint files is created and another group is corrupted.
The test validates if all application states are consistent, regardless of the recovery strategy.

The *keep_last_consistency* tests is similar to *ckpt_consistency*.
However, instead of simulating crashes to test the recovery, the application finishes and stores its last checkpoint on the PFS.
Then, the function asserts that FTI uses the last checkpoint from PFS when a re-run is issued with the same configuration file.

The *double_fti_init* asserts that FTI is capable of functioning if the initialization function is called twice.
Moreover, this function mimics a live restart and/or protection of individual application segments.

The remainder test functions are related to the hierarchical relationship between checkpoint levels.
There are two test functions targeting these corner cases:
(i) *subsequent_checkpoints* and 
(ii) *subsequent_ckpts_restart*.
FTI is expected to overwrite less recent files depending on the order the checkpoints are taken.
Hence, the former function asserts that FTI maintains the most secure checkpoint after taking subsequent checkpoints.
Finally, the *subsequent_ckpts_restart* function asserts that FTI restores from the most recent non-corrupted checkpoint.

FTI use cases
~~~~~~~~~~~~~~~~~

The use cases suite is located in the *testing/suites/core/useCases* folder.
The ITF suite file is declared under the name *use_cases.itf*.
The suite is composed of three applications that simulate a simplified use case for FTI.
These tests can be considered as true integration tests given that they are based on mini-kernels.
There are two test functions on this test case: 
(i) *nodeflag* and
(ii) *simulated_use_cases*.

Synchronization interval
~~~~~~~~~~~~~~~~~

The synchronization interval suite is located in the *testing/suites/core/syncIntv* folder.
The ITF suite file is declared under the name *sync_intv.itf*.
It contains only one function, *checkpoint_interval*.
This test executes a 3d heat distribution kernel.
Furthermore, the function asserts that checkpoints are taken in the correct application iterations and time intervals.


Ranks with different checkpoint sizes
~~~~~~~~~~~~~~~~~

The *ckptDiffSizes* suite is located in the *testing/suites/core/ckptDiffSizes* folder.
The ITF suite file is declared under the name *diff_sizes.itf*.
This suite checks if FTI is capable of checkpointing ranks with different checkpoint sizes.
It contains two test functions:
(i) *verify_log* and 
(ii) *verify_log_disrupt*.
Both functions use FTI logs to assert that all the data is being checkpointed regardless of the difference in size.
The latter check also adds disruption to the checkpoint files between application runs.


Keep level 4 checkpoints
~~~~~~~~~~~~~~~~~


The *keepL4Ckpt* suite is located in the *testing/suites/core/keepL4Ckpt* folder.
The ITF suite file is declared under the name *keepl4.itf*.
It contains a single test function, *standard*.
The function asserts that FTI pushes the L4 checkpoint into an archive when configured to do so.


Additional features test category
-----------------


The *feature* test category applies to test suites that validate FTI features beyond the scope of the main checkpoint/restart feature.
Those are variations for API functions, support for IO libraries, and other non-essential functionalities.
Test suites that adhere to this category are located under the *testing/suites/features* folder.


Differential Checkpointing
~~~~~~~~~~~~~~~~~


The differential checkpoint suite is located in the *testing/suites/features/differentialCkpt* folder.
The ITF suite file is declared under the name *dCP.itf*.
It contains two test functions:
(i) *standard* and
(ii) corrupt_check;

The *standard* test function asserts the differential checkpoint encodes the correct amount of data.
The *corrupt_check* function asserts that FTI can recover from corrupted differential checkpoint data.

.. note::  The *standard* function implements the checks for POSIX and FTI IO modes.


Variate Processor Restart
~~~~~~~~~~~~~~~~~


The variate processor restart suite is located in the *testing/suites/features/variateProcessorRestart* folder.
The ITF suite file is declared under the name *vpr.itf*.
It contains one test function, *standard*.

The *standard* function asserts that FTI is capable of restarting an application in a different number of ranks.

.. note::  The *standard* function only verifies the behavior for the HDF5 IO library.


Recover variable by name
~~~~~~~~~~~~~~~~~


The *recover-name* suite is located in the *testing/suites/features/recoverName* folder.
The ITF suite file is declared under the name *recovername.itf*.
It contains one test function, *standard*.
The function asserts that FTI can correctly recover variables given their name.

.. warning::  This functionality is not enabled for FTI IO mode and is disabled in the CI environment.


Recover variable by id
~~~~~~~~~~~~~~~~~


The *recover-var* suite is located in the *testing/suites/features/recoverVar* folder.
The ITF suite file is declared under the name *recovervar.itf*.
It contains one test function, *standard*.
The function asserts that FTI can correctly recover variables given a numeric id.


Staging API
~~~~~~~~~~~~~~~~~


The *staging* suite is located in the *testing/suites/features/staging* folder.
The ITF suite file is declared under the name *staging.itf*.
It contains one test function, *standard*.
The function asserts the correct functioning of the staging functionality.
In other words, it asserts that FTI can push files to the PFS in the background as requested by the application.


GetConfig API
~~~~~~~~~~~~~~~~~


The *GetConfig* suite is located in the *testing/suites/features/getConfig* folder.
The ITF suite file is declared under the name *getconfig.itf*.
It contains one test function, *standard*.
This test asserts that FTI can retrieve the configuration file contents during runtime.


HDF5 support
~~~~~~~~~~~~~~~~~


The *hdf5* suite is located in the *testing/suites/features/hdf5* folder.
The ITF suite file is declared under the name *hdf5.itf*.
It contains onde test functions, *hdf5_test*.
This test asserts that FTI yields correct HDF5 structures when issuing HDF5 checkpoint files.


Compilation test category
-----------------


The *compilation* test category applies to test suites that validate the FTI build process.
Test suites that adhere to this category are located under the *testing/suites/compilation* folder.
As of now, there is only one test suite in this category: **cmake_versions**.

The *CMake versions* test suite is used to test FTI compilation under different CMake versions.
It is used to guarantee the build process portability from the minimum CMake required version up to more recent ones.
This test is tailored to function in the FTI CI environment.
Thus, reproducibility will involve changing the behavior of the test so it can find the installed CMake binaries.
