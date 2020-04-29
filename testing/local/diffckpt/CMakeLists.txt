enable_testing()

# Install ITF Test Fixtures/Suites
InstallSuiteITFConvention("dCP" ${test_labels_current} "dCP")


# Install MPI Test Application
InstallTestApplication("checkDCPPosix.exe" "checkDCPPosix.c")
InstallTestApplication("diff_test.exe" "diff_test_func.c" "diff_test.c" "diff_test.h")
