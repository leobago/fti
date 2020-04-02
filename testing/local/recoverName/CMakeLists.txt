enable_testing()

# Install ITF Test Fixtures/Suites
InstallSuiteITFConvention("recovername" ${test_labels_current} "recovername")

# Install MPI Test Application
InstallTestApplication("recoverName.exe" "checkRecoverName.c")
set_property(TARGET recoverName.exe PROPERTY C_STANDARD 99)  
