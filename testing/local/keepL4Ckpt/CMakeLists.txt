enable_testing()

# Install ITF Test Fixtures/Suites
InstallSuiteITFConvention("keepl4" ${test_labels_current} "keepl4")

# Install MPI Test Application
InstallTestApplication("testkeepl4.exe" "test.c")
