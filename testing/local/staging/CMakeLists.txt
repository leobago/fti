enable_testing()

# Install ITF Test Fixtures/Suites
InstallSuiteITFConvention("staging" ${test_labels_current} "GIO")

# Install MPI Test Application
InstallTestApplication("massive.exe" "massive.c")
