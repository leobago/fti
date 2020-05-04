enable_testing()

# Install ITF Test Fixtures/Suites
InstallSuiteITFConvention("standard" ${test_labels_current} "standard")

# Install FTI Test application
InstallTestApplication("check.exe" "check.c")
