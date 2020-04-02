enable_testing()

# Install ITF Test Fixtures/Suites
InstallSuiteITFConvention("vpr" ${test_labels_current} "vpr")

# Install FTI Test application
InstallTestApplication("test.exe" "test.c")
