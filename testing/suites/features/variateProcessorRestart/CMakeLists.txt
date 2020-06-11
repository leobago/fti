enable_testing()

# Install ITF Test Fixtures/Suites
DeclareITFSuite("vpr.itf" ${test_labels_current} "vpr")

# Install FTI Test application
InstallTestApplication("test.exe" "test.c")
