enable_testing()

# Install ITF Test Fixtures/Suites
DeclareITFSuite("standard.itf" ${test_labels_current} "standard")

# Install FTI Test application
InstallTestApplication("check.exe" "check.c")
