enable_testing()

# Install ITF Test Fixtures/Suites
DeclareITFSuite("staging.itf" ${test_labels_current} "GIO")

# Install MPI Test Application
InstallTestApplication("massive.exe" "massive.c")
