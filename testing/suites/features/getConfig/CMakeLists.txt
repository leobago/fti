enable_testing()

# Install ITF Test Fixtures/Suites
DeclareITFSuite("getconfig.itf" ${test_labels_current} "getconfig")

# Install MPI Test Application
InstallTestApplication("getConfig.exe" "getConfig.c")
set_property(TARGET "getConfig.exe" PROPERTY C_STANDARD 99)  