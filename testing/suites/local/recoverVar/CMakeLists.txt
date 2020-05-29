enable_testing()

# Install ITF Test Fixtures/Suites
DeclareITFSuite("recovervar.itf" ${test_labels_current} "recovervar")

# Install MPI Test Application
InstallTestApplication("recoverVar.exe" "checkRecoverVar.c")
set_property(TARGET "recoverVar.exe" PROPERTY C_STANDARD 99)  
