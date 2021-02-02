enable_testing()

# Install ITF Test Fixtures/Suites
DeclareITFSuite("cppoperability.itf" ${test_labels_current} "cppoperability")

# Install MPI Test Application
InstallTestApplication("cppOperability.exe" "cppOperability.cpp")
set_property(TARGET "cppOperability.exe" PROPERTY C_STANDARD 99)  