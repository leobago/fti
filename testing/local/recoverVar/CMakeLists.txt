enable_testing()

# Install ITF Test Fixtures/Suites
#InstallSuiteITFConvention("recovervar" ${test_labels_current} "recovervar")
install(FILES "recovervar.itf" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Install MPI Test Application
InstallTestApplication("recoverVar.exe" "checkRecoverVar.c")
set_property(TARGET "recoverVar.exe" PROPERTY C_STANDARD 99)  
