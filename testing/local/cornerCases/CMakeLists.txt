InstallTestApplication("consistency.exe" "consistency.c")
InstallTestApplication("ckptHierarchy.exe" "ckptHierarchy.c")

DeclareITFSuite("corner_cases.itf" ${test_labels_current} "corner-cases")
