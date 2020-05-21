InstallTestApplication("nodeFlag.exe" "nodeFlag.c")
InstallTestApplication("heatdis.exe" "heatdis.c")
InstallTestApplication("addInArray.exe" "addInArray.c")

DeclareITFSuite("integration.itf" ${test_labels_current} "integration")
