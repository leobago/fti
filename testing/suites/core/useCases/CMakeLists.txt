InstallTestApplication("nodeFlag.exe" "nodeFlag.c")
InstallTestApplication("heatdis.exe" "heatdis.c")
InstallTestApplication("addInArray.exe" "addInArray.c")

DeclareITFSuite("use_cases.itf" ${test_labels_current} "use_cases")
