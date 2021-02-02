InstallTestApplication("diffSizes.exe" "diffSizes.c")

DeclareITFSuite("verify_log.itf" ${test_labels_current} "verify_log")

DeclareITFSuite("verify_log_disrupt.itf" ${test_labels_current} "verify_log_disrupt")
