enable_testing()

# Install ITF Test Fixtures/Suites
DeclareITFSuite("normal_run.itf" ${test_labels_current} "normal_run")

DeclareITFSuite("ckpt_disrupt.itf" ${test_labels_current} "ckpt_disrupt")

# Install FTI Test application
InstallTestApplication("check.exe" "check.c")
