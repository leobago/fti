InstallTestApplication("syncIntv.exe" "syncIntv.c")

DeclareITFSuite("sync_intv.itf" ${test_labels_current} "sync-interval")
