enable_testing()

InstallFortTestApplication("ckpt_f_primitives.exe" "ckpt_primitives.f90")
InstallFortTestApplication("complex.exe" "complex.f90")
InstallTestApplication("ckpt_c_primitives.exe" "ckpt_primitives.c")
DeclareITFSuite("datatypes.itf" ${test_labels_current} "datatypes")