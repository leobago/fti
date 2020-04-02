 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=4 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=4 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=4 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=heatdis CONFIG=configH0I1.fti ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=heatdis CONFIG=configH1I1.fti ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=heatdis CONFIG=configH1I0.fti ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=nodeFlag CONFIG=configH0I1.fti ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=nodeFlag CONFIG=configH1I1.fti ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=nodeFlag CONFIG=configH1I0.fti ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=4 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=4 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=1 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=2 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=4 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.3 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.4 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.5 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.6 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.7 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.8 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.9 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=syncIntv CONFIG=configH1I0.fti ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=hdf5 ./test/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=cornerCases ./test/tests.sh >check.log
