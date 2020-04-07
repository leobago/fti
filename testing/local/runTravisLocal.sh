 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=1 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=2 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=0 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH0I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I1.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=4 CKPTORPTNER=0 CORRORERASE=1 CORRUPTIONLEVEL=0 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH0I1.fti LEVEL=4 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I1.fti LEVEL=4 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes NOTCORRUPT=1 CONFIG=configH1I0.fti LEVEL=4 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=heatdis CONFIG=configH0I1.fti ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=heatdis CONFIG=configH1I1.fti ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=heatdis CONFIG=configH1I0.fti ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=nodeFlag CONFIG=configH0I1.fti ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=nodeFlag CONFIG=configH1I1.fti ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=nodeFlag CONFIG=configH1I0.fti ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH0I1.fti LEVEL=4 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I1.fti LEVEL=4 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=1 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=2 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=addInArray CONFIG=configH1I0.fti LEVEL=4 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.3 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.4 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.5 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.6 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.7 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.8 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=diffSizes CONFIG=configH1I0.fti LEVEL=3 CKPTORPTNER=0 CORRORERASE=0 CORRUPTIONLEVEL=3 CMAKEVERSION=3.9 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=syncIntv CONFIG=configH1I0.fti ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=hdf5 ./testing/tests.sh >check.log
if [ $? = 0 ]; then echo -e "\033[0;32mpassed\033[m"; else  echo -e "\033[0;31mfailed\033[m Line: $((LINENO-1))"; fi
 TEST=cornerCases ./testing/tests.sh >check.log
