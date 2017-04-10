#wget -q http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/public.gpg -O- | sudo apt-key add -
#echo "deb http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/ trusty main" | sudo tee /etc/apt/sources.list.d/ibm-xl-compiler-eval.list

#sudo apt-get update -qq

echo "deb http://public.dhe.ibm.com/software/server/POWER/Linux/rte/xlcpp/le/ubuntu $(lsb_release -s -c) main" | sudo tee -a /etc/apt/sources.list.d/ibm-xlcpp-rte.list

sudo apt-get update -qq
sudo apt-get install libxlc

ls -l /opt/ibm
ls -l /opt/ibm/xlC
ls -l /opt/ibm/xlf

sudo /opt/ibm/xlC/13.1.5/bin/xlc_configure

sudo /opt/ibm/xlf/15.1.5/bin/xlf_configure