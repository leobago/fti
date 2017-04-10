#wget -q http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/public.gpg -O- | sudo apt-key add -
#echo "deb http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/ trusty main" | sudo tee /etc/apt/sources.list.d/ibm-xl-compiler-eval.list

#sudo apt-get update -qq
#sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 4B19F6F50761C815

#echo "http://public.dhe.ibm.com/software/server/POWER/Linux/rte/xlcpp/le/ubuntu/dists/xenial/main" | sudo tee -a /etc/apt/sources.list.d/ibm-xlcpp-rte.list
wget ftp://public.dhe.ibm.com/software/server/POWER/Linux/rte/xlcpp/le/ubuntu/dists/xenial/main/binary-ppc64el/libxlc*

sudo dpkg -iG *.deb 
#sudo apt-get update
#sudo apt-get install libxlc

/usr/vacpp/bin/xlC -qversion

ls -l /opt/ibm
ls -l /opt/ibm/xlC
ls -l /opt/ibm/xlf

sudo /opt/ibm/xlC/13.1.5/bin/xlc_configure

sudo /opt/ibm/xlf/15.1.5/bin/xlf_configure