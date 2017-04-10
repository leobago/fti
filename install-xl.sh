wget -q http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/public.gpg -O- | sudo apt-key add -
echo "deb http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/ trusty main" | sudo tee /etc/apt/sources.list.d/ibm-xl-compiler-eval.list
sudo apt-get update

sudo apt-get install xlc.13.1.5
sudo /opt/ibm/xlC/13.1.5/bin/xlc_configure

sudo apt-get install xlf.15.1.5
sudo /opt/ibm/xlf/15.1.5/bin/xlf_configure