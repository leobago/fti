#!/bash/bin

# Variables
MPIRUN="mpirun $MPIRUN_ARGS"
IO_NAMES=(@IO_NAMES@)
LEVEL=(1 2 3 4)

# Other variables default values
CFG_FILE=runchecks.cfg
diffSize=0
TIMEOUT=30
DFLAG=1
CFLAG=1
PROCS=16
diffSize=0
verbose=0
eraseFiles=0
corruptFiles=0
FAILED=0
SUCCEED=0
FAULTY=0
testFailed=0
io_mode=0

# Bash input processing
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d|--diff-size)
    diffSize=1
    echo "[OPTION] Set different checkpoint sizes -> TRUE"
    ;;
    -v|--verbose)
    verbose=1
    echo "[OPTION] Set verbose mode -> TRUE"
    ;;
    -e|--erase-files)
    eraseFiles=1
    echo "[OPTION] Set erase checkpoint files -> TRUE"
    ;;
    -c|--corrupt-files)
    corruptFiles=1
    echo "[OPTION] Set corrupt checkpoint files -> TRUE"
    ;;
    -t|--set-timeout)
    if [[ $2 -lt "10" ]] || [[ $2 == -* ]]; then
        echo -e "Wrong argument for timeout: "$2
        echo -e "usage: [command] -t <integer -ge 10>"
        exit 0
    fi
    TIMEOUT="$2"
    echo "[OPTION] Set timeout -> "$TIMEOUT
    shift
    ;;
    -h|--help)
    display_usage
    exit 0
    ;;
esac
shift # past argument or value
done