#!/bin/bash

SNUDDA_DIR=$HOME/Snudda/snudda
JOBDIR=networks/test_10k

SIMSIZE=10000

# If the BasalGangliaData directory exists, then use that for our data
#/cfs/klemming/scratch/${USER:0:1}/$USER/BasalGangliaData/data
#BasalGangliaData/Parkinson/PD0
if [[ -d "$HOME/BasalGangliaData/data" ]]; then
    export SNUDDA_DATA="$HOME/BasalGangliaData/data"
    echo "Setting SNUDDA_DATA to $SNUDDA_DATA"
else
    echo "SNUDDA_DATA environment variable not changed (may be empty): $SNUDDA_DATA"
fi

mkdir -p $JOBDIR

if [ "$SLURM_PROCID" -gt 0 ]; then
	mock_string="Not main process"

else

    # For debug purposes:                                                         
    echo "PATH: "$PATH
    echo "IPYTHONDIR: "$IPYTHONDIR
    echo "PYTHONPATH: "$PYTHONPATH
    echo "LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

    echo ">>>>>> Main process starting ipcluster"
    echo

    echo "Start time: " > start_time_input.txt
    date >> start_time_input.txt

    echo "SLURM_NODELIST = $SLURM_NODELIST"
    let NWORKERS="$SLURM_NTASKS - 1"

    echo ">>> NWORKERS " $NWORKERS
    echo ">>> Starting ipcluster `date`"
    

    echo ">>> Input: "`date`
    # cp -a $SNUDDA_DIR/data/input_config/input-v10-scaled.json ${JOBDIR}/input.json
    cp -a $SNUDDA_DIR/data/input_config/external-input-dSTR-scaled-v4.json ${JOBDIR}/input.json

    # snudda input ${JOBDIR} --parallel --time 5
    echo "Temp running in serial -- due to perlmutter bug on Cray https://docs.nersc.gov/development/libraries/hdf5/"
    export HDF5_USE_FILE_LOCKING=FALSE

    snudda input ${JOBDIR} --time 5 --verbose 

    date
    echo "JOB END "`date` start_time_network_connect.txt

fi
