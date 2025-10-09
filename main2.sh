#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source .env.sh
cd $PROJPATH

################################################################################
######################### Running the configs in a loop ########################
################################################################################


#                host    NGPUs  Workers  Rank-Start  Rank-End   Resume Index   Total Configs       script          config_tree   
CFGPREFIXLIST=("gpub068"  "4"     "4"       "0"         "4"          "0"           "40"         "vaehist.py"   "02_adhoc/16_mlpcnd" \
               "gpub035"  "4"     "12"      "0"         "4"          "0"           "40"         "vaehist.py"   "02_adhoc/17_mlpcnd" \
               "gpub097"  "4"     "12"      "4"         "8"          "0"           "40"         "vaehist.py"   "02_adhoc/17_mlpcnd" \
               "gpub091"  "4"     "12"      "8"         "12"         "0"           "40"         "vaehist.py"   "02_adhoc/17_mlpcnd")

BGPIDSSPCD=""
trap 'kill $BGPIDSSPCD 2> /dev/null; exit' INT
BGPIDS=","
BGGPUIDXS=","

mkdir -p joblogs
for (( QIDX=0; QIDX<${#CFGPREFIXLIST[*]}; QIDX+=9)); do
  MYHOST="${CFGPREFIXLIST[$((QIDX+0))]}"
  NGPUS="${CFGPREFIXLIST[$((QIDX+1))]}"
  NWORKERS="${CFGPREFIXLIST[$((QIDX+2))]}"
  RANKMIN="${CFGPREFIXLIST[$((QIDX+3))]}"
  RANKMAX="${CFGPREFIXLIST[$((QIDX+4))]}"
  MYINITRSMIDX="${CFGPREFIXLIST[$((QIDX+5))]}"
  MYNCONFIGS="${CFGPREFIXLIST[$((QIDX+6))]}"
  MYPYSRC="${CFGPREFIXLIST[$((QIDX+7))]}"
  MYCFGTREE="${CFGPREFIXLIST[$((QIDX+8))]}"

  myname=$(hostname)
  if [ "${myname%%.*}" != ${MYHOST} ]; then
    continue
  fi

  RNKPIDSARR=()
  for (( RNKIDX=0; RNKIDX<${NWORKERS}; RNKIDX +=1)); do
    RNKPIDSARR+=("-1")
  done

  CFGIDX=${MYINITRSMIDX}
  while [ ${CFGIDX} -lt ${MYNCONFIGS} ]; do
    # Finding the next rank that belongs to this node
    while : ; do
      # Calculating the current rank and job index based on the current config index
      MYRANK=$(($CFGIDX%$NWORKERS))
      MYINDEX=$(($CFGIDX/$NWORKERS))
      if [ ${MYRANK} -lt ${RANKMIN} ]; then
        CFGIDX=$((CFGIDX+1))
      elif [ ${MYRANK} -ge ${RANKMAX} ]; then
        CFGIDX=$((CFGIDX+1))
      else
        # Getting the active and existing background pids
        LATESTRNKPID=${RNKPIDSARR[$MYRANK]}
        # Waiting for my rank's latest PID to finish. You shouldn't run two tasks under 
        # the same rank simultaneosly as they could write to the same file simultaneosly.
        if [[ ${LATESTRNKPID} != "-1" ]]; then
          wait ${LATESTRNKPID}
        fi
        # this rank can now move on!
        break
      fi
    done

    # Finding the next free GPU index
    while : ; do
      # Getting the active and existing background pids
      jobs >> /dev/null 2>&1
      ACTIVEPIDS=$(jobs -p | xargs | sed -e 's/ /,/g')
      # Getting the next free GPU device index
      MYGPUIDX=$(get_nextfreersrcidx ${NGPUS} ${BGPIDS} ${BGGPUIDXS} ,${ACTIVEPIDS})
      # Checking the call status
      if [ $MYGPUIDX -eq -2 ]; then
        # An error happend when evaluating $MYGPUIDX
        echo Failed getting the next free rank
        exit 1
      elif [ $MYGPUIDX -eq -1 ]; then
        # We need to wait as there is no available gpu
        sleep 1
      else
        # we can break and run something!
        break
      fi
    done

    LOGPSTFIX=$(printf "%02d" ${MYRANK})
    OUTLOG="./joblogs/${MYCFGTREE##*/}_${LOGPSTFIX}.out"
    echo "GPU ${MYGPUIDX} is free."
    echo "Running Configuration $MYCFGTREE"
    echo "  ==> The yaml config will be read from ./configs/${MYCFGTREE}.yml"
    echo "  ==> The results hdf file will be saved at ./results/${MYCFGTREE}.h5"
    echo "  ==> The training logs will be saved at ${OUTLOG}"
    
    # Restricting the CUDA visible devices to disallow tsne-cuda's faiss from occupying other devices
    echo "  + export CUDA_VISIBLE_DEVICES=${MYGPUIDX}"
    export CUDA_VISIBLE_DEVICES=${MYGPUIDX}

    # Emptying the log if we're starting from the beginning
    if [ ${CFGIDX} -eq 0 ]; then
      echo "  + rm -f $OUTLOG"
      rm -f $OUTLOG
    fi

    echo "  + python partnn/${MYPYSRC} -c ${MYCFGTREE} -n 1 -d cuda:0 -s ${NWORKERS} -r ${MYRANK} -i "${MYINDEX}.0" >> $OUTLOG 2>&1"
    python partnn/${MYPYSRC} -c ${MYCFGTREE} -n 1 -d cuda:0 -s ${NWORKERS} -r ${MYRANK} -i "${MYINDEX}.0" >> $OUTLOG 2>&1 &

    # Getting the PID of the python call
    LASTPID=$!
    # Appending the last PID to the list of entire background PIDs so far (comma-separated)
    BGPIDS="${BGPIDS},${LASTPID}"
    # Appending to the same list as the last line, except keeping it space-separated
    BGPIDSSPCD="${BGPIDSSPCD} ${LASTPID}"
    # Appending the current rank to the corresponding list of background PIDs
    BGGPUIDXS="${BGGPUIDXS},${MYGPUIDX}"
    # Storing this rank's latest PID
    RNKPIDSARR[${MYRANK}]=${LASTPID}
    echo "----------------------------------------"

    # Preparing for the next round
    CFGIDX=$((CFGIDX+1))
  done
done

wait