#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source .env.sh
cd $PROJPATH

################################################################################
######################### Running the configs in a loop ########################
################################################################################
#                host      device   rank  size  index     script          config_tree   
CFGPREFIXLIST=("gpub058"  "cuda:0"  "0"   "4"   "0.0"  "vaehist.py"   "02_adhoc/04_klstudy" \
               "gpub058"  "cuda:1"  "1"   "4"   "0.0"  "vaehist.py"   "02_adhoc/04_klstudy" \
               "gpub058"  "cuda:2"  "2"   "4"   "0.0"  "vaehist.py"   "02_adhoc/04_klstudy" \
               "gpub058"  "cuda:3"  "3"   "4"   "0.0"  "vaehist.py"   "02_adhoc/04_klstudy")

BGPIDS=""
trap 'kill $BGPIDS; exit' INT
mkdir -p joblogs
NUMBUSYCUDA=0
for (( i=0; i<${#CFGPREFIXLIST[*]}; i+=7)); do
  MYHOST="${CFGPREFIXLIST[$((i+0))]}"
  MYDEVICE="${CFGPREFIXLIST[$((i+1))]}"
  MYRANK="${CFGPREFIXLIST[$((i+2))]}"
  MYSIZE="${CFGPREFIXLIST[$((i+3))]}"
  MYINDEX="${CFGPREFIXLIST[$((i+4))]}"
  MYPYSRC="${CFGPREFIXLIST[$((i+5))]}"
  MYCFGTREE="${CFGPREFIXLIST[$((i+6))]}"

  myname=$(hostname)
  if [ "${myname%%.*}" != ${MYHOST} ]; then
    continue
  fi

  LOGPSTFIX=$(printf "%02d" ${MYRANK})
  OUTLOG="./joblogs/${MYCFGTREE##*/}_${LOGPSTFIX}.out"
  echo "Running Configuration $MYCFGTREE"
  echo "  ==> The yaml config will be read from ./configs/${MYCFGTREE}.yml"
  echo "  ==> The results hdf file will be saved at ./results/${MYCFGTREE}.h5"
  echo "  ==> The training logs will be saved at ${OUTLOG}"
  
  # Restricting the CUDA visible devices to disallow tsne-cuda's faiss from occupying other devices
  if [[ $MYDEVICE == "cuda:"* ]]; then 
    echo "  + export CUDA_VISIBLE_DEVICES=${MYDEVICE##*:}"
    export CUDA_VISIBLE_DEVICES="${MYDEVICE##*:}"
    MYDEVICE="cuda:0"
  fi

  echo "  + python partnn/${MYPYSRC} -c ${MYCFGTREE} -d ${MYDEVICE} -s ${MYSIZE} -r ${MYRANK} -i ${MYINDEX} > $OUTLOG 2>&1"
  python partnn/${MYPYSRC} -c ${MYCFGTREE} -d ${MYDEVICE} -s ${MYSIZE} -r ${MYRANK} -i ${MYINDEX} > $OUTLOG 2>&1 &
  BGPIDS="${BGPIDS} $!"
  echo "----------------------------------------"

  NUMBUSYCUDA=$((NUMBUSYCUDA+1))
  if (( $NUMBUSYCUDA == $MYSIZE )); then
    wait
    NUMBUSYCUDA=0
  fi

done

wait