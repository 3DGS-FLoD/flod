
runtype="runs"
scenes="bicycle bonsai counter garden room stump kitchen"

for scene in $scenes; do  

    outdir="./output/${runtype}/mipnerf/${scene}"
    mkdir -p $outdir

    python train.py \
     -s /path/to/mipnerf360_dataset/${scene} --eval -r 4 \
     -m $outdir --port $port --data_device cuda \
     
done