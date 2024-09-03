runtype="runs"
scenes="train truck"

for scene in $scenes; do  

    outdir="./output/${runtype}/tnt/${scene}"
    mkdir -p $outdir

    python train.py \
     -s /path/to/tandt_dataset/${scene} --eval -r 2 \
     -m $outdir --port $port --data_device cuda \
     
done