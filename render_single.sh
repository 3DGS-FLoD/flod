lods="1 2 3 4 5"
 
for lod in $lods; do

    outdir="/path/to/your/model"

    python render_single_level.py \
     -m $outdir --lod 5 --render_lod $lod --metric --skip_train

done 