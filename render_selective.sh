outdir="/path/to/your/model"

# selective rendering using level 5, 4, and 3
python render_selective.py \
    -m $outdir --lod 5 --hlod_max 5 --hlod_min 3 --metric --skip_train

# selective rendering using level 5, 4, and 3 with smaller screen size threshold (hlod_screensize)
python render_selective.py \
    -m $outdir --lod 5 --hlod_max 5 --hlod_min 3 --metric --skip_train --hlod_screensize 0.7 \

# selective rendering using level 4, 3, and 2
python render_selective.py \
    -m $outdir --lod 5 --hlod_max 4 --hlod_min 2 --metric --skip_train

# selective rendering using level 3, 2, and 1
python render_selective.py \
    -m $outdir --lod 5 --hlod_max 3 --hlod_min 1 --metric --skip_train