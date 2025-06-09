#!/bin/bash

device=0

#* MipNerf360
scenes="bicycle flowers garden stump treehill room counter kitchen bonsai"
for scene in $scenes; do  
    outdir="./output/mipnerf360/${scene}"

    # single_level for render_lod 5 4 3 2 1
    for render_lod in 5 4 3 2 1; do
        # First: without --save_video
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode single_level -m $outdir --lod 5 --render_lod $render_lod --skip_train
        # Second: with --save_video
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode single_level -m $outdir --lod 5 --render_lod $render_lod --skip_train --save_video
    done

    # selective rendering for (hlod_max, hlod_min) in (5,3), (4,2), (3,1)
    for pair in "5 3" "4 2" "3 1"; do
        set -- $pair
        hlod_max=$1
        hlod_min=$2
        # per_view_selective_rendering
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_per_view -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_per_view -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0 --save_video
        # predetermined_selective_rendering
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_predetermined -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_predetermined -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0 --save_video
    done
done

#* Tanks and Temples
scenes="train truck"
for scene in $scenes; do  
    outdir="./output/tnt/${scene}"

    # single_level for render_lod 5 4 3 2 1
    for render_lod in 5 4 3 2 1; do
        # First: without --save_video
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode single_level -m $outdir --lod 5 --render_lod $render_lod --skip_train
        # Second: with --save_video
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode single_level -m $outdir --lod 5 --render_lod $render_lod --skip_train --save_video
    done

    # selective rendering for (hlod_max, hlod_min) in (5,3), (4,2), (3,1)
    for pair in "5 3" "4 2" "3 1"; do
        set -- $pair
        hlod_max=$1
        hlod_min=$2
        # per_view_selective_rendering
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_per_view -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_per_view -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0 --save_video
        # predetermined_selective_rendering
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_predetermined -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_predetermined -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0 --save_video
    done
done

#* DL3DV-10K
scenes="9f518d266943220090b58a2fc950772767a6eba1162fb49ff721cbdabeba3b95 aeb33502d50088e27d6b8e2bf0fcd2f7e89c5dee893d5807afded09e65b91489 9df87dfc4c40915e0252b622555403743464a87c4ddc03f69cd0a91ffefbf409 58e78d9c8246767555662dc42f67c130bf2fc0d88dac4c45c15097579860d923 ce06045bca7391ab81b177194bed130530f6d6ccca05b8b136a00feaf221602f 2bfcf4b343836f63570c24c28b5937b0872e4ff7dba00ef60dcbae285c2c494a"
for scene in $scenes; do  
    outdir="./output/dl3dv/${scene}"

    # single_level for render_lod 5 4 3 2 1
    for render_lod in 5 4 3 2 1; do
        # First: without --save_video
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode single_level -m $outdir --lod 5 --render_lod $render_lod --skip_train
        # Second: with --save_video
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode single_level -m $outdir --lod 5 --render_lod $render_lod --skip_train --save_video
    done

    # selective rendering for (hlod_max, hlod_min) in (5,3), (4,2), (3,1)
    for pair in "5 3" "4 2" "3 1"; do
        set -- $pair
        hlod_max=$1
        hlod_min=$2
        # per_view_selective_rendering
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_per_view -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_per_view -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0 --save_video
        # predetermined_selective_rendering
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_predetermined -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0
        CUDA_VISIBLE_DEVICES=$device python render.py \
            --render_mode selective_rendering_predetermined -m $outdir --lod 5 --hlod_max $hlod_max --hlod_min $hlod_min --skip_train --hlod_screensize 1.0 --save_video
    done
done
