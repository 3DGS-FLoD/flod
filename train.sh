device=0
port=6090


#* MipNerf360 
# MipNerf360 Outdoor
scenes="bicycle flowers garden stump treehill"

for scene in $scenes; do  

    outdir="./output/mipnerf360/${scene}"
    mkdir -p $outdir

    CUDA_VISIBLE_DEVICES=$device python train.py \
     -s /dataset/mipnerf360/${scene} --eval -r 4 \
     -m $outdir --port $port --data_device cuda 
done

# MipNerf360 Indoor
scenes="room counter kitchen bonsai"

for scene in $scenes; do  

    outdir="./output/mipnerf360/${scene}"
    mkdir -p $outdir

    CUDA_VISIBLE_DEVICES=$device python train.py \
     -s /dataset/mipnerf360/${scene} --eval -r 2 \
     -m $outdir --port $port --data_device cuda 
done



#* Tanks and Temples
scenes="train truck"
for scene in $scenes; do  

    outdir="./output/tnt/${scene}"
    mkdir -p $outdir

    CUDA_VISIBLE_DEVICES=$device python train.py \
     -s /dataset/tandt/${scene} --eval -r 2 \
     -m $outdir --port $port --data_device cuda 
done



#* DL3DV-10K
#DL3DV-10K/2K/
scenes="9f518d266943220090b58a2fc950772767a6eba1162fb49ff721cbdabeba3b95"
for scene in $scenes; do  

    outdir="./output/dl3dv/${scene}"
    mkdir -p $outdir

    CUDA_VISIBLE_DEVICES=$device python train.py \
     -s /dataset/DL3DV-10K/2K/${scene} --eval -r 2 \
     -m $outdir --port $port --data_device cuda 
done

#DL3DV-10K/1K/
scenes="aeb33502d50088e27d6b8e2bf0fcd2f7e89c5dee893d5807afded09e65b91489 9df87dfc4c40915e0252b622555403743464a87c4ddc03f69cd0a91ffefbf409 58e78d9c8246767555662dc42f67c130bf2fc0d88dac4c45c15097579860d923 ce06045bca7391ab81b177194bed130530f6d6ccca05b8b136a00feaf221602f 2bfcf4b343836f63570c24c28b5937b0872e4ff7dba00ef60dcbae285c2c494a"
for scene in $scenes; do  

    outdir="./output/dl3dv/${scene}"
    mkdir -p $outdir

    CUDA_VISIBLE_DEVICES=$device python train.py \
     -s /dataset/DL3DV-10K/1K/${scene} --eval -r 2 \
     -m $outdir --port $port --data_device cuda 
done

