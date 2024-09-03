runtype="runs"  
scenes="9f518d266943220090b58a2fc950772767a6eba1162fb49ff721cbdabeba3b95 aeb33502d50088e27d6b8e2bf0fcd2f7e89c5dee893d5807afded09e65b91489 9df87dfc4c40915e0252b622555403743464a87c4ddc03f69cd0a91ffefbf409 58e78d9c8246767555662dc42f67c130bf2fc0d88dac4c45c15097579860d923 ce06045bca7391ab81b177194bed130530f6d6ccca05b8b136a00feaf221602f 2bfcf4b343836f63570c24c28b5937b0872e4ff7dba00ef60dcbae285c2c494a"

for scene in $scenes; do

    outdir="./output/${runtype}/dl3dv/${scene}"
    mkdir -p $outdir

    python train.py \
        -s /path/to/DL3DV_dataset/${scene} --eval -r 2 \
        -m $outdir --port $port --data_device cuda \

done  
