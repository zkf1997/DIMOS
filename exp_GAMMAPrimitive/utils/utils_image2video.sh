#!/bin/bash

motion_gen_res="$1"

# if ! [[ "$motion_gen_res" == */ ]]
# then
#     motion_gen_res="$motion_gen_res"/
# fi

for seq in ${motion_gen_res}/*_renderbody/*
do
    echo "$seq"
    ffmpeg -n -r 20 -i "$seq"/img_%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p "$seq".mp4 #-n means skip existing files and do not overwrite
    # ffmpeg -i "$seq".mp4 -pix_fmt rgb24 "$seq".gif
    # rm -rf "$seq"
    # rm -rf "$seq".mp4
done


## example: upto the dataset name
# ./exp_GAMMAPrimitive/utils/utils_image2video.sh results/exp_GAMMAPrimitive/MPVAEPlus_1frame_female_v0/results/seq_gen_seed0_genMPKps/HumanEva