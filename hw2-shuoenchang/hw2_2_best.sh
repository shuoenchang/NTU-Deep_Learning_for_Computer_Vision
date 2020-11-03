#!/bin/bash

# Download dataset from Google Drive
FILEID='15ut6qXBk4RPOmp1wyTISWXYZCHFHTrMS'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O 2-2_best_0.700.pth && rm -rf /tmp/cookies.txt

python3 test_q2.py --image_folder $1 --save_folder $2 --best