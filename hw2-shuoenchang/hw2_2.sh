#!/bin/bash

# Download dataset from Google Drive
FILEID='19H95s6Sf6OsfapWZ11muRXetV86hTx4v'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O weights/2-2_0.689.pth && rm -rf /tmp/cookies.txt

python3 test_q2.py --image_folder $1 --save_folder $2