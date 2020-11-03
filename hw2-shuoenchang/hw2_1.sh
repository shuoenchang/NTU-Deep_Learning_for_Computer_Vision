#!/bin/bash

# Download dataset from Google Drive
FILEID='1l53Ai-Nb5T9eUyYSXvXxKrwoUggd6rAA'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O 2-1_0.878.pth && rm -rf /tmp/cookies.txt

python3 test_q1.py --image_folder $1 --save_folder $2