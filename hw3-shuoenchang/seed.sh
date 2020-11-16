#!/bin/bash
for i in {1..10000}
do
  CUDA_VISIBLE_DEVICES=1 python test_q1.py --image_folder hw3_data/face --save_folder outputs/q1/con --seed $i
done

