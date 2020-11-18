#!/bin/bash
for i in {1..10000}
do
  CUDA_VISIBLE_DEVICES=1 python test_q2.py --save_folder outputs/q2 --seed $i
done

