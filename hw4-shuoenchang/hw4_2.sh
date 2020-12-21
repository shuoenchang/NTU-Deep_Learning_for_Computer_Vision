# TODO: create shell script for running your data hallucination model

# Example
python3 test_q2.py --test_csv $1 --test_data_dir $2 --testcase_csv $3 --output_csv $4 \
    --model weights/q2_49.00_m.pth --hall weights/q2_49.00_h.pth