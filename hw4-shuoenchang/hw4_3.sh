# TODO: create shell script for running your improved data hallucination model

# Example
python3 test_q2.py --test_csv $1 --test_data_dir $2 --testcase_csv $3 --output_csv $4 \
    --model weights/q3_49.82_m.pth --hall weights/q3_49.82_h.pth
