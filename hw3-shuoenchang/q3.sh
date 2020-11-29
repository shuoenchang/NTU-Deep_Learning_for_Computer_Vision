bash hw3_p3.sh hw3_data/digits/mnistm/test mnistm outputs/q3/mnistm.csv
python hw3_eval.py outputs/q3/mnistm.csv hw3_data/digits/mnistm/test.csv


echo ''

bash hw3_p3.sh hw3_data/digits/svhn/test svhn outputs/q3/svhn.csv
python hw3_eval.py outputs/q3/svhn.csv hw3_data/digits/svhn/test.csv 

echo ''

bash hw3_p3.sh hw3_data/digits/usps/test usps outputs/q3/usps.csv
python hw3_eval.py outputs/q3/usps.csv hw3_data/digits/usps/test.csv 
 