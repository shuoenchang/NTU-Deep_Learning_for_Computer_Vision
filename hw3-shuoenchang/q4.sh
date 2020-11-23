python test_q4.py --dataset mnistm --model_name mnistm
python test_q4.py --dataset mnistm --model_name usps-mnistm
python test_q4.py --dataset mnistm --model_name usps

echo $'\n'

python test_q4.py --dataset svhn --model_name svhn
python test_q4.py --dataset svhn --model_name mnistm-svhn
python test_q4.py --dataset svhn --model_name mnistm

echo $'\n'

python test_q4.py --dataset usps --model_name usps
python test_q4.py --dataset usps --model_name svhn-usps
python test_q4.py --dataset usps --model_name svhn