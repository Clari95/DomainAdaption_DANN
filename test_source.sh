
#RESUME_svhn='model_target_svhn.pth.tar?dl=1'
RESUME_mnistm='/content/log/model_25.pth.tar'
python test_source.py --resume_mnistm $RESUME_mnistm  --data_dir $1 --target $2 --save_dir $3
