# wget https://www.dropbox.com/s/skymm88jdu1veaa/model_best.pth.tar?dl=1
#wget https://www.dropbox.com/s/myt9l7d59katg0m/model_target_mnistm.pth.tar?dl=1
#wget https://www.dropbox.com/s/4kled8n69zym1av/model_target_svhn.pth.tar?dl=1
RESUME_svhn='model_target_svhn.pth.tar?dl=1'
RESUME_mnistm='model_target_mnistm.pth.tar?dl=1'
python test.py --resume_mnistm $RESUME_mnistm --resume_svhn $RESUME_svhn --data_dir $1 --target $2 --save_dir $3
