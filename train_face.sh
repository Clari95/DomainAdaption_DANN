VALEPOCH=1
LR=0.0002
python train_sourceclass.py --data_dir $1 --val_epoch $VALEPOCH --lr $LR
