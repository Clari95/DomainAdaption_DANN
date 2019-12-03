from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DANN')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='data', 
                    help="root path to data directory")
    
    #workers?
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='In homework, please always set to 0')
    parser.add_argument('--epoch', default=30, type=int,
                    help="max number of epochs")
    parser.add_argument('--val_epoch', default=10, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.01, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")

    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')
    
    # resume trained model
    parser.add_argument('--resume_mnistm', type=str, default='', 
                    help="path to the trained model target mnistm")
    parser.add_argument('--resume_svhn', type=str, default='', 
                    help="path to the trained model target svhn")

    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    #else
    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')
    parser.add_argument('--embed_plot_epoch', type= int, default=100, help= 'Epoch number of plotting embeddings.')
   

    parser.add_argument('--target_set', type= str, default='', help= 'model tained on this target')
   
    args = parser.parse_args()

    return args
