import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.optim as optim
import torch.utils.data

import argparser as parser
import model_2 as models
import data 
import test_source as test

import numpy as np

#from test import evaluate

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)


if __name__=='__main__':

    args = parser.arg_parse()
    
    data_mnist =data.Mnist(args, mode='train')
    data_SVHN =data.SVHN(args, mode='train')
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    dataloader_source = torch.utils.data.DataLoader(data_mnist,
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    dataloader_target = torch.utils.data.DataLoader(data_SVHN,
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
   # val_loader = torch.utils.data.DataLoader(data_c.DATA(args, mode='val'),
   #                                            batch_size=args.train_batch, 
   #                                            num_workers=args.workers,
   #                                            shuffle=False)

    print(args.lr)
    print(args.epoch)
    print(args.train_batch)


    ''' load model '''
    print('===> prepare model ...')
   # print(args.size())
    model = models.Net(args)

    #checkpoint = torch.load('./log/model_best.pth.tar')
    #model.load_state_dict(checkpoint)

    ###Hier auskomentieren##
    model.cuda() # load model to gpu

    #print(model)

    ''' define loss '''
    criterion_class = nn.CrossEntropyLoss()
    #criterion_domain = nn.CrossEntropyLoss()

    criterion_class = criterion_class.cuda()
    #criterion_domain = criterion_domain.cuda()


    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))


    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):

        model.train()

        len_dataloader = len(dataloader_source)
        data_source_iter = iter(dataloader_source)
        #data_target_iter = iter(dataloader_target)
        #print(len_dataloader)
        for idx in range(len_dataloader): #range

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len_dataloader)
            iters += 1

            p = float(idx + epoch * len_dataloader) / args.epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            #next(iter(data_mnist))
            ''' move data to gpu '''
            ###hier auskommentieren
            #imgs, labels = imgs.cuda(), labels.cuda()
            
            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source
            s_img, s_label = s_img.cuda(), s_label.cuda()
            
            #batchsize changes for the last epoch because of different size of datatsets.
           # bs = s_img.size(0)
           # domain_label = torch.zeros(bs)
           # domain_label = domain_label.long()
           # domain_label = domain_label.cuda()


            ''' forward path source'''
            #print(size(s_img))
            #print(alpha)
            class_output = model(s_img, alpha)
            

            loss_class = criterion_class(class_output, s_label)
           # err_s_domain = criterion_domain(domain_output, domain_label)

        

           
            
            ''' backpropagation, update parameters '''
            optimizer.zero_grad()         # set grad of all parameters to zero
            loss_class.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss_class.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss_class.data.cpu().numpy())

            print(train_info)
           
        print(epoch%args.val_epoch)
        if epoch%args.val_epoch == 0:
            print('next epoch')
            ''' evaluate the model '''
            acc = test.evaluate(model, dataloader_target, alpha)
            writer.add_scalar('val_acc', acc, iters)
            #print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            #if acc > best_acc:
            #    save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
            #    best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
