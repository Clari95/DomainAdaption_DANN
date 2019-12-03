import os
import torch

import argparser as parser
import models
#import models_best
import data
#import hw3_eval

import cv2
import json

import glob
import csv

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from sklearn.metrics import accuracy_score



def evaluate(model, data_loader, alpha):
    
    args = parser.arg_parse()
    ''' set model to evaluate mode '''
    #model.eval()
    
    preds = []
    gts = [] #ground truth
    print('start evaluate')
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred_class, pred_domain = model(imgs, alpha)
           
            
            _, pred = torch.max(pred_class, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)
        
        
    gts = np.concatenate(gts)
    
    preds = np.concatenate(preds)

    np.save(args.save_dir + 'preds.npy', preds) 		    
    return accuracy_score(gts, preds) #maybe gts preds#, preds#accuracy_score(gts, preds)


if __name__ == '__main__':

    
    args = parser.arg_parse()

    '''csv fild for predictions'''
    print('csv')
    with open(args.save_dir, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(["image_name", "label"])

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    data_test =data.DATA_TEST(args, mode='test')
    #data_SVHN =data.SVHN(args, mode='test')

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=False)
    ''' prepare mode '''
    #load best model
    if(args.target_set == 'svhn'):
      if(args.resume_svhn =='model_target_svhn.pth.tar?dl=1'):
          model = models.Net(args).cuda()
      else:
          model = models_best.Net(args).cuda()
      print('checkpoint schn')
      checkpoint = torch.load(args.resume_svhn)
    
    elif (args.target_set == 'mnistm'):
      if(args.resume_mnistm =='model_target_mnistm.pth.tar?dl=1'):
          model = models.Net(args).cuda()
      else:
          model = models_best.Net(args).cuda()
      print('checkpoint mnistm')
      checkpoint = torch.load(args.resume_mnistm)
    #save predictions

    ''' resume save model '''
    
    model.load_state_dict(checkpoint)
    model.eval()

    #len_dataloader = (len(dataloader_source), len(dataloader_target))
    data_test_iter = iter(test_loader)

   
    #acc = evaluate(model, test_loader)
    #print('Testing Accuracy: {}'.format(acc))
    preds = []
    for idx in range(len(test_loader)):
            
            data_test = data_test_iter.next()
            imgs, _ = data_test
            imgs = imgs.cuda()
            pred_class, pred_domain = model(imgs, alpha=0)
            
            _, pred = torch.max(pred_class, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            
            
            preds.append(pred)
   # print(preds)
    
    #prediction as list --> save in directory: predictions --save_dir
    preds = np.concatenate(preds)

    for idx, pred in enumerate(preds):
        
        if idx<10:
          name = '0000' + str(idx)+ '.png'
        elif idx<100 :
          name = '000' + str(idx)+ '.png'
        elif idx<1000:
          name = '00' + str(idx)+ '.png'
        elif idx<10000:
          name = '0' + str(idx)+ '.png'
        else:
          name = str(idx)+ '.png'
       
        
    
        #csv file for predictions
        
        print(name)
        print(pred)
        with open(args.save_dir, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([name, pred])

        #im = Image.fromarray(pred_img.astype('uint8'))
        #save_pred = os.path.join(args.save_dir, name)

        #im.save(save_pred)
