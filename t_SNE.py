import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.manifold import TSNE
import data #import MNISTM, SVHN, USPS
import models
import argparser as parser



if __name__ == "__main__":
    #device = torch.device("cuda") if torch.cuda.is_available() else  torch.device("cpu")
    
    #mnistm_dataset = data.Mnist(image_path="./hw3_data/digits/mnistm/test", label_path="./hw3_data/digits/mnistm/test.csv", test_mode=False, transform=T.ToTensor())
    #svhn_dataset = data.SVHN(image_path="./hw3_data/digits/svhn/test", label_path="./hw3_data/digits/svhn/test.csv", test_mode=False, transform=T.ToTensor())
    args = parser.arg_parse() 

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    mnistm_dataset =data.Mnist(args, mode='train', visualization=True)
    svhn_dataset =data.SVHN(args, mode='train', visualization=True)

    if args.target_set == "mnistm":
        source_dataset = svhn_dataset
        target_dataset = mnistm_dataset
        source = "svhn"
        target = "mnistm"
    elif args.target_set == "svhn":
        source_dataset = mnistm_dataset
        target_dataset = svhn_dataset
        source = "mnistm"
        target = "svhn"
   

    #batch_size = 64
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.test_batch, num_workers=args.workers, shuffle=False)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.test_batch, num_workers=args.workers, shuffle=False)

    
    
    '''tSNE'''
    tsne = TSNE(n_components=2, init="pca")
    X = np.array([]).reshape(0, 800)
    Y_class = np.array([], dtype=np.int16).reshape(0,)
    Y_domain = np.array([], dtype=np.int16).reshape(0,)
    '''model'''
    model = models.Net(args)
    #if torch.cuda.is_available:
       # model.cuda()
     #??
    checkpoint = (torch.load("./modelDANN/model_target_{}.pth.tar".format(target), map_location=torch.device('cpu')))  
    model.load_state_dict(checkpoint)
    model.eval()



    with torch.no_grad():
        steps = len(source_loader)
        for i, data in enumerate(source_loader):
            inputs, classes = data
            #if torch.cuda.is_available:
            #    inputs= inputs.cuda()
            #    classes = classes.cuda()

            outputs = model.feature(inputs).contiguous().view(inputs.size(0), -1).cpu().numpy()
            classes = classes.numpy()
            #outputs =  outputs.numpy()
            #outputs = outputs.cpu()
            
            print(X.shape)
            print(outputs.shape)
            #X = X.to(device)
            print(type(X))
            print(type(outputs))
            X = np.vstack((X, outputs))
            Y_class = np.concatenate((Y_class, classes))
            Y_domain = np.concatenate((Y_domain, np.array([0 for _ in range(inputs.size(0))], dtype=np.int16)))
            
            print("Progress Source steps: [{}/{}]".format(i, steps))
        
        print(X.shape)
        print(Y_class.shape)
        print(Y_domain.shape)

        steps = len(target_loader)
        for i, data in enumerate(target_loader):
            inputs, classes = data
           # inputs= inputs.to(device)
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            outputs = model.feature(inputs).contiguous().view(inputs.size(0), -1).cpu().numpy()
            #outputs = outputs.view(-1, 50 * 4 * 4)
            classes = classes.numpy()

            #outputs = outputs.cpu()
            #X = X.to(device)

            X = np.vstack((X, outputs))
            Y_class = np.concatenate((Y_class, classes))
            Y_domain = np.concatenate((Y_domain, np.array([1 for _ in range(inputs.size(0))], dtype=np.int16)))
            
            print("Target stpes: [{}/{}]".format(i, steps))
        
        print(X.shape)
        print(Y_class.shape)
        print(Y_domain.shape)

    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)


    color = ['r', 'g', 'b', 'k', 'gold', 'm', 'c', 'orange', 'cyan', 'pink']
    class_color = [color[label] for label in Y_class]
    domain_color = [color[label] for label in Y_domain]


    plt.figure(1, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=class_color, s=1)
    plt.savefig("./dann{}_{}_class.png".format(source, target))
    plt.close("all")


    plt.figure(2, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=domain_color, s=1)
    plt.savefig("./dann{}_{}_domain.png".format(source, target))
    plt.close("all")

