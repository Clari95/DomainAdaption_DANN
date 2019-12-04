import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.optim as optim
import torch.utils.data

import argparser as parser
import model_face
import data 
import test

import numpy as np

#from test import evaluate

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__=='__main__':

    args = parser.arg_parse()
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of feature maps in discriminator
    ndf = 64
      
    
    data_set =data.face(args, mode='train')
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    dataloader = torch.utils.data.DataLoader(data_set,
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
     # val_loader = torch.utils.data.DataLoader(data_c.DATA(args, mode='val'),
   #                                            batch_size=args.train_batch, 
   #                                            num_workers=args.workers,
   #                                            shuffle=False)
    #print(type(dataloader))
    #real_batch = next(iter(dataloader.numpy()))
    #plt.figure(figsize=(8,8))
    #plt.axis("off")
    #plt.title("Training Images")
    #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    print(args.lr)
    print(args.epoch)
    print(args.train_batch)


    ''' load model '''
    print('===> prepare model ...')
   # print(args.size())
    modelG = model_face.Generator(args)

    #checkpoint = torch.load('./log/model_best.pth.tar')
    #model.load_state_dict(checkpoint)

    ###Hier auskomentieren##
    modelG.cuda() # load model to gpu
    modelG.apply(weights_init)
    print(modelG)

    modelD = model_face.Discriminator(args)
    modelD.cuda() # load model to gpu
    modelD.apply(weights_init)
    print(modelD)

    ''' define loss '''
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

   
    


    ''' setup optimizer '''
    beta1 = 0.5
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(modelD.parameters(), lr=args.lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(modelG.parameters(), lr=args.lr, betas=(beta1, 0.999))
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))


    ''' train model '''
    print('===> start training ...')
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):

        #model.train()

        for idx in range(len(dataloader)): #range

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(dataloader))
            iters += 1

            ''' move data to gpu '''
            ###hier auskommentieren
            #imgs, labels = imgs.cuda(), labels.cuda()
            
             ## Train with all-real batch
            modelD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = modelD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = modelG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = modelD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()          # update parameters

                ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            modelG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = modelD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            
            
            ''' write out information to tensorboard '''
            writer.add_scalar('loss', errD.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(errD.data.cpu().numpy())

            print(train_info)

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
           
        print(epoch%args.val_epoch)
        if epoch%args.val_epoch == 0:
            #print('evaluate')
            ''' evaluate the model '''
            acc = test.evaluate(model, dataloader_target, alpha)
            writer.add_scalar('val_acc', acc, iters)
            #print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
