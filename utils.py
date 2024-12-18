import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
import math

def adjust_lr(optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 200):
    
    lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.8)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

"""

To get R random different initializations of z from L steps of Gradient Descent.
rec_iter : the number of L of Gradient Descent steps 
tec_rr : the number of different random initialization of z

"""

def get_z_sets(model, data, lr, device, rec_iter = 200, rec_rr = 10, input_latent = 64, global_step = 1):
    
    display_steps = 100
    
    rec_loss = torch.Tensor(rec_rr*data.size(0))
    # the output of R random different initializations of z from L steps of GD
    z_hats_recs = torch.Tensor(rec_rr*data.size(0), input_latent)
    
    # the R random differernt initializations of z before L steps of GD
    z_hats_orig = torch.Tensor(rec_rr*data.size(0), input_latent)
    
    z_hat = torch.normal(0,np.sqrt(1.0 / input_latent),size = (rec_rr*data.size(0), input_latent)).to(device)
    z_hat = z_hat.detach().requires_grad_()
    
    z_hats_orig = z_hat.cpu().detach().clone()
    
    data_rr = data.view(data.size(0),data.size(1)*data.size(2)*data.size(3))#50*784
    data_rr = data_rr.repeat(1,rec_rr)#50,784*R
    data_rr = data_rr.view(data.size(0)*rec_rr, data.size(1), data.size(2), data.size(3)).to(device)#50*R,1,28,28
    
    cur_lr = lr
    
    optimizer = optim.SGD([z_hat], lr = cur_lr, momentum = 0.7)
    
    for iteration in range(rec_iter):
            
        optimizer.zero_grad()
            
        fake_image = model(z_hat)
            
        fake_image = fake_image.view(-1, data_rr.size(1), data_rr.size(2), data_rr.size(3))
            
#         reconstruct_loss = loss(fake_image,data_rr)
        reconstruct_loss = torch.square(fake_image-data_rr).mean(dim=[1,2,3])
        
        rec_loss = reconstruct_loss.cpu().detach().clone()
        
        reconstruct_loss.backward(reconstruct_loss.detach().clone())
            
        optimizer.step()
            
        cur_lr = adjust_lr(optimizer, cur_lr, global_step = global_step, rec_iter= rec_iter)
            
    z_hats_recs = z_hat.cpu().detach().clone()
    
#     for idx in range(len(z_hats_recs)):
        
#         cur_lr = lr

#         optimizer = optim.SGD([z_hat], lr = cur_lr, momentum = 0.7)
        
#         z_hats_orig[idx] = z_hat.cpu().detach().clone()
        
#         for iteration in range(rec_iter):
            
#             optimizer.zero_grad()
            
#             fake_image = model(z_hat)
            
#             fake_image = fake_image.view(-1, data.size(1), data.size(2), data.size(3))
            
#             reconstruct_loss = loss(fake_image, data)
             
#             reconstruct_loss.backward()
            
#             optimizer.step()
            
#             cur_lr = adjust_lr(optimizer, cur_lr, global_step = global_step, rec_iter= rec_iter)
           
#         z_hats_recs[idx] = z_hat.cpu().detach().clone()
        
    return z_hats_orig, z_hats_recs , rec_loss

"""

To get z* so as to minimize reconstruction error between generator G and an image x

"""

def get_z_star(model, data, z_hats_recs, rec_loss, device, rec_rr):
    
    reconstructions = torch.Tensor(data.size(0))
    z_best = torch.Tensor(data.size(0),z_hats_recs.size(-1))#bacth_size*input_latent
    
    for i in range(data.size(0)):
        ind = i * rec_rr + torch.argmin(rec_loss[i*rec_rr:(i+1)*rec_rr].to(device))
        
#         z = model(z_hats_recs[i*rec_rr:i*rec_rr+rec_rr].to(device))#Z:(R,1,28,28)
        
#         z = z.view(-1, data.size(1), data.size(2), data.size(3))#Z:(R,1,28,28)
        
#         data_rr = data[i].view(data.size(1)*data.size(2)*data.size(3))#DATA_RR:(784)
#         data_rr = data_rr.repeat(1,rec_rr)#DATA_RR:(R*784)
#         data_rr = data_rr.view(-1, data.size(1), data.size(2), data.size(3))#(R,1,28,28)
        
#         reconstructions[i] = loss(z, data_rr).cpu().item()#计算MSELOSS
        
#         best = torch.argmin(reconstructions[i])#选出loss最小的序号
        
        z_best[i] = z_hats_recs[ind].cpu().detach().clone()# 选择最优的加入z_best
    
    return z_best


def Resize_Image(target_shape, images):
    
    batch_size, channel, width, height = target_shape
    
    Resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((width,height)),
        transforms.ToTensor(),
    ])
    
    result = torch.zeros((batch_size, channel, width, height), dtype=torch.float)
    
    for idx in range(len(result)):
        result[idx] = Resize(images.data[idx])

    return result

