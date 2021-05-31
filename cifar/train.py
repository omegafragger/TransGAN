from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

from utils import *
from models import *
from fid_score import *
from inception_score import *


parser = argparse.ArgumentParser()
#parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
#parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
#parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
#parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
#parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
#parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
#parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
#parser.add_argument('--early_stopping',type=int, default=10, help='Tolerance for early stopping (# of epochs).')
#parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')
#parser.add_argument('--start_test', type=int, default=80, help='define from which epoch test')
#parser.add_argument('--train_jump', type=int, default=0, help='define whether train jump, defaul train_jump=0')
#parser.add_argument('--dataset', type=str, default="cora", help='define your dataset eg. "cora" ')
#parser.add_argument('--train_percentage', type=float, default=0.1 , help='define the percentage of training data.')
#parser.add_argument('--attack_dimension', type=int, default=0, help='define how many dimension of the node feature to attack')


parser.add_argument('--image_size', type=int, default= 32 , help='Size of image for discriminator input.')
parser.add_argument('--initial_size', type=int, default=8 , help='Initial size for generator.')
parser.add_argument('--patch_size', type=int, default=16 , help='Patch size for generated image.')
parser.add_argument('--num_classes', type=int, default=1 , help='Number of classes for discriminator.')
parser.add_argument('--lr_gen', type=int, default=0.0002 , help='Learning rate for generator.')
parser.add_argument('--lr_dis', type=int, default=0.0002 , help='Learning rate for discriminator.')
parser.add_argument('--weight_decay', type=int, default=3e-5 , help='Weight decay.')
parser.add_argument('--latent_dim', type=int, default=384 , help='Latent dimension.')
parser.add_argument('--n_critic', type=int, default=5 , help='n_critic.')
parser.add_argument('--gener_batch_size', type=int, default=16 , help='Batch size for generator.')
parser.add_argument('--dis_batch_size', type=int, default=16 , help='Batch size for discriminator.')
parser.add_argument('--epoch', type=int, default=200 , help='Number of epoch.')
parser.add_argument('--output_dir', type=str, default='checkpoint' , help='Checkpoint.')
parser.add_argument('--dim', type=int, default=384 , help='Embedding dimension.')
parser.add_argument('--img_name', type=str, default="img_name" , help='Name of pictures file.')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:",device)

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
"""if args.cuda:
torch.cuda.manual_seed(args.seed)"""


#args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()

#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#if args.cuda:
#    torch.cuda.manual_seed(args.seed)


# Load data
#add_all, adj, features, labels, idx_train, idx_val, idx_test = load_data(args.batch_size, args.img_size)

generator= Generator(depth1=5, depth2=2, depth3=2, initial_size=8, dim=384, heads=8, mlp_ratio=4, drop_rate=0.5)#,device = device)
generator.to(device)

discriminator = Discriminator(image_size=32, patch_size=16, input_channel=3, num_classes=1,
                 dim=384, depth=7, heads=8, mlp_ratio=4,
                 drop_rate=0.5)
discriminator.to(device)


generator.apply(inits_weight)
discriminator.apply(inits_weight)

optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                lr=args.lr_gen, weight_decay=args.weight_decay)

optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                lr=args.lr_dis, weight_decay=args.weight_decay)

fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'

writer=SummaryWriter()
writer_dict = {'writer':writer}
writer_dict["train_global_steps"]=0
writer_dict["valid_global_steps"]=0


def train(noise,generator, discriminator, optim_gen, optim_dis,
        epoch, writer,img_size=32, latent_dim = args.latent_dim,
        n_critic = args.n_critic,
        gener_batch_size=args.gener_batch_size,device="cuda:0"):


    writer = writer_dict['writer']
    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()

    transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=30, shuffle=True)

    for index, (img, _) in enumerate(train_loader):

        global_steps = writer_dict['train_global_steps']

        real_imgs = img.type(torch.cuda.FloatTensor)

        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], latent_dim)))#noise(img, latent_dim)#= args.latent_dim)

        optim_dis.zero_grad()
        real_valid=discriminator(real_imgs)
        fake_imgs = generator(noise).detach()
        
        #assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_valid = discriminator(fake_imgs)

        loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)

        if global_steps % n_critic == 0:

            optim_gen.zero_grad()

            gener_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

            generated_imgs= generator(gener_noise)
            fake_valid = discriminator(generated_imgs)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
            writer.add_scalar("gener_loss", gener_loss.item(), global_steps)

            gen_step += 1

            #writer_dict['train_global_steps'] = global_steps + 1

        if gen_step and index % 100 == 0:
            sample_imgs = generated_imgs[:25]
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(sample_imgs, f'generated_images/generated_img_{epoch}_{index % len(train_loader)}.jpg', nrow=5, normalize=True, scale_each=True)            
            tqdm.write("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch+1, index % len(train_loader), len(train_loader), loss_dis.item(), gener_loss.item()))

        #writer_dict['train_global_steps'] = global_steps + 1



def validate(generator, writer_dict, fid_stat):


        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']

        generator = generator.eval()
        fid_score = get_fid(fid_stat, epoch, generator, num_img=5000, val_batch_size=60*2, latent_dim=384, writer_dict=None, cls_idx=None)


        print(f"FID score: {fid_score}")

        writer.add_scalar('FID_score', fid_score, global_steps)

        writer_dict['valid_global_steps'] = global_steps + 1
        return fid_score



best = 1e4


for epoch in range(args.epoch):

    train(noise, generator, discriminator, optim_gen, optim_dis,
    epoch, writer,img_size=32, latent_dim = args.latent_dim,
    n_critic = args.n_critic,
    gener_batch_size=args.gener_batch_size)

    checkpoint = {'epoch':epoch, 'best_fid':best}
    checkpoint['generator_state_dict'] = generator.state_dict()
    checkpoint['discriminator_state_dict'] = discriminator.state_dict()

    score = validate(generator, writer_dict, fid_stat)

    print(f'FID score: {score} - best ID score: {best} || @ epoch {epoch+1}.')
    if epoch == 0 or epoch > 30:
        if score < best:
            save_checkpoint(checkpoint, is_best=(score<best), output_dir=args.output_dir)
            print("Saved Latest Model!")
            best = score


checkpoint = {'epoch':epoch, 'best_fid':best}
checkpoint['generator_state_dict'] = generator.state_dict()
checkpoint['discriminator_state_dict'] = discriminator.state_dict()
score = validate(generator, writer_dict, fid_stat) ####CHECK AGAIN
save_checkpoint(checkpoint,is_best=(score<best), output_dir=args.output_dir)
