import torch
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import os
import clip
import argparse
import cv2
from pytorch_msssim import ssim
from postproc import pprocess

from functools import partial

import numpy as np

import random
import math

from diffusers import DDIMScheduler

'''

ur diffusion

@htoyryla June 2023

TILED GENERATION

with md2 ddim sampling with init and target image and clip conditioning 

using diffusers scheduler (not urdiffusion lib)

'''

from cutouts import cut

parser = argparse.ArgumentParser()

# define params and their types with defaults if needed
parser.add_argument('--text', type=str, default="", help='text prompt')
parser.add_argument('--image', type=str, default="", help='path to init image')
parser.add_argument('--img_prompt', type=str, default="", help='path to image prompt')
parser.add_argument('--tgt_image', type=str, default="", help='path to target image')
parser.add_argument('--lr', type=float, default=5., help='learning rate')
parser.add_argument('--ssimw', type=float, default=1., help='target image weight')
parser.add_argument('--textw', type=float, default=1., help='text weight')
parser.add_argument('--tdecay', type=float, default=1., help='text weight decay')
parser.add_argument('--imgpw', type=float, default=1., help='image prompt weight')

parser.add_argument('--trainsteps', type=int, default=1000, help='diffusion steps')

parser.add_argument('--skip', type=int, default=0, help='skip steps')
parser.add_argument('--dir', type=str, default="out", help='base directory for storing images')
parser.add_argument('--name', type=str, default="test", help='basename for storing images')
parser.add_argument('--mul', type=float, default=1., help='noise divisor when using init image')
parser.add_argument('--show', action="store_true", help='show image in a window')
parser.add_argument('--ema', action="store_true", help='use ema model')
parser.add_argument('--imageSize', type=int, default=512, help='image size')
parser.add_argument('--h', type=int, default=0, help='image height')
parser.add_argument('--w', type=int, default=0, help='image width')
parser.add_argument('--modelSize', type=int, default=512, help='native image size of the model')
parser.add_argument('--saveEvery', type=int, default=0, help='image save frequency')
parser.add_argument('--saveAfter', type=int, default=0, help='save images after step')
parser.add_argument('--low', type=float, default=0.4, help='lower limit for cut scale')
parser.add_argument('--high', type=float, default=1.0, help='higher limit for cut scale')
parser.add_argument('--cutn', type=int, default=24, help='number of cutouts for CLIP')
parser.add_argument('--load', type=str, default="", help='path to pt file')
parser.add_argument('--saveiters', action="store_true", help='')
parser.add_argument('--mults', type=int, nargs='*', default=[1, 1, 2, 2, 4, 4, 8, 8], help='')
parser.add_argument('--weak', type=int, default=0, help='weaken init image')
parser.add_argument('--model', type=str, default="unet2", help='model architecture: unet0, unet1, unet2, unetcn0')
parser.add_argument('--spher', action="store_true", help='use spherical loss')

parser.add_argument('--steps', type=int, default=50, help='sampling steps')
parser.add_argument('--eta', type=float, default=0.5, help='ddim eta')

parser.add_argument('--c', type=float, default=0.5, help='adjust im values')
parser.add_argument('--clampim', action="store_true", help='clamp img values')

parser.add_argument('--canvasSize', type=int, default=1024, help='image size')
parser.add_argument('--tilemin', type=int, default=512, help='image size')
parser.add_argument('--tilemax', type=int, default=1024, help='image size')
parser.add_argument('--tiles', type=int, default=64, help='image size')
parser.add_argument('--grid', action="store_true", help='')

parser.add_argument('--postproc', action="store_true", help='use post processing')
parser.add_argument('--contrast', type=float, default=1, help='contrast, 1 for neutral')
parser.add_argument('--saturation', type=float, default=1, help='saturation, 1 for neutral')
parser.add_argument('--gamma', type=float, default=1, help='gamma, 1 for neutral')
parser.add_argument('--unsharp', type=float, default=0, help='unsharp mask')
parser.add_argument('--eqhist', type=float, default=0., help='histogram eq level')
parser.add_argument('--median', type=int, default=0, help='median blur kernel size, 0 for none')
parser.add_argument('--c1', type=float, default=0., help='do not use')
parser.add_argument('--c2', type=float, default=1., help='do not use')
parser.add_argument('--sharpenlast', action="store_true", help='do not use')
parser.add_argument('--sharpkernel', type=int, default=3, help='sharpening kernel')
parser.add_argument('--ovl0', type=float, default=0, help='blend original with blurred image')
parser.add_argument('--bil', type=int, default=0, help='bilateral filter kernel')
parser.add_argument('--bils1', type=int, default=75, help='bilateral filter sigma for color')
parser.add_argument('--bils2', type=int, default=75, help='bilateral filter sigma for space')
parser.add_argument('--noise', type=float, default=0., help='add noise')

parser.add_argument('--latest', action="store_true", help='save latest image for display')
parser.add_argument('--rsort', action="store_true", help='sort input files randomly')

opt = parser.parse_args()

mtype = opt.model

if opt.h == 0:
    opt.h = opt.imageSize

if opt.w == 0:
    opt.w = opt.imageSize
    

if mtype == "unet0":
  from alt_models.Unet0 import Unet
elif mtype == "unet0k5":
  from alt_models.Unet0k5 import Unet
elif mtype == "unet1":
  from alt_models.Unet1 import Unet
elif mtype == "unet2":
  from alt_models.Unet2 import Unet    
elif mtype == "unetcn0":
  from alt_models.UnetCN0 import Unet
else:
  print("Unsupported model: "+mtype)
  exit()


name = opt.name #"out5/testcd"
steps = opt.steps
bs = 1
ifn = opt.image 

model = Unet(
    dim = 64,
    dim_mults = opt.mults # (1, 2, 4, 8)
).cuda()

def load_model(fn):
  data = torch.load(fn)

  try:
    print("loaded "+fn+", correct mults: "+",".join(str(x) for x in data['mults']))
  except:
    print("loaded "+fn+", no mults stored")

  m = "ema" if opt.ema else "model"
  dd = data[m].copy()
  
  # if using DDIM remove original scheduler steps
  
  if opt.steps < dd['betas'].shape[0]:
    sched_keys = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod', 'posterior_variance', 'posterior_log_variance_clipped', 'posterior_mean_coef1', 'posterior_mean_coef2']
    for k in sched_keys:
       del dd[k]

  return dd
  
dd = load_model(opt.load)
    
dd_ = {}
for k in dd.keys():
    v = dd[k]
    k_ = k.replace("denoise_fn.","")
    dd_[k_] = v  

model.load_state_dict(dd_, strict=False)

if opt.textw > 0:
    perceptor, clip_preprocess = clip.load('ViT-B/32', jit=False)
    perceptor = perceptor.eval()
    cnorm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

text = opt.text 

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

transform = transforms.Compose([transforms.Resize((opt.h, opt.w)), transforms.ToTensor()])

if opt.tgt_image != "":   
  if opt.tgt_image == "init":
    imS = imT_.clone()
  else:
    imS = transform(Image.open(opt.tgt_image).convert('RGB')).float().cuda().unsqueeze(0)
    imS = (imS * 2) - 1

if opt.img_prompt != "":   
    imP = transform(Image.open(opt.img_prompt).convert('RGB')).float().cuda().unsqueeze(0)
    nimg = imP.clip(0,1)
    nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
    imgp_enc = perceptor.encode_image(nimg.detach()).detach()

if opt.text != "" and opt.textw > 0:
    tx = clip.tokenize(text)                        # convert text to a list of tokens 
    txt_enc = perceptor.encode_text(tx.cuda()).detach()   # get sentence embedding for the tokens
    del tx
    
def tilexy():
    th = (random.randint(opt.tilemin, opt.tilemax)//64)*64
    tw = (random.randint(opt.tilemin, opt.tilemax)//64)*64
    ty = random.randint(0, opt.h - th)
    tx = random.randint(0, opt.w - tw)
    #print(ty, tx, th, tw)
    return (ty, tx, th, tw)


    
def tileList():
    tlist = []
    if opt.grid:
        size = opt.tilemin
        nx = opt.w // size
        ny = opt.h // size
        tx = 0
        print(nx, ny)
        for ix in range(0, nx):
            ty = 0
            for iy in range(0, ny):
              tlist.append((ty, tx, size, size))
              ty += size
            tx += size
    else:
        for i in range(0, opt.tiles):
            tlist.append(tilexy())                          
    random.shuffle(tlist)
    return tlist

def getTile(field, pos):
    ty, tx, th, tw = pos
    tile = field[:, :, ty:ty+th, tx:tx+th]
    return tile
    
def putTile(field, pos, content):
    ty, tx, th, tw = pos
    field[:, :, ty:ty+th, tx:tx+th] = content
    return field    



@torch.enable_grad()
def cond_fn(x, t, x_s):
    global opt    
    x_is_NaN = False
    x.grad = None
    x.requires_grad_()
    n = x.shape[0]         
    
    x_s.requires_grad_()
    x_grad = torch.zeros_like(x_s)
                    
    loss = 0
    losses = []

    nimg = None

    if opt.text != "" and opt.textw > 0:
        nimg = x_s.clip(-1, 1) + 0.5    
        nimg = cut(nimg, cutn=opt.cutn, low=opt.low, high=opt.high, norm = cnorm)
        
        # get image encoding from CLIP
 
        img_enc = perceptor.encode_image(nimg) 
  
        # we already have text embedding for the promt in txt_enc
        # so we can evaluate similarity
     
        if opt.spher:
            loss = opt.textw * spherical_dist_loss(txt_enc.detach(), img_enc).mean()
        loss = opt.textw*10*(1-torch.cosine_similarity(txt_enc.detach(), img_enc)).view(-1, bs).T.mean(1)
        losses.append(("Text loss",loss.item())) 
        if opt.tdecay < 1.:
            opt.textw = opt.tdecay * opt.textw
        x_grad += torch.autograd.grad(loss.sum(), x_s, retain_graph = True)[0]

        del nimg

    if opt.img_prompt != "":
        if nimg == None:
            nimg = x_s.clip(-1, 1) + 0.5     
            nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
            img_enc = perceptor.encode_image(nimg)
            del nimg
        loss1 = opt.imgpw*10*(1-torch.cosine_similarity(imgp_enc, img_enc)).view(-1, bs).T.mean(1)  
        losses.append(("Img prompt loss",loss1.item())) 
        loss = loss + loss1     
        
        x_grad += torch.autograd.grad(loss1.sum(), x_s, retain_graph = True)[0]
        
    if opt.tgt_image != "":
          loss_ = opt.ssimw * (1 - ssim((x_s+1)/2, (imS+1)/2)).mean() 
          losses.append(("Ssim loss",loss_.item())) 
          loss = loss + loss_    
          
          x_grad += torch.autograd.grad(loss_.sum(), x_s, retain_graph = True)[0]
    
    if torch.isnan(x_grad).any()==False:
        grad = -torch.autograd.grad(x_s, x, x_grad)[0]
    else:
      x_is_NaN = True
      grad = torch.zeros_like(x)             
          
    del x, x_s, x_grad, loss
          
    return opt.lr*grad.detach()


# important! will not work with diffusers default betas 

def make_betas(timesteps):
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    
    return betas.numpy()
    

scheduler = DDIMScheduler(num_train_timesteps=opt.trainsteps, prediction_type = "epsilon", trained_betas = make_betas(opt.trainsteps), clip_sample=False)
scheduler.set_timesteps(opt.steps, device="cuda")


def get_timesteps(skip = opt.skip):
    offset = scheduler.config.get("steps_offset", 0)
    
    # get the original timestep using init_timestep
    init_timestep = opt.skip + offset
    init_timestep = min(skip, opt.steps)
    
    timesteps = scheduler.timesteps[init_timestep:]
    t_start = max(opt.steps - init_timestep, 0)

    return {'timesteps':timesteps, 't_start': t_start}
       
def getx(ifn=None):
    global timesteps
    init_noise = torch.zeros(bs,3,opt.h,opt.w).normal_(0,1) #.cuda()
    im = None
    if ifn != None:   
        x = transform(Image.open(ifn).convert('RGB')).float().unsqueeze(0)
        im = x.clone()
        x -= 0.5 #(imT_ * 2) - 1
        x = opt.mul*scheduler.add_noise(x, init_noise, timesteps[opt.weak])
        print(x.std())
    else:
        x = opt.mul*init_noise * scheduler.init_noise_sigma

    return x, im

if os.path.isdir(opt.image):
    imgList = os.listdir(opt.image)
    inputlist = []
    for fname in imgList:
      # skip non-images
      ext = fname.split('.')[-1].lower()
      imgname = fname.split('.')[0].lower()
      if not ext in ['jpg', 'jpeg', 'png', 'tiff', 'tif']:
        continue
        
      fpath = opt.image+"/"+ fname
      inputlist.append(fpath)
    
    if opt.rsort:
        random.shuffle(inputlist)
    else:    
        inputlist.sort() # todo proper numeric sort
    
elif opt.image == "":
    inputlist = [None]

else:     
    inputlist = [opt.image]
    
timesteps = get_timesteps(opt.skip)['timesteps']    

ctr = 0
for inp in inputlist:
  print(inp)    
  xf, im = getx(inp)
  #x = x.cuda()
  #im = im.cuda()
  
  # prepare for tiling
  
  imTf = im.clone()
  #save_image((imTf.clone()+0.5, opt.dir+"/"+name+"-init.png")
  
  tilelist = tileList()
  print(tilelist)
  
  imTf_ = {}
  #for k in opt.saveIters:
  #    imTf_[str(k)] = imTf.clone()
      
  for tn in range(0, len(tilelist)):
    j = 0
    tile = tilelist[tn] #tilexy(opt.h, opt.w)
    x = getTile(xf, tile).cuda()
    
    print(ifn, tn)

    #if opt.tgt_image != "":   
    # imS = getTile(imSf.clone(), tile).cuda() 

    #if opt.image != "":
    #  my_t = torch.ones([bs], device='cuda', dtype=torch.long).cuda() * indices[0]
    #  imT = diffusion.q_sample(imT, my_t, init_noise)    

    for t in tqdm(timesteps):
      my_t = torch.tensor([t] * bs, device='cuda').cuda().detach()
      
      if (opt.text!="" and opt.textw > 0):
         with torch.enable_grad():
             with torch.autocast(device_type='cuda', dtype=torch.float16):
                 x.requires_grad_() 
                 noise = model(x, my_t).cuda() 
                 
                 alpha_prod_t = scheduler.alphas_cumprod[t]
                 beta_prod_t = 1 - alpha_prod_t
                 pred_original_sample = (x - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
                 fac = torch.sqrt(beta_prod_t)
                 sample = pred_original_sample * (fac) + x * (1 - fac)
                 
                 grad = cond_fn(x, t, sample)  
                 noise = noise - torch.sqrt(beta_prod_t) * grad               
                 x = scheduler.step(noise, t, x, eta=opt.eta)['prev_sample'].detach() 
                 
                 del sample, grad, alpha_prod_t, beta_prod_t
      else:              
          with torch.no_grad():
              with torch.autocast(device_type='cuda', dtype=torch.float16): 
                  noise = model(x, my_t).cuda() #.sample
                  x = scheduler.step(noise, t, x, eta=opt.eta)['prev_sample'].detach()                  
      
      del noise 
    

    im = (x.clone()+opt.c)
  
    if opt.clampim:
      im = im.clamp(0,1)
 
    #if opt.postproc:
    #  im = pprocess(im.cpu(), opt) 
    #  im -= im.min()
    #  im /= im.max()
    #  #  #save_image((im), opt.dir+"/"+name+"-"+str(ctr)+"-"+str(tn)"-finalp.png")    
  
    #else:
    #save_image(im, opt.dir+"/"+name+"-"+str(ctr)+"-"+str(tn)"-final.png")
    #pass
      
    imTf = putTile(imTf, tile, im.cpu().detach()) 
    #save_image(imTf, opt.dir+"/"+name+"-"+str(ctr)+"finalp.png")      

    im_ = imTf.clone().cpu()
    if opt.postproc:
      im_ = pprocess(im_, opt) 
      im_ -= im_.min()
      im_ /= im_.max()

    save_image(im_, opt.dir+"/"+name+"-"+str(ctr)+"finalp.png")  

    if opt.latest:
      save_image(im_, "/var/www/html/latest.jpg")
        
  ctr += 1
