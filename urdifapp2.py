from diffusers import DDIMScheduler

import torch
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import os
import clip
import argparse
import cv2

from postproc import pprocess
from functools import partial
import numpy as np
import random

from alt_models.Unet2 import Unet 

import types
import gradio as gr
from cutouts import cut
import math


# set up initial values

opt = types.SimpleNamespace()

opt.load = "md2un2-msgbox-624.pt"

opt.text = ""
opt.image = ""
opt.img_prompt = ""
opt.tgt_image = ""
opt.lr = 10.
opt.ssimw = 0
opt.textw = 0
opt.imgpw = 0
opt.tdecay = 1

opt.mul = 1

opt.trainsteps = 1000
opt.modelSize = 512

opt.skip = 0
opt.ema = True
opt.imageSize = 768
opt.h = 0
opt.w = 0
opt.mults = [1, 1, 2, 2, 4, 4, 8, 8]
opt.spher = False

opt.steps = 100
opt.eta = 0.5

opt.cutn = 32
opt.low = 0.4
opt.high = 0.97

opt.weak = 0

opt.postproc = True
opt.onorm = True
opt.contrast = 0.8
opt.saturation = 1.
opt.gamma = 1.
opt.eqhist = 0.5
opt.unsharp = 1.
opt.c1 = 0.
opt.c2 = 1.
opt.sharpenlast = True
opt.sharpkernel = 3
opt.median = 0
opt.ovl0 = 0
opt.noise = 0.02
opt.bil = 0.
opt.bils1 = 75
opt.bils2 = 75


if opt.h == 0:
    opt.h = opt.imageSize

if opt.w == 0:
    opt.w = opt.imageSize
    
steps = opt.steps

# initialize model and diffusion

model = Unet(
    dim = 64,
    dim_mults = opt.mults # (1, 2, 4, 8)
).cuda()

# set up clip model for textual guidance


perceptor, clip_preprocess = clip.load('ViT-B/32', jit=False)
perceptor = perceptor.eval()
cnorm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
  
# get a list of available models

dc = os.listdir("./models/")
ml = []
for d in dc:
        mn = d.split("/")[-1]
        ml.append(mn)
opt.modellist = ml    

if opt.load in opt.modellist:
    fn = "./models/"+opt.load
else:
    opt.load = ml[0]
    fn = "./models/"+opt.load
        
# function to load model weights  
  
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
  
# load default model  
  
dd = load_model(fn)
    
dd_ = {}
for k in dd.keys():
    v = dd[k]
    k_ = k.replace("denoise_fn.","")
    dd_[k_] = v  

model.load_state_dict(dd_, strict=False)

current_modelname = fn.split(os.sep)[-1]
opt.modelname = current_modelname

bs = 1

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

transform = transforms.Compose([transforms.Resize((opt.h, opt.w)), transforms.ToTensor()]) 

tensor_to_pil = transforms.ToPILImage()

    
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

# create scheduler    

scheduler = DDIMScheduler(num_train_timesteps=opt.trainsteps, prediction_type = "epsilon", trained_betas = make_betas(opt.trainsteps), clip_sample=False)
scheduler.set_timesteps(opt.steps, device="cuda")

scheduler.alphas_cumprod = scheduler.alphas_cumprod.to("cuda")

timesteps = None

# function to set timesteps

def get_timesteps(skip = opt.skip):
    offset = scheduler.config.get("steps_offset", 0)
    
    # get the original timestep using init_timestep
    init_timestep = opt.skip + offset
    init_timestep = min(skip, opt.steps)
    
    timesteps = scheduler.timesteps[init_timestep:]
    t_start = max(opt.steps - init_timestep, 0)

    return {'timesteps':timesteps, 't_start': t_start}
       


# diffusion run to be run by each user instance

def diffusion_run(modelsel, im, o, progress):

    # only global objects are the currently active model and its name    
    
    global current_modelname, model 
    

    # load model weights unless same model is already loaded
    
    o.modelname = modelsel
    
    print(current_modelname, o.modelname)        
    
    if o.modelname != current_modelname:
        fn = "models"+os.sep + opt.modelname
        print("loading "+fn)
        dd = load_model(fn)
    
        dd_ = {}
        for k in dd.keys():
            v = dd[k]
            k_ = k.replace("denoise_fn.","")
            dd_[k_] = v  

        model.load_state_dict(dd_, strict=False)
        current_modelname = fn.split(os.sep)[-1]
    
    # cond function for guidance by text prompt and images

    @torch.enable_grad()
    def cond_fn(x, t, x_s):
        x_is_NaN = False
        x.grad = None
        x.requires_grad_()
        n = x.shape[0]         
    
        x_s.requires_grad_()
        x_grad = torch.zeros_like(x_s)
                    
        loss = 0
        losses = []

        nimg = None

        if o.text != "" and o.textw > 0:
            nimg = x_s.clip(-1, 1) + 0.5    
            nimg = cut(nimg, cutn=o.cutn, low=opt.low, high=o.high, norm = cnorm)
        
            # get image encoding from CLIP
 
            img_enc = perceptor.encode_image(nimg) 
  
            # we already have text embedding for the promt in txt_enc
            # so we can evaluate similarity
     
            if o.spher:
                loss = o.textw * spherical_dist_loss(txt_enc.detach(), img_enc).mean()
            loss = o.textw*10*(1-torch.cosine_similarity(txt_enc.detach(), img_enc)).view(-1, bs).T.mean(1)
            losses.append(("Text loss",loss.item())) 
            if o.tdecay < 1.:
                o.textw = o.tdecay * o.textw
            x_grad += torch.autograd.grad(loss.sum(), x_s, retain_graph = True)[0]

            del nimg

        '''''
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
        ''' 
        
        if torch.isnan(x_grad).any()==False:
            grad = -torch.autograd.grad(x_s, x, x_grad)[0]
        else:
          x_is_NaN = True
          grad = torch.zeros_like(x)             
          
        del x, x_s, x_grad, loss
          
        return o.lr * grad.detach()
        
        
    # get text encoding for the prompt    

    if o.text != "" and o.textw > 0:
        tx = clip.tokenize(o.text)                        # convert text to a list of tokens 
        txt_enc = perceptor.encode_text(tx.cuda()).detach()   # get sentence embedding for the tokens
        del tx
    else:
        txt_enc = None    
    

    #if opt.tgt_image != "":   
    #  if opt.tgt_image == "init":
    #    imS = imT_.clone()
    #  else:
    #    imS = transform(Image.open(opt.tgt_image).convert('RGB')).float().cuda().unsqueeze(0)
    #    imS = (imS * 2) - 1

    #if opt.img_prompt != "":   
    #    imP = transform(Image.open(opt.img_prompt).convert('RGB')).float().cuda().unsqueeze(0)
    #    nimg = imP.clip(0,1)
    #    nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
    #    imgp_enc = perceptor.encode_image(nimg.detach()).detach()

        
    #indices = list(range(opt.steps - opt.skip))[::-1] 

    #x = init_noise.cuda()
    

    
    # function to initialize diffusion with noise & init image (when applicable)
        
    def getx(im=None):
        init_noise = torch.zeros(bs,3,opt.h,opt.w).normal_(0,1).cuda()
        
        # if init image, mix it with noise
        
        if im is not None:   
            im = Image.fromarray(im)
            x = transform(im).cuda().unsqueeze(0)
            x -= 0.5 #(imT_ * 2) - 1
            x = o.mul*scheduler.add_noise(x, init_noise, timesteps[o.weak])
        
        # otherwise pure noise adjusted to proper level
        
        else:
            x = o.mul*init_noise * scheduler.init_noise_sigma

        return x   

    # main diffusion loop starts here
    
    timesteps = get_timesteps(o.skip)['timesteps'] 

    progress(0)

    ctr = 0
    
    # get initial noise (mixed with init image if present)
    
    x = getx(im).cuda()      
    
    # iterate through timesteps
    
    #for i in tqdm(timesteps):
    for i in timesteps:
    
        with torch.no_grad():
            t = torch.tensor([i] * bs, device='cuda').detach()
            
            if (o.text!="" and o.textw > 0):
               with torch.enable_grad():
                   with torch.autocast(device_type='cuda', dtype=torch.float16):
                       x.requires_grad_() 
                       t = t.to(device=x.device)
                       noise = model(x, t).to(device=x.device, dtype=x.dtype)
                       
                       alpha_prod_t = scheduler.alphas_cumprod[t.cpu()].to(device=x.device, dtype=x.dtype)
                       beta_prod_t = 1 - alpha_prod_t
                       
                       pred_original_sample = (x - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
                       fac = torch.sqrt(beta_prod_t) #.cuda()
                       sample = pred_original_sample * (fac) + x * (1 - fac)
                 
                       grad = cond_fn(x, t, sample).to(device=x.device, dtype=x.dtype)
                       noise = noise - torch.sqrt(beta_prod_t) * grad
                                    
                       s = scheduler.step(noise, t, x, eta=opt.eta) #
                       x = s['prev_sample'].cuda().detach() 
                       x_s = s['pred_original_sample'].detach() 
                       del sample, grad, alpha_prod_t, beta_prod_t
            else:              
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16): 
                        noise = model(x, t).cuda() #.sample
                        s = scheduler.step(noise, t, x, eta=opt.eta) #
                        x = s['prev_sample'].detach() 
                        x_s = s['pred_original_sample'].detach()                  
      
            del noise
            
            # adjust output into raw (unprocessed) image and a post processed one
            
            im = (x_s.clone()+0.5)
  
            if o.onorm:
                im -= im.min()
                im /= im.max()
            else:    
              im = im.clamp(0,1)
              
            imout_raw = tensor_to_pil(im[0].clone())  
           
            im = pprocess(im*2 - 1, opt)
            im -= im.min()
            im /= im.max()
            
            imout = tensor_to_pil(im[0])
            
            ctr += 1
            
            progress(float(i) / (o.steps - o.skip))
        
            yield imout, imout_raw, str(ctr)+"/"+str(o.steps - o.skip)
            
# gradio UI           
      
with gr.Blocks() as demo:
    
    imo = gr.State(Image.new('RGB', (768, 768)))
    opt_ = gr.State(types.SimpleNamespace())
    opt_ = opt
    opt_.modelname = current_modelname
    
    with gr.Row():
        with gr.Column(min_width = 80):
            html = '<div style="font-size: 36px; color:#666666;">#urdiffusion</div><br>'+  \
            '<div style="font-size: 12px; color:#666666;">experimental single user version</div>' + \
            '<div style="font-size: 20px; color:#666666;">by @htoyryla 2023</div>'
            logo = gr.HTML(html)
        with gr.Column():
            text_input = gr.Textbox(label="Prompt")
            textw = gr.Slider(minimum=0., maximum=100., value=10., label="Text weight")
        with gr.Column():
            skip = gr.Slider(minimum=0, maximum=100, value=10, label="Skip")
            mul = gr.Slider(minimum=0., maximum=1., value=1., label="Noise level")
            weak = gr.Slider(minimum=0., maximum=1., value=0., label="Softness")
        init_image = gr.Image()
        with gr.Column():
            modelsel = gr.Dropdown(choices = opt.modellist, value=opt.load, label="Select model")        
            text_button = gr.Button("Go")
            process_status = gr.Textbox(label="Generation status")
        
    with gr.Row():    
        image_output_raw = gr.Image(label="generated")
        image_output = gr.Image(label="postprocessed")
    
    with gr.Row():    
        with gr.Column():
            contrast = gr.Slider(minimum=0.5, maximum=2., value=1., label="Contrast")
            gamma = gr.Slider(minimum=0.5, maximum=2., value=1., label="Gamma")
            saturation = gr.Slider(minimum=0., maximum=2., value=1., label="Saturation")
        with gr.Column():
            eqhist = gr.Slider(minimum=0., maximum=4., value=0.5, label="Eq hist")
            unsharp = gr.Slider(minimum=0., maximum=4., value=1., label="Sharpen")
            noise = gr.Slider(minimum=0., maximum=0.1, value=0.02, label="Noise")
        with gr.Column():
            bil = gr.Slider(minimum=0., maximum=100, value=0, step=1, label="Bilateral blur")
            bils1 = gr.Slider(minimum=3, maximum=177, value=75, step=2, label="Sigma space")
            bils2 = gr.Slider(minimum=3, maximum=177, value=75, step=2, label="Sigma color")
        with gr.Column():
            proc_button = gr.Button("Change")
            post_process_status = gr.Textbox(label="Postprocess status")
    
    # function to apply changed post processing settings to current image
   
    def pproc(contrast, gamma, saturation, eqhist, unsharp, noise, bil, bils1, bils2):
        global opt_, imo
        opt_.contrast = float(contrast)
        opt_.gamma = float(gamma)
        opt_.saturation = float(saturation)
        opt_.eqhist = float(eqhist)
        opt_.unsharp = float(unsharp)
        opt_.noise = float(noise)
        
        if bil > 0:
            bil = int(bil*2) + 1
        opt_.bil = int(bil)
        opt_.bils1 = int(bils1)    
        opt_.bils2 = int(bils2)    
        
        post_process_status.value = "Postprocessing..."
        imT = TF.to_tensor(imo).unsqueeze(0)*2 - 1
        imT = pprocess(imT, opt)
        imT = imT - imT.min()
        imT = imT / imT.max()
        imout = TF.to_pil_image(imT[0])
        post_process_status.value = "Done"
        return imout, "Done"
        
    # function to run diffusion based on current settings    
    
    def run(ms, t, s, m, im, skip, weak):
        global opt_, imo
        
        #print("starting run: ",opt_)
        
        opt_.textw = float(s)
        opt_.mul = float(m)
        opt_.text = t
        opt_.skip = int(skip)
        opt_.weak = int(weak*(opt_.steps - opt_.skip - 1))
        
        process_status.value = "Processing..."
        for im, imr, i in diffusion_run(ms, im, opt_, progress=gr.Progress()):
            #print(i)
            imo = imr
            yield im, imr, str(i)     
            
    # buttons to trigger diffusion and post processing        
             
    text_button.click(queue=True, fn=run, inputs=[modelsel, text_input, textw, mul, init_image, skip, weak], outputs=[image_output, image_output_raw, process_status])
    
    proc_button.click(queue=True, fn=pproc, inputs=[contrast, gamma, saturation, eqhist, unsharp, noise, bil, bils1, bils2], outputs=[image_output, post_process_status])

demo.queue(concurrency_count=1)
demo.launch(server_name = "0.0.0.0")
