# urdiffusion
Your own diffusion with small, private datasets. Pronounced somewhat like your-diffusion. 

## philosophy

At the core of urdiffusion lies the idea of a diffusion model as an image generator with a limited range of expression and aesthetics, hooked up with control from init images and text prompts, as well as optional guidance from target images, image prompts and why not even style guidance using Gram matrices.

It is possible to train a limited range model with a small dataset of visually similar images in 12 to 48 hours on a home GPU. As soon as you have one pretrained model, you can retrain it with new image material overnight, sometimes even in a few hours depending on the image material.

When you start guiding such a model with init images and prompts, it is best to start with an open mind, clear of expectations. There is bound to be a tension as to what the guidance from the prompt seems to require and what the model is able to produce. This very tension is the core creative element in urdiffusion. Experiment how the model reacts to the guidance, depending on the image material used in training. The results may be far different from what you expected, but you are still likely to find interesting and useful results and styles. Train new models and experiment with different materials, find out what works for you, collect your own model library.

<img src="https://github.com/htoyryla/urdiffusion/assets/15064373/beca5e0c-e27e-4402-b8d7-fef8acf24c60" width="720px">

## requirements 

* pytorch, torchvision, cudatoolkit
* numpy
* PIL
* pytorch_mssim
* argparse
* tqdm
* einops
* diffusers
* CLIP
* kornia
* opencv (imported but not currently used)

See https://github.com/htoyryla/urdiffusion/discussions/2 for more detailed installation procedure.

## models

Get model-230.pt to start with from https://drive.google.com/drive/folders/1s4fBeZvr23ma-a6eZMQqgeICyxiku6y_?usp=sharing
and place it into folder models/

Generally, I don't share my models. You should train your own with your own materials. It should be quite easy, starting from my example model as a basis.

NOTE: The permissions of the above Google Drive link were wrong but have now been corrected, so you should be able to access the model.

## how to get started

Generate an image as follows. Use your own image path however :D

```
python urdifs.py --dir test/ --name portr2 --eta 0.5 --steps 100  --modelSize 512 --w 768  --h 768 --model unet2 --mults 1 1 2 2 4 4 8 8 --text "an abstract portrait" --textw 50 --lr 10  --load models/model-230.pt --image /home/hannu/Pictures/hft269e-adjusted.png --skip 10  --ema
```

In practice, you can ignore (leave out) settings like model and mults, unless you have experimented in training models with different settings. Unet2 and  1 1 2 2 4 4 8 8 have been found to work well in general work. Modelsize is the size the model was trained with, this too should be left to 512 (and can be omitted). 

--image can be a path to a single image or a folder of images

--dir where to place output images

--name base name for output images

--eta DDIM eta, between 0 and 1

--steps DDIM steps, lower value results in faster generation, possibly lower quality, 50 or 100 usually work well enough

--skip skip steps, as usual in image to image diffusion

--ema use averaged model, usually works better

--text prompt

--textw prompt weight


<img src="https://github.com/htoyryla/urdiffusion/assets/15064373/6fdd4d70-1827-4305-a961-7df24b7405cb" width="320px">


<img src="https://github.com/htoyryla/urdiffusion/assets/15064373/ee14b08a-7dab-4914-b40e-9056cf9e7dc3" width="320px">


You can also enable postprocessing on the generated image using --postproc, which gives access to many additional settings:

```
python urdifs.py --dir test/ --name portr --eta 0.5 --steps 100  --modelSize 512 --w 768  --h 768 --text "an abstract portrait" --textw 50 --lr 10  --load test/model-230.pt --image /home/hannu/Pictures/hft269e-adjusted.png --skip 10 --ema --postproc  --eqhist 0.5 --noise 0. --contrast 1 --gamma 0.8  --unsharp 2  --saturation 1
```

## more use cases

How to use the weak, skip and textw params to finetune the style https://github.com/htoyryla/urdiffusion/discussions/3


## training

To train a model you need a dataset of images. I typically use sets from a hundred to 10 000 images, starting training from a pretrained one and training from a few hours to a day or two on a single 3090. Training works best when the images are visually reasonably similar, i.e. we are training with visual features rather than content. 

Note: the dataset should simply be a folder containing images (ie. no subfolders).


Example

```
python urdiftrainer.py --images /work4/mat/fadedmem/ --lr 8e-5 --steps 1000 --accum 10 --dir test/ --imageSize 512 --batchSize 2 --saveEvery 100 --nsamples 2 --mults 1 1 2 2 4 4 8 8 --model unet2  --load test/model-230.pt --losstype l1
```

## Tiled generation

```
python urdifstile.py --dir test/ --name ptiles2 --eta 0.5 --steps 100  --w 1024  --h 1024 --model unet2 --mults 1 1 2 2 4 4 8 8 --text "abstract geometric planes and shapes" --textw 50 --lr 10  --load path-to-your-model --image path-to-your-image(s) --skip 10 --eqhist 0.5 --noise 0. --contrast 0.7 --gamma 1 --postproc --unsharp 2  --saturation 1.2 --ema --rsort --weak 30  --tilemin 512 --grid
```

Here we are doing img2img generation of a larger image, tile by tile, with no attempt to blend image borders, but rather enjoying the mosaic or tiled wall like result.

Here, tiles are 512x512 and the full canvas is 1024x1024.You can experiment with different tile and canvas size.

Param --weak is used here to give a somewhat softer look. 

Now that we are converting images from a folder, I have used --rsort to process them in random order.  
 

<img src="https://github.com/htoyryla/urdiffusion/assets/15064373/866c9314-d0f5-4c58-af3b-4d232e8e9d42" width="512px">

## model interpolation

If we think of urdiffusion models as something like styles, it also makes sense to think of mixing them. You can do this using model interpolation script to mix two models into a new model:

```
python interpolate_models.py --m1 squares/model-490.pt --m2 fadedmem/model-550.pt --out squares-fadedmem-08.pt --beta 0.8
```

## web app urdifapp

Install gradio

```
pip install gradio
```

Start the server from command line

```
python urdifapp.py
```


Use your browser to go to localhost:7860 or <ip-of-your-computer>:7860 if accessing from another computer in your LAN (as I do, I have bix linux boxes in a small room and work from a more convenient place from Mac)

The client is still in alpha phase, it works for a single user but it is not robust. It is better to wait until generation is finished before doing changes. On the other hand, if something goes wrong, use two or three control-C's to stop the server and restart it.

See a short introductory video https://vimeo.com/839476262?share=copy












