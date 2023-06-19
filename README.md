# urdiffusion
Your own diffusion with small, private datasets 

## how to get started

Get model-230.pt to start with from https://drive.google.com/drive/folders/1s4fBeZvr23ma-a6eZMQqgeICyxiku6y_?usp=sharing
and place it into folder test/

Then generate an image as follows. Use your own image path however :D

```
python urdifs.py --dir test/ --name portr2 --eta 0.5 --steps 100  --modelSize 512 --w 768  --h 768 --model unet2 --mults 1 1 2 2 4 4 8 8 --text "an abstract portrait" --textw 50 --lr 10  --load test/model-230.pt --image /home/hannu/Pictures/hft269e-adjusted.png --skip 10  --ema
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
## training

To train a model you need a dataset of images. I typically use sets from a hundred to 10 000 images, starting training from a pretrained one and training from a few hours to a day or two on a single 3090. Training works best when the images are visually reasonably similar, i.e. we are training with visual features rather than content. 

Example

```
python urdiftrainer.py --images /work4/mat/fadedmem/ --lr 8e-5 --steps 1000 --accum 10 --dir test/ --imageSize 512 --batchSize 2 --saveEvery 100 --nsamples 2 --mults 1 1 2 2 4 4 8 8 --model unet2  --load /work/owntest/mdif2/md2-un2-od0-h2pa-ddims2/model-230.pt --losstype l1
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








