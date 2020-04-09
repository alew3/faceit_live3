# faceit_live3
This is an update to http://github.com/faceit_live using [first order model](https://github.com/AliaksandrSiarohin/first-order-model) by Aliaksandr Siarohin to generate the images. This model only requires a single image, so no training is needed and things are much easier.

# Setup

## Requirements
This has been tested on **Ubuntu 18.04 with a Titan RTX/X GPU**.
You will need the following to make it work:

    Linux host OS
    NVidia fast GPU (GTX 1080, GTX 1080i, Titan, etc ...)
    Fast Desktop CPU (Quad Core or more)
    NVidia CUDA 10 and cuDNN 7 libraries installed
    Webcam


## Setup Host System
To use the fake webcam feature to enter conferences with our stream we need to insert the **v4l2loopback** kernel module in order to create */dev/video1*. Follow the install instructions at  (https://github.com/umlaeute/v4l2loopback), then let's setup our fake webcam:

```
$ git clone https://github.com/umlaeute/v4l2loopback.git
$ make && sudo make install
$ sudo depmod -a
$ sudo modprobe v4l2loopback devices=1
$ sudo modprobe v4l2loopback exclusive_caps=1 card_label="faceit_live" video_nr=1
$ v4l2-ctl -d /dev/video1 -c timeout=1000
```


Change the video_nr above in case you already have a webcam running on /dev/video1

To check if things are working, try running an mp4 to generate a video the */dev/video1* (replace ale.mp4 with your own video).
```
$ ffmpeg -re -i media/ale.mp4 -f v4l2 /dev/video1 -loop 10
```
And view it 
```
$ ffplay -f v4l2 /dev/video1
```

On Ubuntu 18, I had to make a minor change to the source code of v4l2loopback.c to get loopback working. In case the above doesn't work, you can try this change before running *make* : 

```
# v4l2loopback.c
from
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 29)

to
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,7,0)
```

You can also inspect your /dev/video* devices:

```
$ v4l2-ctl --list-devices
$ v4l2-ctl --list-formats -d /dev/video1

```


If you have more than one GPU, you might need to set some environment variables:
```
# specify which display to use for rendering
$ export DISPLAY=:1

# which CUDA DEVICE to use (run nvidia-smi to discover the ID)
$ export CUDA_VISIBLE_DEVICES = 0

```

## Clone this repository
Don't forget to use the *--recurse-submodules* parameter to checkout all dependencies.

    $ git clone --recurse-submodules https://github.com/alew3/faceit_live3.git /local_path/

## Create an Anaconda environment and install requirments
```
$ conda create -n faceit_live3 python=3.8
$ source activate faceit_live3
$ conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.1 -c pytorch
$ pip install -r requirements.txt

```

## Download 'vox-adv-cpk.pth.tar' to /models folder

You can find it at: [google-drive](https://drive.google.com/open?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH) or [yandex-disk](https://yadi.sk/d/lEw8uRm140L_eQ).


# Usage

Put in the `./media/` directory the images in jpg/png you want to play with.


# Run the program

```
$ python faceit_live.py
```

## Shortcuts
    --webcam # the videoid of the Webcam e.g. 0 if /dev/video0 (default is 0)
    --image # the face to use for transformations, put the files inside media (by default it loads the first image in the folder)
    --streamto # the /dev/video number to stream to (default is 1)

## Example
```
$ python faceit_live.py --webcam_id 0 --stream_id 1
```

## Shortcuts when running
```
N - cycle next image in media folder
C - recenter webcam
```