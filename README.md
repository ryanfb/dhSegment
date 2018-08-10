# dhSegment

dhSegment allows you to extract content (segment regions) from different type of documents. See [some examples here](https://dhlab-epfl.github.io/dhSegment/).

The corresponding paper is now available on [arxiv](https://arxiv.org/abs/1804.10371), to be presented as oral at [ICFHR2018](http://icfhr2018.org/).

It was created by [Benoit Seguin](https://twitter.com/Seguin_Be) and Sofia Ares Oliveira at DHLAB, EPFL.

## Installation and requirements
 See `INSTALL.md` to install environment and to use `dh_segment` package.
 
 *NB : a good nvidia GPU (6GB RAM at least) is most likely necessary to train your own models. We assume CUDA and cuDNN are installed.*

## Usage
#### Training
* You need to have your training data in a folder containing `images` folder and `labels` folder. The pairs (images, labels) need to have the same name (it is not mandatory to have the same extension file, however we recommend having the label images as `.png` files). 
* The annotated images in `label` folder are (usually) RGB images with the regions to segment annotated with a specific color
* The file containing the classes has the format show below, where each row corresponds to one class (including 'negative' or 'background' class) and each row has 3 values for the 3 RGB values. Of course each class needs to have a different code.
``` class.txt
0 0 0
0 255 0
...
```
* [`sacred`](https://sacred.readthedocs.io/en/latest/quickstart.html) package is used to deal with experiments and trainings. Have a look at the documentation to use it properly.

In order to train a model, you should run `python train.py with <config.json>`

## Demo
This demo shows the usage of dhSegment for page document extraction. It trains a model from scratch (optional) using the [READ-BAD dataset](https://arxiv.org/abs/1705.03311) and the annotations of [pagenet](https://github.com/ctensmeyer/pagenet/tree/master/annotations) (annotator1 is used).
In order to limit memory usage, the images in the dataset we provide have been downsized to have 1M pixels each.

__How to__


1. Get the annotated dataset [here](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip), which already contains the folders `images` and `labels` for training, validation and testing set. Unzip it into `model/pages`. 
```
cd demo/
wget https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip
unzip pages.zip
cd ..
```
2. (Only needed if training from scratch) Download the pretrained weights for ResNet :
```
cd pretrained_models/
python download_resnet_pretrained_model.py
cd ..
```
3. You can train the model from scratch with: 
    `python train.py with demo/demo_config.json` but because this takes quite some time,
    we recommend you to skip this and just download the [provided model](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip) (download and unzip it in `demo/model`)
```
cd demo/
wget https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip
unzip model.zip
cd ..
```
4. (Only if training from scratch) You can visualize the progresses in tensorboard by running `tensorboard --logdir .` in the `demo` folder.
5. Run `python demo.py`
6. Have a look at the results in `demo/processed_images`

## MicroPasts Photogrammetry Masking

### Training from scratch:

Download and extract the [MicroPasts / British Museum Photogrammetry Masking Dataset 1.0](https://archive.org/details/micropasts_masking_dataset_10): (**WARNING: 26GB**)
```
wget https://archive.org/download/micropasts_masking_dataset_10/micropasts_masking_dataset_10.zip
```

Split the images into train/val/test sets so that you wind up with the following directories:
```
micropasts/micropasts/train/images/
micropasts/micropasts/train/labels/
micropasts/micropasts/val_a1/images/
micropasts/micropasts/val_a1/labels/
micropasts/micropasts/test_a1/images/
micropasts/micropasts/test_a1/labels/
```

For the existing training below, a train/val/test split of 80/10/10 was used, i.e. 3773 image pairs for train, 471 image pairs for each of val and test. The exact filenames used for each in the existing training can be found [in these text files](https://gist.github.com/ryanfb/261472dbcd01c556be437c032fb089db) if desired.

For training from scratch, download the pretrained weights for ResNet :
```
cd pretrained_models/
python download_resnet_pretrained_model.py
cd ..
```
Then run: `python train.py with micropasts/micropasts_config.json`

### Using the existing model:

Download the [provided model](https://github.com/ryanfb/dhSegment/releases/download/micropastsv1/micropasts_model.zip) and unzip it in `micropasts/` (so there's a resulting directory `micropasts/micropasts_model`).
```
cd micropasts/
wget https://github.com/ryanfb/dhSegment/releases/download/micropastsv1/micropasts_model.zip
unzip micropasts_model.zip
cd ..
```

### Testing a trained model

Put images in `micropasts/micropasts/test_a1/images`. Run `python demo_micropasts.py`. Results will be in `micropasts/processed_images`.
