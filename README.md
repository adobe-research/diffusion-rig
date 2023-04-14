# DiffusionRig

**DiffusionRig: Learning Personalized Priors for Facial Appearance Editing**<br>
[Zheng Ding](), [Xuaner Zhang](https://ceciliavision.github.io),
[Zhihao Xia](https://likesum.github.io), [Lars Jebe](https://lcjebe.github.io),
[Zhuowen Tu](https://pages.ucsd.edu/~ztu), [Xiuming Zhang](https://xiuming.info)
<br>CVPR 2023<br>
[arXiv](https://arxiv.org/pdf/2304.06711.pdf) / [Project Page](https://diffusionrig.github.io) / [Video](https://www.youtube.com/watch?v=6ZQbiNiJJEE) / [BibTex](bib.txt)

![teaser](figs/teaser.webp)

## Setup & Preparation

### Environment Setup

```bash
conda create -n diffusionrig python=3.8
conda activate diffusionrig
conda install pytorch=1.11 cudatoolkit=11.3 torchvision -c pytorch
conda install mpi4py dlib scikit-learn scikit-image tqdm -c conda-forge
pip install lmdb opencv-python kornia yacs blobfile chumpy face_alignment
```

You need to also install [pytorch3d](https://github.com/facebookresearch/pytorch3d) to render the physical buffers:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

### DECA Setup

Before doing data preparation for training, please first download the source files and checkpoints of DECA to set it up (you will need to create an account to download FLAME resources):
1. `deca_model.tar`: Visit [this page](https://github.com/YadiraF/DECA#:~:text=You%20can%20also%20use%20released%20model%20as%20pretrained%20model%2C%20then%20ignor%20the%20pretrain%20step.) to download the pretrained DECA model.
2. `generic_model.pkl`: Visit [this page](https://flame.is.tue.mpg.de/download.php) to download `FLAME 2020` and extract `generic_model.pkl`.
3. `FLAME_texture.npz`: Visit [this same page](https://flame.is.tue.mpg.de/download.php) to download the `FLAME texture space` and extract `FLAME_texture.npz`.
4. Download the other files listed below from [DECA's Data Page](https://github.com/YadiraF/DECA/tree/master/data) and put them also in the `data/` folder:

```bash
data/
  deca_model.tar
  generic_model.pkl
  FLAME_texture.npz
  fixed_displacement_256.npy
  head_template.obj
  landmark_embedding.npy
  mean_texture.jpg
  texture_data_256.npy
  uv_face_eye_mask.png
  uv_face_mask.png
```

### Data Preparation

We use FFHQ to train the first stage and a personal photo album to train the second stage. Before training, you need to extract, with DECA, the physical buffers for those images.

For FFHQ, you need to align the images first with:

```bash
python scripts/create_data.py --data_dir PATH_TO_FFHQ_ALIGNED_IMAGES --output_dir ffhq256_deca.lmdb --image_size 256 --use_meanshape False
```

For the personal photo album (we use around 20 per identity in our experiments), put all images into a folder and then align them by running:

```bash
python scripts/align.py -i PATH_TO_PERSONAL_PHOTO_ALBUM -o personal_images_aligned -s 256
```

Then, create a dataset by running:

```bash
python scripts/create_data.py --data_dir personal_images_aligned --output_dir personal_deca.lmdb --image_size 256 --use_meanshape True
```

## Training

### Stage 1: Learning Generic Face Priors

Our 256x256 model uses eight GPUs for Stage 1 training with a batch size of 32 per GPU:

```bash
mpiexec -n 8 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log/stage1 --data_dir ffhq256_deca.lmdb --lr 1e-4 \
    --p2_weight True --image_size 256 --batch_size 32 --max_steps 50000 \
    --num_workers 8 --save_interval 5000 --stage 1
```

To keep the model training indefinitely, set `--max_steps 0`. If you want to resume a training process, simply add `--resume_checkpoint PATH_TO_THE_MODEL`.

:white_check_mark: We also provide the Stage 1 model trained by us [here](https://drive.google.com/file/d/1lnFLNGguvQ150unuOXJqbXiVQtJz0jli/view?usp=sharing) so that you can fast-forward to training your personalized model.

### Stage 2: Learning Personalized Priors

Finetune the model on your tiny personal album:

```bash
mpiexec -n 1 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log/stage2 --resume_checkpoint log/stage1/[MODEL_NAME].pt \
    --data_dir peronsal_deca.lmdb --lr 1e-5 \
    --p2_weight True --image_size 256 --batch_size 4 --max_steps 5000 \
    --num_workers 8 --save_interval 5000 --stage 2
```

It takes around 30 minutes on a single Nvidia V100 GPU.

## Inference

We provide a script to edit face appearance by modifying the physical buffers. Run:

```bash
python scripts/inference.py --source SOURCE_IMAGE_FILE --target TARGET_IMAGE_FILE --output_dir OUTPUT_DIR --modes light --model_path PATH_TO_MODEL --meanshape PATH_TO_MEANSHAPE --timestep_respacing ddim20 
```

to use the physical parameters (e.g., lighting, expression, or head pose) of the target image to edit the source image.


## Issues or Questions?

If the issue is code-related, please open an issue here.

For questions, please also consider opening an issue as it may benefit future reader. Otherwise, email Zheng Ding at 
[zhding@ucsd.edu](mailto:zhding@ucsd.edu).


## Acknowledgements

This codebase was built upon and drew inspirations from [Guided-Diffusion](https://github.com/openai/guided-diffusion), [DECA](https://github.com/yfeng95/DECA) and [Diff-AE](https://github.com/phizaz/diffae). We thank the authors for making those repositories public.
