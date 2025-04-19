# ExFMan: Rendering 3D Dynamic Humans with Hybrid Monocular Blurry Frames and Events


https://github.com/user-attachments/assets/5744a7a8-f275-44b2-9c02-e611667a5e97


## TODO list
- [x] Code for training and inference
- [ ] ZJU-MOCAP Dataset processing code
- [ ] Real-world dataset

## Prerequisite

### `Configure environment`

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/).

Create and activate a virtual environment.

    conda create --name exfman python=3.7
    conda activate exfman

Install the required packages.

    pip install -r requirements.txt

### `Download SMPL model`

Download the gender neutral SMPL model from [here](https://smplify.is.tue.mpg.de/), and unpack **mpips_smplify_public_v2.zip**.

Copy the smpl model.

    SMPL_DIR=/path/to/smpl
    MODEL_DIR=$SMPL_DIR/smplify_public/code/models
    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models

Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.


## Run on ZJU-Mocap-blur Dataset

Below we take the subject 386 as an example. Due to license restrictions, you will need to request access to the dataset from the original authors from [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset). We will release the data processing code soon.


### `Prepare a dataset`

First, download 386 subject of ZJU-Mocap-blur dataset from [here](https://hkustgz-my.sharepoint.com/:u:/g/personal/kchen879_connect_hkust-gz_edu_cn/Ed9q9po8OndPpKbOdzCWRncBLgi7Bckx8usl_xl-XfSbQQ?e=SJe52m). 


### `Train/Download models`

Now you can either download a pre-trained model for 386 [here](https://hkustgz-my.sharepoint.com/:u:/g/personal/kchen879_connect_hkust-gz_edu_cn/ESbFByiYULlOpCezWtMdCCoB84nCPvhtflPJjuIUgmVC7w?e=wRB3gu).



or train a model by yourself. We used single GPUs (A800) to train a model. 

    python train.py --cfg configs/human_nerf/zju_mocap/386/exfman.yaml


### `Render output`

Render the frame input (i.e., observed motion sequence).

    python run.py \
        --type movement \
        --cfg configs/human_nerf/zju_mocap/386/exfman.yaml 

Run free-viewpoint rendering on a particular frame (e.g., frame 128).

    python run.py \
        --type freeview \
        --cfg configs/human_nerf/zju_mocap/386/exfman.yaml \
        freeview.frame_idx 128


Render the learned canonical appearance (T-pose).

    python run.py \
        --type tpose \
        --cfg configs/human_nerf/zju_mocap/386/exfman.yaml 



## Acknowledgement

The implementation took reference from [HumanNeRF](https://github.com/chungyiweng/humannerf) [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch), [Neural Body](https://github.com/zju3dv/neuralbody). We thank the authors for their generosity to release code.

