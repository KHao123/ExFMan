# ExFMan: Rendering 3D Dynamic Humans with Hybrid Monocular Blurry Frames and Events

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

Below we take the subject 386 as an example. Due to licensing restrictions, you will need to request access to the dataset from the original authors. We will release the data processing code soon.


### `Prepare a dataset`

First, download ZJU-Mocap-blur dataset from [here](...). 


### `Train/Download models`

Now you can either download a pre-trained model [here](...).



or train a model by yourself. We used 4 GPUs (NVIDIA RTX 2080 Ti) to train a model. 

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

