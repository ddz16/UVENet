# UVENet
 Pytorch implementation of [UVENet](https://arxiv.org/abs/2403.11506) (End-To-End Underwater Video Enhancement: Dataset and Model). 

## Environment

Our environment:
```
Ubuntu 18.04.6 LTS
CUDA Version: 12.1
```
Based on anaconda or miniconda, you can install the required packages as follows:

```setup
conda create -n uve python=3.9
conda activate uve

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html

python setup.py develop
```

## Data Preparation

Since the dataset files are too large, please download them yourself from [this link](https://drive.google.com/drive/folders/1kr3mYyctNbcnJdSR0hjEUkahoDd5cH0h?usp=sharing). To prepare the datasets as follows:

- **SUVE**

    Download three compressed files: `GT.zip` (all ground-truth videos), `UW.zip` (all underwater-style videos), `SUVE_test.zip` (underwater-style videos for testing). Then extract them to the `./datasets/SUVE` folder. It should be like `./datasets/SUVE/GT/xxxx`, `./datasets/SUVE/UW/xxxx`, and `./datasets/SUVE/UW_test/xxxx`, where `xxxx` is the name of one video. Please note that each ground-truth video corresponds to three underwater-style videos. For example, the video `basement_0001a` in the `./datasets/SUVE/GT` folder corresponds to the videos `basement_0001a_465`, `basement_0001a_472`, and `basement_0001a_677` in the `./datasets/SUVE/UW` folder.

- **MVK**
    
    Download the compressed file: `MVK.zip` (all real underwater videos). Then extract it to the `./datasets/MVK` folder. It should be like `./datasets/MVK/xxxx`, where `xxxx` is the name of one video.

## Train

We train our UVENet on four V100 GPUs. If you want to change the training parameters such as the number of GPUs and batch size, please modify the corresponding parameters in the `train_UVENet_SUVE.yml` file.

Run this command:

```
python basicsr/train.py -opt options/train_UVENet_SUVE.yml
```
Model checkpoints and logs will be saved to the `./experiments` folder.

## Inference

You can inference using either the model trained by yourself or the [pretrained model](https://drive.google.com/file/d/1KjwKFVQmb3KPyDS9l8BMc18CHveGDhWa/view?usp=drive_link) we provide. Please note that you need to change the `model_path` option in the commands below to the expected model path.

- **Inference on test videos of SUVE**
```
python inference/inference_uvenet_allvideos.py --input_path datasets/SUVE/UW_test --save_path results/UVENet/SUVE --gt_path datasets/SUVE/GT --model_path xxx
```

- **Inference on real videos of MVK**
```predict
python inference/inference_uvenet_allvideos.py --input_path datasets/MVK --save_path results/UVENet/MVK --model_path xxx
```

## Citation

```
@article{uvenet,
  title={End-To-End Underwater Video Enhancement: Dataset and Model},
  author={Dazhao Du, Enhan Li, Lingyu Si, Fanjiang Xu, Jianwei Niu},
  journal={arXiv preprint arXiv:2403.11506},
  year={2024}
}
```
