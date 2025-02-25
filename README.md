# MaskGIT PyTorch

[![GitHub stars](https://img.shields.io/github/stars/valeoai/MaskGIT-pytorch.svg?style=social)](https://github.com/valeoai/MaskGIT-pytorch/stargazers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valeoai/MaskGIT-pytorch/blob/main/colab_demo.ipynb)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)
<img src="saved_img/frog.png" alt="drawing" width="25"/>

Welcome to the unofficial MaskGIT PyTorch repository. 
This project aims to provide an external reproduction of the results from [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200), a PyTorch reimplementation of the models, and pretrained weights.
Official JAX implementation of MaskGIT can be found [here](https://github.com/google-research/maskgit).

## Repository Structure

Here's an overview of the repository structure:
  
      ├ MaskGIT-pytorch/
      |    ├── Metrics/                               <- evaluation tool
      |    |      ├── inception_metrics.py                  
      |    |      └── sample_and_eval.py
      |    |    
      |    ├── Network/                             
      |    |      ├── Taming/                         <- VQGAN architecture   
      |    |      └── transformer.py                  <- Transformer architecture  
      |    |
      |    ├── Trainer/                               <- Main class for training
      |    |      ├── trainer.py                      <- Abstract trainer     
      |    |      └── vit.py                          <- Trainer of maskgit
      |    ├── save_img/                              <- Image samples         
      |    |
      |    ├── colab_demo.ipynb                       <- Inference demo 
      |    ├── download_models.py                     <- download the pretrained models
      |    ├── LICENSE.txt                            <- MIT license
      |    ├── requirements.yaml                      <- help to install env 
      |    ├── README.md                              <- Me :) 
      |    └── main.py                                <- Main

## Usage

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/valeoai/MaskGIT-pytorch.git
   cd MaskGIT-pytorch

2. Install requirement 

   ```bash
   conda env create -f environment.yaml
   conda activate maskgit

3. (Opt.) Download Pretrained models  

   ```bash
   python download_models.py
   
4. Resume training for 1 additional epoch
   ```bash
   data_folder="/datasets_local/ImageNet/"
   vit_folder="./pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth"
   vqgan_folder="./pretrained_maskgit/VQGAN/"
   writer_log="./logs/"
   num_worker=16
   bsize=64
   # Single GPU
   python main.py  --bsize ${bsize} --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}" --writer-log "${writer_log}" --num_workers ${num_worker} --img-size 256 --epoch 301 --resume
   # Multiple GPUs single node
   torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py  --bsize ${bsize} --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}" --writer-log "${writer_log}" --num_workers ${num_worker} --img-size 256 --epoch 301 --resume
## Demo

You are interested only in the inference of the model? You can run the demo_colab.ipynb in google collab! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valeoai/MaskGIT-pytorch/blob/main/colab_demo.ipynb)

## Training Details

The model consists of a total of 246.303M parameters, with 174.161M for the transformer and 72.142M for VQGAN.
The VQGAN reduces a 256x256 (resp. 512x512) image to a 16x16 (resp. 32x32) token representation, over a bank of 1024 possible codes.
During the masked transformer training, I used a batch size of 512 over 300 epochs, leveraging 8 GPUs (~768 GPUs/hour on Nvidia A100) for 755 200 iterations on ImageNet 256x256.
Then, I finetune the same model on ~750 000 iterations on ImageNet 512x512 with a batch size of 128 and ~384 GPUs/hour on Nvidia A100.

The transformer architecture hyperparameters:

| Hidden Dimension | Codebook Size | Depth | Attention Heads | MLP Dimension | Dropout Rate |
|------------------|---------------|-------|-----------------|---------------|--------------|
| 768              | 1024          | 24    | 16              | 3072          | 0.1          |

The optimizer employed is Adam with a learning rate of 1e-4, utilizing an 'arccos' scheduler for masking. Additionally, during training, I applied a 10% dropout for the CFG.

For all the details, please refer to our [technical report](https://arxiv.org/abs/2310.14400v1)

## Performance on ImageNet

Using the following hyperparameters for sampling:

| Image Size | Softmax Temp | Gumbel Temp | CFG (w) | Randomization | Schedule | Schedule Step |   
|------------|--------------|-------------|---------|---------------|----------|---------------|
| 256*256    | 1            | 4.5         | 3       | "linear"      | "arccos" | 8             |    
| 512*512    | 1            | 7           | 2.8     | "linear"      | "arccos" | 15            |    

We reach this performance on ImageNet:

| Metric                           | Ours 256*256 | Paper 256*256 | | Ours 512*512 | Paper 512*512 |
|----------------------------------|--------------|---------------|-|--------------|---------------|
| FID (Fréchet Inception Distance) | 6.80         | 6.18          | | 7.26         | 7.32          |
| IS (Inception Score)             | 214.0        | 182.1         | | 223.1        | 156.0         |
| Precision                        | 0.82         | 0.80          | | 0.85         | 0.78          |
| Recall                           | 0.51         | 0.51          | | 0.49         | 0.50          |
| Density                          | 1.25         | -             | | 1.33         | -             | 
| Coverage                         | 0.84         | -             | | 0.86         | -             |

The IS rises monotonically along the training while the FID decrease:

<img src="saved_img/perf_along_train.png" alt="drawing" width="1024"/>

For visualization, to boost the image quality, we increase the amount of steps (32) the softmax temperature (1.3) and the cfg weight (9) to trade diversity for fidelity
###  Performance on ImageNet 256
![sample](saved_img/256_256/sample.png)

### Performance on ImageNet 512
![sample](saved_img/512_512/sample.png)

And generation process:
![sample](saved_img/512_512/gen_process_bear.png)
![sample](saved_img/512_512/gen_process_rabbit.png)

## Inpainting
The model demonstrates good capabilities in inpainting ImageNet-generated images into scenes:
<img src="saved_img/7.png" alt="drawing" width="1024"/>

## Pretrained Model

You can download the pretrained MaskGIT models in [hugging face](https://huggingface.co/llvictorll/Maskgit-pytorch/tree/main).

## Contribute

The reproduction process might encounter bugs or issues, or there could be mistakes on my part. If you're interested in collaborating or have suggestions, please feel free to reach out (by [creating an issue](https://github.com/valeoai/MaskGIT-pytorch/issues/new)). Your input and collaboration are highly valued!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgement

This project is powered by IT4I Karolina Cluster located in the Czech Republic. 

The pretrained VQGAN ImageNet (f=16), 1024 codebook. The implementation and the pre-trained model is coming from the [VQGAN official repository](https://github.com/CompVis/taming-transformers/tree/master)

## BibTeX
If you find our work beneficial for your research, please consider citing both our work and the original source.
```
@misc{besnier2023MaskGit_pytorch,
      title={A Pytorch Reproduction of Masked Generative Image Transformer}, 
      author={Victor Besnier and Mickael Chen},
      year={2023},
      eprint={2310.14400},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@InProceedings{chang2022maskgit,
  title = {MaskGIT: Masked Generative Image Transformer},
  author={Huiwen Chang and Han Zhang and Lu Jiang and Ce Liu and William T. Freeman},
  booktitle = {CVPR},
  month = {June},
  year = {2022}
}
```

## Stars History

[![Star History Chart](https://api.star-history.com/svg?repos=valeoai/Maskgit-pytorch&type=Date)](https://star-history.com/#valeoai/Maskgit-pytorch&Date)
