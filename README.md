# P2I-MI
[ECCV 2024] [Prediction Exposes Your Face: Black-box Model Inversion via Prediction Alignment](https://arxiv.org/abs/2407.08127)

![](https://img.shields.io/badge/arXiv-2407.08127-AE2525)

![](https://github.com/lyufan/P2I-MI/blob/main/assets/figure2.png)

# Requirement

Install the environment as follows:

```bash
# create conda environment
conda env create -f environment.yaml
conda activate p2i
```

# Preparation

### Dataset

- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (we devide the CelebA follow the [KED-MI](https://github.com/SCccc21/Knowledge-Enriched-DMI/issues/1),the config file can be found in our "./data" and "./inversion/data_files")
- [FaceScrub](http://vintage.winklerbros.net/facescrub.html) (config files in "./inversion/data_files")
- [Pubfig83](https://vision.seas.harvard.edu/pubfig83/) (as above)
- [FFHQ](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)

### Models

- Standard setting: we follow previous work (like [KED-MI](https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN) or [PLG-MI](https://drive.google.com/drive/folders/1Cf2O2MVvveXrBcdBEWDi-cMGzk0y_AsT)) to use their target models and evaluation models, and put them in folder: "./inversion/checkpoints".
- Distribution shifts setting: You can download target models and evaluation models at: https://drive.google.com/drive/folders/1QCm90NAxDWckjRBSvjYxawjLdOtPSpt5?usp=sharing, and put them in folder: "./inversion/checkpoints"
- Download pretrained model from this [link](https://drive.google.com/file/d/1RnnBL77j_Can0dY1KOiXHvG224MxjvzC/view?usp=sharing) and put it in folder "./pretrained_models".

# Training dataset Preparation
### 1.Generate synthesized data
```bash
python generate_imgs.py
```
+ "--stylegan_model_path": Download StyleGAN model from this [link](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing), put it in folder :"./pixel2style2pixel/pretrained_models"
### 2.Select real public data
```bash
python top_n_select.py --model target_model --data_name public_data_source
```
### 3.Get prediction
```bash
python get_logscore.py
```
# Train
```bash
python train.py
```
+ "--real_dataset_path": your select data's prediction from public dataset
+ "--dataset_path": synthesized data's prediction
+ "--label_path": your saved seed files from "generate_imgs.py"
+ "--stylegan_model_path": StyleGAN model from this [link](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing) in "./pixel2style2pixel/pretrained_models"
+ "--arcface_model_path": Arcface model from this [link](https://drive.google.com/file/d/1coFTz-Kkgvoc_gRT8JFzqCgeC3lAFWQp/view?usp=sharing), put it in folder :"./pretrained_models"
+ "--parsing_model_path": Parsing model from this [link](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812), put it in folder :"./pretrained_models"
+ "--log_path": Checkpoint saved path
# Aligned Ensemble Attack
```bash
python ensemble_attack.py
```
+ "--pretrained_model_path": Checkpoint in log_path/001/xxx.pth.tar
+ "--input": your select data's prediction from public dataset(real_dataset_path in train.py)
