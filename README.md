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

# Training dataset Preparation
### Generate synthesized data
```python
python generate_imgs.py
```
### Select real public data
```python
python top_n_select.py --model target_model --data_name public_data_source
```
### Get prediction
```python
python get_logscore.py
```
# Train
```python
python train.py
```

# Aligned Ensemble Attack
```python
python ensemble_attack.py
```
