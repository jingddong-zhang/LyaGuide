# Lyapunov Guidance of Flow Matching (Submitted to ICML 2026)


---

We propose a unifed framework for guidance of flow matching: Lyapunov Guidance, which accommodates all the existing guidance methods into a single Lyapunov control problem.

We also devise the Lyapunov pseudo projection method, which enhances the performance of existing guidance methods with only one-line code.

# Synthetic Dataset Experiments

## Installation
In the `synthetic` folder and with Python 3.11 installed, install the following packages:
```bash
conda env create -f environment.yml
conda activate guided_flow
pip install -e .
```

## Datasets

The datasets are generated during training, as the distributions are relatively simple.

## Reproducing the Results

First, train the base models:
```
bash script/train_cfm.sh

```

Note that to run training-based guidance methods, you need first to train the guidance models using:
```
bash script/train_guidance_matching.sh
script/train_guidance_matching_lya_s1.sh
```


```
bash script/train_ceg.sh

```



For the CEG method in Scenario 1
```
bash script/train_ceg_lya_s1.sh

```

For the g_phi method in Scenario 2 in three tasks.
```
bash script/train_lya.sh

```


Then, guidance methods can be evaluated using the notebooks: 

```notebooks/fig_lya_8gaussian.ipynb``` and  ```notebooks/fig_lya_scurve.ipynb``` to reproduce Figure 2.

```notebooks/fig_lya_s2.ipynb``` to reproduce Figure 3.

```notebooks/fig_lya_variants.ipynb``` to reproduce comparison of ES/AS/CS on 8 gaussian generation task in Appendix.


# Image Inverse Problem Experiments

## Installation

In the `image` folder and with Python 3.11 installed, install the following packages:
```
pip install -r requirements.txt
pip install -e .
```

## Datasets
We downloaded the Celeba-HQ dataset from Kaggle, which contains 30,000 high-quality celebrity faces resampled to 256px. This dataset was used by NVIDIA in the research paper “Progressive Growing of GANs for Improved Quality, Stability, and Variation.” Before feeding the data into the model, the values were normalized to the range of 0 to 1. The data was then randomly split into 8:1:1 ratios for training, testing, and validation, with the corresponding split file being image/gflow_img/config/celeba_hq_splits.json.

## Reproducing the results

First, download the CelebA-HQ dataset and put it in `./data_cache/celeba_hq_256`, train the model of CelebA 256 with 
```
accelerate launch run/main_train.py
```
The model will be saved in `./results/{cfm,ot}_punet256_celeba256`.

Then, evaluate different guidance methods using scripts in  `.scripts/baseline` and `.scripts/lya`.
