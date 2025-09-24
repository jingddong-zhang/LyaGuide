# Lyapunov Guidance of Flow Matching (Submitted to ICLR 2026)


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
