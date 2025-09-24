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

```

and 

```
bash script/train_ceg.sh

```

for the exact diffusion guidance of the contrastive energy guidance.

Then, guidance methods can be evaluated using the notebooks: ```notebooks/fig.ipynb``` and notebooks/mc.ipynb to reproduce Figure 2 and Figure 4.

You can play around with other notebooks to see the guidance quality of different methods, including gradient, contrastive energy guidance, and out g^MC.
