Abnormality detection is a crucial yet challenging task in medical image analysis. Distinguishing abnormalities from normal data by learning to reconstruct normal-only data alleviates the reliance on labeled datasets. However, many studies, even if unsupervised, rely on a labeled validation set to select the best model for inference from multiple training iterations. For many diseases labeled data are unavailable and substantially time consuming to obtain. To address this,  $\boldsymbol{AUC}_p$ - a novel metric that supports abnormality detection for unsupervised and self-supervised methods is proposed. Instead of evaluating the realism of reconstructed images to select the best of model for inference, it focuses on actual detection performance and without requiring an annotated test set. Assuming the pseudo ground truth of all unannotated samples in the test set as abnormal/positive and using traditional ${AUC}$ calculation,  ${AUC}_p$ scores are derived. Given a large and representative training set of normal samples, we show mathematical and empirical evidence that model selection using ${AUC}_p$ scores improves disease detection in terms of unsupervised and self-supervised methods over conventional metrics. Using two unsupervised methods for neurologic disease detection and self-supervised methods on diverse datasets, our results demonstrate that the ${AUC}_p$ score effectively identifies the optimal model for inference, significantly enhancing abnormality and disease detection.


## Environment

- Python 3.10
- PyTorch 2.1.2



## Data Preparation

We provide the pre-processed seven datasets. We use the same dataset as the MedIAnomaly: A comparative study of anomaly detection in medical images](https://arxiv.org/abs/2404.04518).

1. Download our pre-processed datasets from: [MedIAnomaly-Data](https://zenodo.org/records/12677223)
2. Download the ISIC2018_Task3_Training, ISIC2018_Task3_Test, and their ground truth from [ISIC2018](https://challenge.isic-archive.com/data/#2018)
3. Unzip the datasets via:

```shell
tar -zxvf RSNA.tar.gz
tar -zxvf VinCXR.tar.gz
tar -zxvf BrainTumor.tar.gz
tar -zxvf LAG.tar.gz
tar -zxvf ISIC2018_Task3.tar.gz
tar -zxvf Camelyon16.tar.gz
tar -zxvf BraTS2021.tar.gz
```

Place the `MedIAnomaly-Data` directory in the user's home directory, i.e., `~/MedIAnomaly-Data/`. (Otherwise, you need to modify the data root in your code.)

**Finally, the data path should have the following structure:**

```
~/MedIAnomaly-Data
├─RSNA
│  ├─images
│  └─data.json
├─VinCXR
│  ├─images
│  └─data.json
├─BrainTumor
│  ├─images
│  └─data.json
├─LAG
│  ├─images
│  └─data.json
├─ISIC2018_Task3
│  ├─ISIC2018_Task3_Training_Input
│  ├─ISIC2018_Task3_Training_GroundTruth
│  ├─ISIC2018_Task3_Test_Input
│  └─ISIC2018_Task3_Test_GroundTruth
├─Camelyon16
│  ├─train
│  │  ├─good
│  ├─test
│  │  ├─good
│  └─ └─Ungood
├─BraTS2021
│  ├─train
│  ├─test
│  │  ├─normal
│  │  ├─tumor
└─ └─ └─annotation
```


## Train & Evaluate

### [Reconstruction-based methods](./reconstruction)

- [x] AE ($\ell_2$, $\ell_1$, SSIM, Perceptual Loss)

- [x] AE-Spatial

- [x] VAE

- [x] Constrained AE

- [x] MemAE

- [x] CeAE

- [x] GANomaly

- [x] AE-U

- [x] DAE

- [x] AE-Grad

- [x] VAE-Grad ($Grad_{ELBO}$, $Grad_{KL}$, $Grad_{rec}$, $Grad_{Combi}$)



Train and evaluate these methods via:

```bash
cd reconstruction/;
./train_eval.sh
```

**[Reproduce the results in [AE4AD](https://github.com/caiyu6666/AE4AD)]** Train and evaluate AE with different latent size via:
```bash
cd reconstruction/;
./train_eval_latent_size.sh
```


### [SSL-based methods](./ssl)

#### one-stage

- [x] CutPaste
- [x] FPI
- [x] PII
- [x] NSA



Train and evaluate these methods via:

```bash
cd ssl/one_stage/;
./train_eval.sh
```



#### two-stage

- [x] CutPaste
- [x] AnatPaste
- [x] ResNet18-ImageNet



Train and evaluate these methods via:

```bash
cd ssl/two_stage/;
./train_eval.sh
```
## AUCp Calculation and Evaluation
python ssl/one_stage/med_evaluation.py -d 'dataset' -s method -g $gpu_id -f 1

python ssl/two_stage/aucp_eval.py
