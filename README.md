# Fine‑Grained Perception for Generalized Medical Image Segmentation (MedFineSAM)

<img src="https://github.com/user-attachments/assets/80e90b71-20c3-4e28-9b09-3e748a610dee" width="75%">

## Requirement
``pip install -r requirements.txt``


## Data Preparation
[Prostate Segmentation](https://liuquande.github.io/SAML/)<br>
[Drishti-GS](https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation/data)<br>
[RIM-ONE r3]()<br>
[RIGA+ Segmentation](https://zenodo.org/records/6325549)<br>
REFUGE can be downloaded directly via [Baidu Netdisk](https://pan.baidu.com/s/1400JPodPk_zkcBGCspgMfQ?pwd=9dpo) or [Google Drive](https://drive.google.com/file/d/1lIBJTbRy2v6l3zary3YkXp4ZOwDPcrWl/view?usp=sharing)<br>
We got the REFUGE datasets from [PCSDG](https://github.com/HopkinsKwong/PCSDG)<br>
Please download the pretrained [SAM model](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view?usp=share_link) 
(provided by the original repository of SAM) and put it in the ./pretrained folder. 

## Prostate Segmentation
Source = RUNMC，Target = BIDMC / BMC / HK / I2CVB / UCL

```bash
cd prostate

# Training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --root_path dataset_path \
    --output output_path \
    --Source_Dataset RUNMC \
    --Target_Dataset BIDMC BMC HK I2CVB UCL

# Test
CUDA_VISIBLE_DEVICES=0 python test.py \
    --root_path dataset_path \
    --output_dir output_path \
    --Source_Dataset RUNMC \
    --Target_Dataset BIDMC BMC HK I2CVB UCL \
    --snapshot snapshot_path
```

## RIGA+ Segmentation
Source = BinRushed，Target = MESSIDOR_Base1 / Base2 / Base3

```bash
cd fundus

# Training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --root_path dataset_path \
    --output output_path \
    --Source_Dataset BinRushed \
    --Target_Dataset MESSIDOR_Base1 MESSIDOR_Base2 MESSIDOR_Base3

# Test
CUDA_VISIBLE_DEVICES=0 python test.py \
    --root_path dataset_path \
    --output output_path \
    --Source_Dataset BinRushed \
    --Target_Dataset MESSIDOR_Base1 MESSIDOR_Base2 MESSIDOR_Base3 \
    --snapshot snapshot_path
```

## Acknowledgement

We appreciate the developers of [Segment Anything Model](https://github.com/facebookresearch/segment-anything). 
The code of DAPSAM is built upon [DAPSAM](https://github.com/wkklavis/DAPSAM), and we express our gratitude to these projects.
