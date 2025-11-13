<img width="1508" height="967" alt="image" src="https://github.com/user-attachments/assets/77af1864-ee65-46c3-9f27-05fafaa5cc92" /># Fine‑Grained Perception for Generalized Medical Image Segmentation (MedFineSAM)


<img width="1508" height="967" alt="image" src="https://github.com/user-attachments/assets/80e90b71-20c3-4e28-9b09-3e748a610dee" />

## Requirement
``pip install -r requirements.txt``


## Data Preparation
[Prostate Segmentation](https://liuquande.github.io/SAML/)
[Drishti-GS](https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation/data)
[RIM-ONE r3]()
[RIGA+ Segmentation](https://zenodo.org/records/6325549)
We got the REFUGE datasets from [PCSDG](https://github.com/HopkinsKwong/PCSDG)


Please download the pretrained [SAM model](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view?usp=share_link) 
(provided by the original repository of SAM) and put it in the ./pretrained folder. 

## Acknowledgement

We appreciate the developers of [Segment Anything Model](https://github.com/facebookresearch/segment-anything). 
The code of DAPSAM is built upon [DAPSAM](https://github.com/wkklavis/DAPSAM), and we express our gratitude to these projects.
