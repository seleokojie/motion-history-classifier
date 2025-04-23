# MHI-based Activity Classification

**Overview**
This project implements Motion History Image (MHI)–based activity classification for the six actions: walking, jogging, running, boxing, clapping, and waving.  

**Environment**
- Python 3.8.8  
- Dependencies in `requirements.txt`  

**Data Preparation**
1. Download the KTH Actions dataset and place all `.avi` files under `data/`.  
2. Copy `sequences.txt` (from the KTH download) into the project root.  

**Usage**
1. **Extract features**  
   ```bash
   python scripts/extract_features.py \
     --dataset-dir data/ \
     --sequences-txt sequences.txt \
     --output features.pkl
    ```
2. **Train the model**  
   ```bash
   python scripts/train_model.py \
    --features-file features.pkl \
    --model-out knn_model.pkl \
    --output-cm confusion_matrix.png
  ```
3. **Annotate video**
    ```bash
    python scripts/annotate_video.py \
     --input data/person11_walking_d1.avi \
     --model knn_model.pkl \
     --output results/person11_walking_annotated.avi
  ```