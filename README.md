# Pokemon Image Classifier

## Overview
This project implements a deep learning-based image classifier for first-generation Pokémon. Using TensorFlow and Keras, the model is trained to classify Pokémon images from datasets sourced from Kaggle. The optimized model achieves approximately **89% validation accuracy**. Additionally, comparison models using ANN and K-Means clustering are provided for comparison.

---

## Features
- Fine-tuned VGG16 architecture with data augmentation and optional fine-tuning.
- Distributed training using Horovod for scalability.
- Configurable training parameters including batch size, epochs, and augmentation settings.
- Visualization of training accuracy and validation accuracy trends.
- Comparison with baseline models (ANN and K-Means).

---

## Datasets
The classifier is trained using datasets containing images of first-generation Pokémon:

1. [Pokémon Images (17,000 files)](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files/data)
2. [Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data)
3. [Pokémon Gen 1 Dataset (38,914 files)](https://www.kaggle.com/datasets/echometerhhwl/pokemon-gen-1-38914)

These datasets provide a diverse set of labeled Pokémon images for training and evaluation. 
Data directory contains 51,444 total images.

---

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Horovod
- Keras
- Matplotlib
- CUDA-compatible GPU and drivers (optional, for GPU acceleration)

### Installation Steps
1. Clone this repository.
2. Install dependencies:
    ```bash
    pip install tensorflow keras horovod matplotlib
    ```
3. Ensure the datasets are downloaded and structured as follows:
    ```
    <stratified-data>/
        train/
            class_1/
            class_2/
            ...
        test/
            class_1/
            class_2/
            ...
    ```

---

## Usage

### Command-Line Interface
Run the main script using the following command:
```bash
python optimized-model.py --data <path_to_data> \
                     --batch_size <batch_size> \
                     --epochs <num_epochs> \
                     --main_dir <output_directory> \
                     --augment_data <true_or_false> \
                     --fine_tune <true_or_false>
```

### Arguments 
- `--data`: Path to the dataset directory (no trailing slash).
- `--batch_size`: Batch size for training.
- `--epochs`: Number of epochs for training.
- `--main_dir`: Directory to save outputs (models, plots).
- `--augment_data`: Boolean to enable data augmentation (default: `false`).
- `--fine_tune`: Boolean to enable fine-tuning of the base model (default: `false`).

---

### Running with Shell Script
The provided shell script facilitates running the model on a Slurm-based cluster with the following configuration:

- Partition: `dgx`
- GPUs: 8
- CPUs per GPU: 9
- Time limit: 3 hours 

To run the script:

1. Ensure your Python script and the shell script are in the same directory.
2. Submit the job using:

```bash
sbatch pokemon-optimized-model.sh
```

Here is the sample shell script (pokemon-optimized-model.sh):

```bash
#!/bin/bash

#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=9
#SBATCH --error='sbatcherrorfile.out'
#SBATCH --time=0-3:0

container="/data/containers/msoe-horovod-tf1.sif"
command="horovodrun -np 8 -H localhost:8 python optimized-model.py --data /home/woodm/CSC2611/pokemon-image-classifier/stratified-data --batch_size 16 --epochs 10 --main_dir /home/woodm/CSC2611/pokemon-image-classifier/src/optimized-model --augment_data true --fine_tune true"

singularity exec --nv -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}

```

---


## Model Architecture
The classifier is built on the VGG16 architecture with the following modifications:

1. The base VGG16 model is initialized with ImageNet weights and frozen during initial training.
2. A global average pooling layer is added after the convolutional base.
3. A dense layer with 149 output classes (for Pokémon) and softmax activation is used for classification.

Fine-tuning unfreezes the base model to further improve accuracy by adapting the weights to the Pokémon dataset.

---

### Results

#### Optimized Model

- **Validation Accuracy**: ~89%
- Fine-tuning improves accuracy marginally in later epochs.

#### Comparison Models

1. **Artificial Neural Network (ANN)**
   - Used dense layers for classification.
   - Accuracy: ~1%.
2. **K-Means Clustering**
   - Used image features for clustering.
   - Accuracy: ~4%.

---

### Output

1. **Model File**: Saved as `.h5` in the specified output directory.
2. **Training Plots**: Accuracy plots for training and validation.

---

### Acknowledgments
Thanks to the creators of the datasets for providing the resources to train and evaluate this model.


#### Authors

- Michael Wood
  - [GitHub](https://github.com/woodrmichael)
  - [LinkedIn](https://www.linkedin.com/in/woodrmichael/)
- Alec Weinbender
  - [GitHub](https://github.com/weinbendera)
  - [LinkedIn](https://www.linkedin.com/in/alec-weinbender/)
- Pranaav Paladugu
  - [GitHub](https://github.com/pranaavP)
  - [LinkedIn](https://www.linkedin.com/in/pranaav-paladugu-545b02308/)