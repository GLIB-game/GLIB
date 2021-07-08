# GLIB

CNN-based visual understanding for detecting UI glitches in game Apps
>> GLIB: Towards Automated Test Oracle for Graphically-Rich Applications <br> Paper URL: https://arxiv.org/abs/2106.10507

## Architecture

![Image](GLIB_architecture.png)

## Code-based Generation

![Image](code_gen.png)

<!-- ### Rule-based Generation:

![Image](rule_gen.png) -->

## Requirements

On Ubuntu:

- Python3.5.2
- pytorch(0.4.0)
- cuda(9.0.176)
- cudnn(7.4.2)



## Installation

##### Step0: Clone the GLIB repository

```shell
git clone --recursive https://github.com/GLIB-game/GLIB.git 
cd GLIB
pip install -r requirements.txt
```

##### Step1: Download dataset 

download the UI image [dataset](https://zenodo.org/record/5081242/files/data.zip?download=1) and unzip:

```shell
unzip data.zip
```

data/images: 

- *data/images/Base* : 132 screenshots of game1 & game2 with UI display issues from 466 test reports.
- *data/images/Code* : 9,412 screenshots of game1 & game2 with UI display issues generated by our Code augmentation method.
- *data/images/Normal*: 7,750 screenshots of game1 & game2 without UI display issues collected by randomly traversing the game scene.
- *data/images/Rule(F)* : 7,750 screenshots of game1 & game2 with UI display issues generated by our Rule(F) augmentation method.
- *data/images/Rule(R)* : 7,750 screenshots of game1 & game2 with UI display issues generated by our Rule(R) augmentation method.
- *data/images/testDataSet* : 192 screenshots  with UI display issues from 466 test reports(exclude game1 & game2).

data/data_csv:

- *data/data_csv/Base* : dataset for baseline method.
- *data/data_csv/Code* : dataset for our Code Augmentation method.
- *data/data_csv/Rule(F)* : dataset for our Rule(F) Augmentation method.
- *data/data_csv/Rule(R)* : dataset for our Rule(R) Augmentation method.
- *data/data_csv/Code_plus_Rule(F)* : dataset for our Code&Rule(F) Augmentation method.
- *data/data_csv/Code_plus_Rule(R)* : dataset for our Code&Rule(R) Augmentation method.
- *data/data_csv/testDataSet* : test dataset(normal image and real glitch images from 466 test reports).

download the pre-trained [model](https://zenodo.org/record/5081280/files/model.zip?download=1) and unzip:

```shell
unzip model.zip
```

- *model/Base* : pre-trained model for baseline method.
- *model/Code* : pre-trained model for our Code Augmentation method.
- *model/Rule(F)* : pre-trained model for our Rule(F) Augmentation method.
- *model/Rule(R)* : pre-trained model for our Rule(R) Augmentation method.
- *model/Code_plus_Rule(F)* : pre-trained model for our Code&Rule(F) Augmentation method.
- *model/Code_plus_Rule(R)* : pre-trained model for our Code&Rule(R) Augmentation method.

##### Step2: Train CNN Model

Training from scratch:

```shell
python train.py --train_data train_file_path --eval_data eval_file_path --augType Type
```

Example:

```shell
python train.py --train_data data/data_csv/Code/Code_train.csv --eval_data data/data_csv/Code/Code_test.csv --augType Code
```

Training from pre-trained model:

```shell
python train.py --train_data train_file_path --eval_data eval_file_path --augType Type --model_path model_path
```

Example:

```shell
python train.py --train_data data/data_csv/Code/Code_train.csv --eval_data data/data_csv/Code/Code_test.csv --augType Code --model_path model/Code/Code.pkl
```

##### Step3: Evaluate Model

```shell
python test.py --test_data test_data_path --model model_path
```

Example:

```shell
python test.py --test_data data/data_csv/testDataSet/testData_test.csv --model model/Code/Code.pkl
```

##### Step4: Generate Saliency Map

```shell
python saliencymap.py --test_data test_data_path --model model_path
```

Example:

```shell
python saliencymap.py --test_data data/data_csv/testDataSet/testData_test.csv --model model/Code/Code.pkl
```



## Configuration

Changing hyper-parameters is possible by editing the file [config.py](https://github.com/GLIB-game/GLIB/blob/main/config.py)

##### config.EPOCH:

The max number of epochs to train the model. Stopping earlier must be done manually (kill).

##### config.TRAIN_BATCH_SIZE:

Batch size in training.

##### config.SAVE_STEP:

After how many training steps a model should be saved.

##### config.EVAL_STEP:

After how many training steps the model test its performance on evaluation dataset.

##### config.LR:

The learning rate in training.

##### config.EVAL_BATCH_SIZE

Batch size in evaluation step.

##### config.TEST_BATCH_SIZE

Batch size in test step.



## Supplementary explanation

#### The correlation between our self-defined code & rule approaches and corresponding UI glitches:
![Image](Method_2_UIglitch.png)

#### Reult for Practical Evaluation (RQ4)

|         |       Code        |      Rule(R)      |      Rule(F)      |
| :-----: | :---------------: | :---------------: | :---------------: |
|   PC    |         7         |         3         |         1         |
| Android |        35         |        22         |        15         |
|   iOS   |        11         |         6         |         5         |
|  total  | 53 (48 confirmed) | 31 (28 confirmed) | 21 (17 confirmed) |

#### AutoTest FrameWork

![Image](AutoTestFramework.png)

