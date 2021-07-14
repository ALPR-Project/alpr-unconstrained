# ALPR in Unscontrained Scenarios

## Resultados
### benchmark 11/06
Resultados do modelo baseline

Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Carro (515) | 0.66 | 0.70 | 361 | 182 | 154
Moto (56) | 0.28 | 0.29 | 16 | 41 | 40

Resultados do modelo CEIA 
* Dropout 40%
* Trainset carros

Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Carro (515) | 0.62 | 0.98 | 505 | 309 | 10
Moto (56) | 0.21 | 0.84 | 47 | 177 | 9

Resultados do modelo CEIA  
* Dropout 40%
* Trainset motos

Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Carro (515) | 0.15 | 0.39 | 183 | 1078 | 332
Moto (56) | 0.14 | 0.62 | 35 | 211 | 21

### benchmark 18/06
Resultados do modelo CEIA 
* Dropout 40%
* Trainset carros
* Ajuste de proporção de placa (2.84, 3.07)
* Tamanho imagem (304,304)
* Proporção placa/amostra (0.1, 0.5)

Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Carro (640) | 0.47 | 0.50 | 318 (49,7%) | 360 | 322 (50,31%)


Resultados do modelo CEIA  
* Dropout 40%
* Trainset motos
* Ajuste de proporção de placa (1)
* Tamanho imagem (304,304)
* Proporção placa/amostra (0.1, 0.5)

Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Moto (97) | 0.37 | 0.51 | 49 (50,5%) | 83 | 48 (49,9%)

### benchmark 20/06
Resultados do modelo CEIA 
* Dropout 40%
* Trainset carros
* Ajuste de proporção de placa (2, 3.07)
* Tamanho imagem (208,208)
* Proporção placa/amostra (0.2, 1.)
* modelo: modelo-ceia-ft-dpout-c-208_final


Objeto | Precision | Recall | True Pos | False Pos | False Neg
------------ | --------- | -------- | --------- | ------------- | -------------
Carro (640) | 0.56 | 0.90 | 578 (90,3%) | 446 | 62 (9,68%)

### Treino 25/06
* modelo modelo-ceia-ft-dpout-m-208-5
* Trainset motos
* Ajuste de proporção de placa (0.7, 1.2)
* Tamanho imagem (208,208)
* Proporção placa/amostra (0.3, 0.7)
* 3.*logloss(non_obj_probs_true,non_obj_probs_pred,(b,h,w,1))

Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Moto (180) | 0.37 | 0.96 | 173 | 655 | 7 

### Treino 26/06
* alpr-br-model-fs-dpout-moto-char-1_backup
* Trainset motos com ao menos 4 caracteres 
* Ajuste de proporção de placa (0.7, 1.2)
* Tamanho imagem (208,208)
* Proporção placa/amostra (0.3, 0.7)


Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Moto (130) | 0.91 | 0.98 | 128 | 13 | 2
Moto (180) | 0.81 | 0.84 | 152 | 35 | 28

True positives:  0.997747 media  0.023183 std  | False positives 0.273685 media 0.395876 std

### Treino 01/07
* modelo-ceia-ft-dpout-c-208-chars-3_backup
* Trainset carros com ao menos 4 caracteres 
* Ajuste de proporção de placa (2.84, 3.07)
* Tamanho imagem (208,208)
* Proporção placa/amostra (0.3, 0.7)
* Dropout 0.8


Objeto | Precision | Recall | True Positive | False Positive | False Negative
------------ | --------- | ------------- | --------- | ------------- | -------------
Carro (640) | 0.45 | 0.86 | 550 | 661 | 90


True positives:  0.983463 media  0.063782 std  | False positives 0.749641 media 0.219385 std



### Tabela resumo melhor modelo 
Resultados do modelo CEIA 
* Dropout 40%
* Trainset carros
* Ajuste de proporção de placa (2, 3.07)
* Tamanho imagem (208,208)
* Proporção placa/amostra (0.2, 1.)
* modelo: modelo-ceia-ft-dpout-c-208_final

* Trainset motos
* Ajuste de proporção de placa (0.7, 1.2)
* Tamanho imagem (208,208)
* Proporção placa/amostra (0.2, 1.)
* modelo: modelo-ceia-ft-dpout-m-208-3_best
* threshold 0.4


Modelo | Objeto | Precision | Recall | True Pos | False Pos | False Neg
 ------| ------------ | --------- | ------------- | --------- | ------------- | -------------
Baseline | Carro (640) | 0.70 | 0.74 | 473 (74,3%) | 201 | 167 (26,1%)
CEIA | Carro (640) | 0.56 | 0.90 | 578 (90,3%) | 446 | 62 (9,68%)
Baseline | Moto (180) | 0.66 | 0.70 | 126 (70,0%) | 65 | 54 (30,0%)
CEIA | Moto (180) | 0.26 | 0.98 | 177 (98,3%) | 481 | 3 (1,66%)

whratio = random.uniform(2, 3.07)
wsiz = random.uniform(dim_w*.2,dim_w*1.)


### Dataset

Dataset | Train | Validation 
 ------| ------------ | --------- 
Carro | 2.801 | 640
Moto | 399 | 97


This repository contains the author's implementation of ECCV 2018 paper "License Plate Detection and Recognition in Unconstrained Scenarios".

* Paper webpage: http://sergiomsilva.com/pubs/alpr-unconstrained/

If you use results produced by our code in any publication, please cite our paper:

```
@INPROCEEDINGS{silva2018a,
  author={S. M. Silva and C. R. Jung}, 
  booktitle={2018 European Conference on Computer Vision (ECCV)}, 
  title={License Plate Detection and Recognition in Unconstrained Scenarios}, 
  year={2018}, 
  pages={580-596}, 
  doi={10.1007/978-3-030-01258-8_36}, 
  month={Sep},}
```

## Requirements

In order to easily run the code, you must have installed the Keras framework with TensorFlow backend. The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in "darknet" folder:

```shellscript
$ cd darknet && make
```

**The current version was tested in an Ubuntu 16.04 machine, with Keras 2.2.4, TensorFlow 1.5.0, OpenCV 2.4.9, NumPy 1.14 and Python 2.7.**

## Download Models

After building the Darknet framework, you must execute the "get-networks.sh" script. This will download all the trained models:

```shellscript
$ bash get-networks.sh
```

## Running a simple test

Use the script "run.sh" to run our ALPR approach. It requires 3 arguments:
* __Input directory (-i):__ should contain at least 1 image in JPG or PNG format;
* __Output directory (-o):__ during the recognition process, many temporary files will be generated inside this directory and erased in the end. The remaining files will be related to the automatic annotated image;
* __CSV file (-c):__ specify an output CSV file.

```shellscript
$ bash get-networks.sh && bash run.sh -i samples/test -o /tmp/output -c /tmp/output/results.csv
```

## Training the LP detector

To train the LP detector network from scratch, or fine-tuning it for new samples, you can use the train-detector.py script. In folder samples/train-detector there are 3 annotated samples which are used just for demonstration purposes. To correctly reproduce our experiments, this folder must be filled with all the annotations provided in the training set, and their respective images transferred from the original datasets.

The following command can be used to train the network from scratch considering the data inside the train-detector folder:

```shellscript
$ mkdir models
$ python create-model.py eccv models/eccv-model-scracth
$ python train-detector.py --model models/eccv-model-scracth --name my-trained-model --train-dir samples/train-detector --output-dir models/my-trained-model/ -op Adam -lr .001 -its 300000 -bs 64
```

For fine-tunning, use your model with --model option.

## A word on GPU and CPU

We know that not everyone has an NVIDIA card available, and sometimes it is cumbersome to properly configure CUDA. Thus, we opted to set the Darknet makefile to use CPU as default instead of GPU to favor an easy execution for most people instead of a fast performance. Therefore, the vehicle detection and OCR will be pretty slow. If you want to accelerate them, please edit the Darknet makefile variables to use GPU.
