# DE_DL
**Heuristic hyperparameter optimization of deep learning models for genomic prediction**

__Junjie Han, Cedric Gondro, Kenneth Reid & Juan P. Steibel__

If you find this resource helpful, please cite.


This repository explains multilayer perceptron (MLP) and convolutional neural network (CNN), and shows the code to implement differential evolution (DE) on top of MLPs and CNNs.

## Table of contents

* [Software](#Software)
* [MLP](#MLP)
* [CNN](#CNN)
* [Workflow](#Workflow)
* [Model_selection](#Model_selection)
* [Code](#Code)
* [Dataset](#Dataset)
* [Demo](#Demo)

## Software
* All experiments are implemented in [R](https://cloud.r-project.org/) (version 3.6.1)

* MLPs and CNNs are fitted using [Keras](https://keras.rstudio.com/) on top of [TensorFlow](https://tensorflow.rstudio.com/)

* GPU computing is available and is much faster compared to CPU. However, an [NVIDIA](https://developer.nvidia.com/cuda-gpus) graphic card is required for GPU computing.

## MLP
Typical MLP models consist of an input layer, a variable number of hidden layer(s), and an output layer. Each layer contains several neurons (also known as nodes). Depending on the type of layer, the nature of the nodes will change. For instance, the number of nodes in the input layer is equal to the number of predictor features. In this study, the input layer represents an individual’s genotype, and thus, the input layer will have as many nodes as SNP markers used for genomic prediction. In the following figure, there are M nodes in the input layer, and its *kth* node ![xnk](https://latex.codecogs.com/gif.latex?x_%7Bn%2Ck%7D) will receive input as the allelic count of the reference allele at the *kth* SNP for the nth individual. The output layer represents the prediction of the response variable produced by MLP. In this case, the output will contain the prediction of an individual’s phenotypic value ![yn](https://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D_n), which can be a continuous outcome, an ordinal outcome, and a categorical outcome. 

The nodes in one layer are connected to the nodes in the previous layer by a weighted sum operator. For instance, the input of the jth node in hidden layer 1 is

![z1j](https://latex.codecogs.com/gif.latex?z_j%5E1%3Df%28%5Csum_%7Bk%3D1%7D%5E%7BM%7Dw_%7Bjk%7D%5E%7B0%7D%20x_k%29) ,

where ![xk](https://latex.codecogs.com/gif.latex?x_k) represents the kth node from previous layer, the weights ![wjk0](https://latex.codecogs.com/gif.latex?w_%7Bjk%7D%5E0) are unknown and connected to ![xk](https://latex.codecogs.com/gif.latex?x_k) (kth SNP) and need to be determined through an learning process. *f()* is the activation function that is specified by the user. Noteworthy, non-linear functions can be used as *f()*, relaxing the linearity assumption of classic genomic prediction models. We discuss possible activation functions later in this paper. Likewise, nodes between layers are fully connected, which means that the input sum of each node in a layer will contain as many terms as nodes are in the previous layer:

![zj'ifz](https://latex.codecogs.com/gif.latex?z_%7Bj%27%7D%5Ei%3Df%28%5Csum_%7Bj%3D1%7D%5E%7Bnneuron_%7Bi-1%7D%7Dw_%7Bj%2Cj%27%7D%5E%7Bi-1%7Dz_j%5E%7Bi-1%7D%29) ,

where *j* represents the nodes from layer *i-1*, ![nneuroni-1](https://latex.codecogs.com/gif.latex?nneuron_%7Bi-1%7D) is the number of nodes in hidden layer *i-1*, *i* is the index of hidden layer *i*, ![wjj'](https://latex.codecogs.com/gif.latex?w_%7Bj%2Cj%27%7D%5E%7Bi-1%7D) is a weight connecting *jth* node (![zji-1](https://latex.codecogs.com/gif.latex?z_j%5E%7Bi-1%7D)) in hidden layer *i-1* and *j'th* node in hidden layer *i* ( ![zj'i](https://latex.codecogs.com/gif.latex?z_%7Bj%27%7D%5E%7Bi%7D) ).

![](https://github.com/jun-jieh/DE_DL/blob/master/Figures/MLP.png)

## CNN

In the context of genomic prediction, the input layer for a single observation in a CNN is a one-dimensional array Specifically, the input layer will contain an animal’s (nth individual's) genotype and the number of units will be equal to the number of SNP markers. In the following figure, there are *M* units in the input layer and the *kth* unit represents the allelic count of the reference allele at the *kth* SNP for the nth individual (![xnk](https://latex.codecogs.com/gif.latex?x_%7Bn%2Ck%7D)). The output layer represents the predicted response value ![yn](https://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D_n) for the phenotype or breeding value of the nth individual. After the input layer, a CNN contains a variable number of convolutional layers followed by pooling layers. For instance, in Convolutional Layer 1 of the figure, several filters are applied to the nodes of the input layer, where filters are arrays containing certain number of weights to convolve the input. In this case, each filter has three weights ![wi11](https://latex.codecogs.com/gif.latex?w_%7Bi%2C1%7D%5E1), ![wi21](https://latex.codecogs.com/gif.latex?w_%7Bi%2C2%7D%5E1), and ![wi31](https://latex.codecogs.com/gif.latex?w_%7Bi%2C3%7D%5E1), where *i* represents the *ith* filter defined by the user. These filters are applied to every three consecutive units of the input layer (filter size equal to three). Also, the stride of the filter is equal to its length, which means that the filter is applied to non-overlapping sets of three contiguous SNP. The length of the filter (kernel) is defined by the number of weights to include i.e. the number of units to be convolved by a filter in the input data. An arbitrary number of filters *i* =1…I  is applied in each convolution. The output of this process will be I feature maps with length equal to ![(M-F)/S+1](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7BM-F%7D%7BS%7D&plus;1), where M represents the number of SNP markers, F is the length of the filter, and S is the stride. In our case, because stride is equal to filter size, the length is simply M/3. Moreover, the input of the jth unit in feature map 1 is ![cji=](https://latex.codecogs.com/gif.latex?%5Cinline%20c_j%5E1%3Df%28w_%7Bi%2C1%7D%5E1x_%7Bn%2Ck%7D&plus;w_%7Bi%2C2%7D%5E1x_%7Bn%2Ck&plus;2%7D&plus;w_%7Bi%2C3%7D%5E1x_%7Bn%2Ck&plus;3%7D%29), where ![xnk](https://latex.codecogs.com/gif.latex?%5Cinline%20x_%7Bn%2Ck%7D), ![xnk+1](https://latex.codecogs.com/gif.latex?%5Cinline%20x_%7Bn%2Ck&plus;1%7D), and ![xnk+2](https://latex.codecogs.com/gif.latex?%5Cinline%20x_%7Bn%2Ck&plus;2%7D) are allelic dosages of individual *n* at three consecutive SNP markers. The weights in the filters are unknown and need to be determined through an DL optimization process. *f()* is the activation function. Convolutional Layer 1, the output of each convolution is saved in feature map 1, where the length of each feature map is ![a1](https://latex.codecogs.com/gif.latex?%5Cinline%20a_1%3D%5Cfrac%7BM%7D%7B3%7D) and the number of feature maps (b1) is equal to the number of filters (kernels) applied to the input layer (in this case b1=5 in the figure). A convolutional layer is followed by a pooling layer for the purposes of dimensionality reduction. In pooling layer 1, ![p1](https://latex.codecogs.com/gif.latex?%5Cinline%20p%5E1%3D%28p_1%5E1%2Cp_1%5E2%2C...%2Cp_%7Ba_1/2%7D%5E1%29) are elements that are summarized by every two consecutive units generated from the previous convolutional layer and the output will be *b1* feature maps with a reduced length equal to ![a1by2](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7Ba_1%7D%7B2%7D). Likewise, feature map 2 is followed by convolutional layer 2 where filters with three weights ![wi'12](https://latex.codecogs.com/gif.latex?%5Cinline%20w_%7Bi%27%2C1%7D%5E2), ![wi'22](https://latex.codecogs.com/gif.latex?%5Cinline%20w_%7Bi%27%2C2%7D%5E2), and ![wi'32](https://latex.codecogs.com/gif.latex?%5Cinline%20w_%7Bi%27%2C3%7D%5E2) are applied. In feature map 3, *b2* features with a length of *a2* are summarized into feature map 4 that has *b2* features with length ![a2by2](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7Ba_2%7D%7B2%7D). If any value among ![Mby3](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7BM%7D%7B3%7D), ![a1by2](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7Ba_1%7D%7B2%7D), ![a1by2by3](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7Ba_1/2%7D%7B3%7D), or ![a2by2](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7Ba_2%7D%7B2%7D) has a remainder, the deficit unit(s) in the input data will be padded with zero(s). The last feature map (feature map 4) is re-arranged into a single vector that has ![b2a2](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7Bb_2%5Ctimes%20a_2%7D%7B2%7D) elements. Each element in the re-arranged vector ![z1](https://latex.codecogs.com/gif.latex?%5Cinline%20z%5E1%3D%28z_1%5E1%2Cz_2%5E1%2C...%2Cz_l%5E1%2Cl%3D%5Cfrac%7Bb_2%5Ctimes%20a_2%7D%7B2%7D%29) is fully connected to a hidden layer (like the ones described in the MLP section of this paper) with nneuron nodes, which are predictors for the output layer.

![](https://github.com/jun-jieh/DE_DL/blob/master/Figures/CNN.png)

## Workflow

The overall workflow is as shown in the following figure:

![](https://github.com/jun-jieh/DE_DL/blob/master/Figures/Workflow.png)

## Model_selection

Given an optimized population (composed of 50 solutions), each individual in the population was refitted for 30 times and this process was employed after DE. We evaluated model stability through repeated training of each hyperparameter (HP) solution. The selection criteria of top model within each population were based on mean fitness and standard deviation (SD) of the fitness obtained by refitting each model 30 times. To select the best model from each population, we first defined medoid of the population given the mean fitness and SDs. Then we computed Euclidean distances Di between each individual i and the medoid given their mean fitness and SDs. The best model satisfied three conditions: 1) the mean fitness of individual i was larger than the medoid’s, and 2) the SD of fitness of individual i was smaller than the medoid’s , and 3) Di was the largest that satisfies 1) and 2) simultaneously. The following figure shows one of the examples for selecting a top model within a population.

![](https://github.com/jun-jieh/DE_DL/blob/master/Figures/Medoid.png)

## R packages used in the code

* [keras](https://keras.rstudio.com/) and [tensorflow](https://tensorflow.rstudio.com/) (Keras will load TensorFlow by default)

* [gtools](https://cran.r-project.org/web/packages/gtools/index.html)

* [ggplot2](https://ggplot2.tidyverse.org/)

* [cluster](https://cran.r-project.org/web/packages/cluster/index.html)

## Code

* R code can be found in the __RCode__ folder

* __Example DE MLP__ is for the DE on top of MLPs and __Example DE CNN__ is for the DE on top of CNNs

* Notice there are multiple functions included in the code:

`gen_neurons` creates adaptive number of neurons according to number of fully connected layers

`gen_filters` creates adaptive number of neurons according to number of convolutional layers

`get_hp` gets hyperparameters from a population. Note: the input is a numeric vector and this function differs in MLPs and CNNs

`metric_cor` customizes a correlation metrices for monitoring the DL model performance

`fitness_MLP` fits an MLP model and returns the fitness given a hyperparameter set

`fitness_CNN` fits a CNN model and returns the fitness given a hyperparameter set

`get_k_size` is a funciton for adaptive filter sizes for each convolutional layer

`build_cnn` generates a CNN architecture

`max_Euc_dist` A function for evaluating and selecting best model given stats of repeated training


## Dateset

To find the dataset used in this study, please refer to this repository: https://github.com/jun-jieh/RealPigData

## Demo

Some users may choose to use Keras with its CPU version. If that is the case, it will cost considerable amount of time to implement the DE on top of DL with the pig dataset provided. Therefore, we developed a simplified version of the DE code and a demo dataset with a much smaller problem size to illustrate the DE on top of DL. The demo codem dataset, and the toy examples (one for MLP and one for CNN) are included in the __Demo__ folder. However, the full program for implementing the exact same procedure mentioned in the paper can still be found in the __Code__ folder.

