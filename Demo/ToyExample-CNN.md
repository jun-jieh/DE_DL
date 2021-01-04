DE for CNN
==========

This is an R Markdown document and we show the toy example for differential evolution (DE) against convolutional neural network (CNN) here. For more information see <https://github.com/jun-jieh/DE_DL>.

Load libraries
--------------

``` r
# Load tensorflow library. We assume this library was successfully installed.
library(tensorflow)
# Load keras library. We assume this library was successfully installed.
library(keras)
# Load libraries to display some tables
library(tidyverse)
library(dplyr)
```

Set up environment and load dataset
-----------------------------------

``` r
# Name the selected model
finalModel="finalModel"
# Set working directory. The data should be ready in this path
setwd("D:/DE_DL/DE Demo/")
# Read the genotype data
geno=readRDS("genotypes.rds")
# Transpose the genotype matrix
geno=t(geno)
# Read the simulated phenotype
pheno=read.table("SimPheno100QTL.txt",header=T,sep="\t")
# The phenotype is used as the response
y=pheno$y
allacc=NULL # store accuracy for all calls - just for testing/eval purposes

# subset into training/testing set
# subset into training/testing set
index = 1:1000
genov = geno[-index,]
genov <- array_reshape(genov, list(nrow(genov), ncol(genov), 1))
yv = y[-index]
geno = geno[index,]
geno <- array_reshape(geno, list(nrow(geno), ncol(geno), 1))
y = y[index]
```

### Dimension of the genotype matrix for the training set

``` r
dim(geno)
```

    ## [1] 1000 1100    1

### Summary of the response variable in the training set

``` r
summary(y)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -26.1519  -4.3652  -0.2698  -0.2582   4.0539  18.5657

Hyperparameter space and constraints
------------------------------------

### We define the hyperparameter space here. It is up to the users to specify the preferred hyperparameter space. Also, feel free to add/delete the unwanted hyperparameters. Note: if a modified hyperparameter space is used, be sure to reflect the changes in `fitness` and `makesol` functions.

``` r
# Define the input shape of CNN
shape <- dim(geno)[2]
nobs = dim(geno)[1]
# The input shape of CNN
geno <- array_reshape(geno, list(nrow(geno), ncol(geno), 1))

optimizers = c('sgd','adam','adagrad','rmsprop','adadelta','adamax','nadam')
activations = c('relu','elu','sigmoid','selu','softplus','linear','tanh')
pools <- c('layer_max_pooling_1d(pool_size = 2)','layer_average_pooling_1d(pool_size = 2)')

nlayers = c(1,2) # fixed number of convolutional layers
epochs = c(21,50) # min/max number of epochs (how many internal random samples and iterations in training set)
dropout=c(0,1) # min/max dropout rate
L2=c(0,1) # min/max kernel regularizer
fSize <- c(2,20) # min/max filter size
batch=32 # fixed batch size 
layerMLP <- c(4,512)

nfilter = c(4,128) # min/max number of filters in the neural network (width of the network)
dfilter =c(1.1,1.9) # min/max rate of increase in neuron number from one layer to the next (defines number of filters in each layer)

constraints=rbind(nlayers,epochs,dropout,L2,fSize,nfilter,dfilter,layerMLP)
colnames(constraints)=c("min","max")
rownames(constraints)=c("nlayers","epochs","dropout","L2","fSize","nfilter","dfilter","layerMLP")
```

### Show the hyperparameters with numeric (continuous) values and their min/max values

``` r
constraints
```

    ##           min   max
    ## nlayers   1.0   2.0
    ## epochs   21.0  50.0
    ## dropout   0.0   1.0
    ## L2        0.0   1.0
    ## fSize     2.0  20.0
    ## nfilter   4.0 128.0
    ## dfilter   1.1   1.9
    ## layerMLP  4.0 512.0

Functions
---------

`corvalid` is a customized metric in TensorFlow, which can be used for monitering the model fitting process. In this study, we define the 'corvalid' as the correlation of the predicted and actual (simulated) phenotype. This value is calculated in each epoch, where the internal validation set is ramdomly selected.

`get_k_size` is a function to adapt the filter size so that the CNN will generate a valid result.

`fitness` is a function to fit DL model and evaluate its predictive performance. The input of `fitness` function is a hyperparameter set from the defined space. If 'store'= T, the MLP model will be saved in the disk.

`makesol` is a function to generate a hyperparameter solution to MLP. The input of this function is the 'constrants' array defined in the last section.

`challenge` is a function to create challengers for the next generation (each title holder will have a corresponding challenger).

`rungen` is a function to execute differential evolution. The input of this function is simply the number of generations.

``` r
corvalid = custom_metric("corvalid", function(y_true, y_pred) # correlation of internal test data
{
  n = K$sum(K$ones_like(y_true))
  sum_x = K$sum(y_true)
  sum_y = K$sum(y_pred)
  sum_x_sq = K$sum(K$square(y_true))
  sum_y_sq = K$sum(K$square(y_pred))
  psum = K$sum(y_true * y_pred)
  num = psum - (sum_x * sum_y / n)
  den = K$sqrt((sum_x_sq - K$square(sum_x) / n) *  (sum_y_sq - K$square(sum_y) / n))
  num / den
}) 

# A funciton for adaptive filter sizes for each layer
get_k_size <- function(shape, nlayers, filter_size)
{
  
  # Creat a vector for CNN layers and Pooling layers by order, restricted by
  # input data shape, number of layers, and kernel size
  layer_actual <- numeric(length = 2*nlayers)
  # Creat a vector for minimum length for each CNN and Pooling
  layer_limit <- numeric(length = 2*nlayers)
  # Fill in the elements. Odd numbers represent CNN layers and even numbers for Pooling
  for(i in 1:nlayers)
  {
    layer_actual[2*(i-1)+1] <- as.integer(shape/((filter_size^i)*(2^(i-1))))
    layer_actual[2*i] <- as.integer(shape/((filter_size^(i))*(2^i)))
    layer_limit[2*(i-1)+1] <- as.integer(4^(nlayers+1-i))
    layer_limit[2*i] <- as.integer(2*4^(nlayers-i))
  }
  
  # Return a vector with the length respective to number of layers
  kernel_size <- rep.int(filter_size,nlayers)
  # Check from which layer we have an error that minimum kernels are not satisfied
  layer_error <- sort(which(layer_actual<layer_limit))[1]
  # If it's an odd index for layer, it happens in a CNN layer
  if(!is.na(layer_error) & layer_error %% 2 == 1)
  {
    cnn_index <- (layer_error+1)/2
    for(i in 1:(cnn_index-1)) { kernel_size[i] <- filter_size}
    kernel_size[cnn_index] <- floor(layer_actual[layer_error-2]/layer_limit[layer_error-1])
    if((cnn_index+1)<nlayers|(cnn_index+1)==nlayers)
    {for(i in (cnn_index+1):nlayers) {kernel_size[i] <- 2}}
  }
  # Otherwise, it happens in a pooling layer
  if(!is.na(layer_error) & layer_error %% 2 == 0)
  {
    cnn_index <- layer_error/2
    for(i in 1:(cnn_index-1)) { kernel_size[i] <- filter_size}
    kernel_size[cnn_index] <- floor(layer_actual[layer_error-1]/layer_limit[layer_error])
    if((cnn_index+1)<nlayers|(cnn_index+1)==nlayers)
    {for(i in (cnn_index+1):nlayers) {kernel_size[i] <- 2}}
  }  
  
  return(kernel_size)
}

fitness = function(pars,store=F) # gets vector of hyperparameters, builds and runs DL model, returns correlation vector of epochs
{
  indexop=(nrow(constraints)+1):((nrow(constraints))+length(optimizers))
  indexac=(max(indexop)+1):((max(indexop)+length(activations)))
  indexpool=(max(indexop)+length(activations)+1):((max(indexop)+length(activations))+length(pools))
  
  # get values from candidate solution
  opti=optimizers[sort(pars[indexop],index.return=T,decreasing=T)$ix[1]]
  activ=activations[sort(pars[indexac],index.return=T,decreasing=T)$ix[1]]
  pooling=pools[sort(pars[indexpool],index.return=T,decreasing=T)$ix[1]]
  
  hp=numeric(nrow(constraints))
  for (i in 1:nrow(constraints)) 
  {
    hp[i]=pars[i]
    if(hp[i]<constraints[i,1]) hp[i]=constraints[i,1] # apply constraint - min
    if(hp[i]>constraints[i,2]) hp[i]=constraints[i,2] # apply constraint - max
  }
  hp[c(1,2,5,6,8)]=round(hp[c(1,2,5,6,8)]) # round HP that need integers
  names(hp)=rownames(constraints)
  
  fSize=hp[5]
  filters=hp[6] # get filters per layer
  if (hp[1]>1) for (i in 2:hp[1]) filters=c(filters,round(filters[i-1]*hp[7])) 
  for (i in 1:length(filters)) # apply constraint on number of filters in each layer
  {
    if(filters[i]<constraints[6,1]) filters[i]=constraints[6,1] # min
    if(filters[i]>constraints[6,2]) filters[i]=constraints[6,2] # max
  }
  
  kernel_size <- get_k_size(shape=shape, nlayers=hp[1], filter_size=hp[5])
  
  if (hp[1]>1)
  {
        model=paste("model_cnn %>% layer_conv_1d(filters=",filters[1],",kernel_size=",kernel_size[1],
              ",activation='",activ,"', stride=",kernel_size[1],", input_shape=c(shape,1)) %>% layer_zero_padding_1d()               %>%                   ",pooling,sep="")
        for (i in 2:hp[1]){
              model=paste(model,"%>% layer_conv_1d(filters=",filters[i],",kernel_size=",kernel_size[i],
              ",activation='",activ,"', stride=",kernel_size[i],") %>% layer_zero_padding_1d() %>%                                           ",pooling,sep="")
        }
  }
  else
        model=paste("model_cnn %>% layer_conv_1d(filters=",filters[1],",kernel_size=",kernel_size,
              ",activation='",activ,"', stride=",fSize,", input_shape=c(shape,1)) %>% layer_zero_padding_1d() %>%                   ",pooling,sep="")
  
  # setup model as character vector

  model=paste(model," %>% layer_flatten() %>% ","layer_dropout(rate=",hp[3],") %>% ",
              "layer_dense(units=",hp[8],",activation='",activ,
              "',kernel_regularizer = regularizer_l2(l =",hp[4],"))"," %>% layer_dense(units = 1)",sep="")
  
  model_cnn = keras_model_sequential() # initialize
  eval(parse(text=model)) # build model
  
  model_cnn %>% compile(loss = 'mse',optimizer = opti, metrics = c(corvalid)) # compile
  
  # fit model
  run = model_cnn %>% fit(geno,y,epochs = hp[1],batch_size = batch,validation_split = 0.2,verbose=0,
                          callbacks = list(callback_early_stopping(monitor = 'val_corvalid', min_delta = 0.01, patience = 10, verbose = 0, mode = "max", baseline = NULL, restore_best_weights = TRUE)))
  if(store==T) save_model_hdf5(model_cnn,finalModel)
  cors=run$metric$val_corvalid # correlation over epochs for internal validation 
  
  pred=as.vector(predict(model_cnn, genov)) # testing fit
  allacc<<-c(allacc,cor(pred,yv)) # just for testing purposes   
  k_clear_session()
  return(cors)
}

makesol=function(constraints)
{
  pars=numeric(nrow(constraints))
  for (i in 1:nrow(constraints)) pars[i]=runif(1,constraints[i,1],constraints[i,2])
  pars=c(pars,runif(length(optimizers)),runif(length(activations)),runif(length(pools)))
  return(pars)
}

challenge=function(x) # create a challenger for the next generation
{
  index=matrix(unlist(lapply(1:popsize, function (x) sample(c(1:popsize)[-x],3))),popsize,3,byrow=T) # choose 4 from population - 1 title holder, 2 template, 3 and 4 mutators
  crtf=matrix(CR>runif(allelesize*popsize),popsize,allelesize) # true or false for CR - generate random numbers for CR and then test
  challenger=pop # make a challenger from title holder
  
  hold=pop[index[,1],]+FR*(pop[index[,2],]-pop[index[,3],])
  challenger[crtf]=hold[crtf]
  
  mut=1
  if (x%%5==0) mut=10 # crank up mutations every 5 gens
  mutate=runif(length(challenger))<noise
  
  challenger[mutate]=challenger[mutate]+(challenger[mutate]*runif(sum(mutate))-0.5)*mut
  challenger
}

# Call challenge function, calculate fitness of new solutions, evaluate against previous solutions
rungen=function(x) 
{
  challenger=challenge(x)
    
  fitchal=numeric(nrow(challenger))
  for (i in 1:length(fitchal))
  {  
    
    for (j in 1:nrow(constraints)) 
    {
        if(challenger[i,j]<constraints[j,1]) challenger[i,j]=constraints[j,1] # apply constraint - min
        if(challenger[i,j]>constraints[j,2]) challenger[i,j]=constraints[j,2] # apply constraint - max
    }
    challenger[i,c(1,2,5,6)]=round(challenger[i,c(1,2,5,6)]) # round HP that need integers
    
    cors=try(fitness(challenger[i,]))
    if('try-error' %in% class(cors)) fitchal[i]=-1
    else fitchal[i]=median(cors,na.rm=T)
    if(is.na(fitchal[i])==T | fitchal[i]==Inf | fitchal[i]==-Inf) fitchal[i]=-1
    print(fitchal[i])
  } 
  #fit=apply(pop,1,fitness) # recalculate fitness of current population
  index=which(fitchal>=fit)
  fit[index]=fitchal[index]
  pop[index,]=challenger[index,]
  assign("pop",pop, envir = .GlobalEnv)
  assign("fit",fit,envir=.GlobalEnv)
  assign("fit_hist",c(fit_hist,round(max(fit),3)),envir=.GlobalEnv)
  cat(paste("generation",x,":",round(max(fit),3),"\n"))
}

K = backend() # connect
```

Differential evolution
----------------------

### In this section, we define some parameters for DE e.g. population size, number of generations, and cross-over rate. Also, it's up to the users to set the proper values for DE parameters.

``` r
# DE parameters
popsize=10 # population size
numgen=5 # number of generations to iterate
CR=0.5 # probability of crossover
FR=0.5 # mutation rate
noise=0.3 # add random noise to crank up mutations occasionally

# Above values are used for a toy example. To evolve a reasonable population,
# consider the following parameter setup
#popsize=50 # population size
#numgen=500 # number of generations to iterate
#CR=0.5 # probability of crossover
#FR=0.5 # mutation rate
#noise=0.3 # add random noise to crank up mutations occasionally


# make initial population of solutions
pop=NULL
for (i in 1:popsize) pop=rbind(pop,makesol(constraints))
allelesize=ncol(pop) # number of parameters to optimize
```

### *pop* is a numeric matrix, where the rows are the individual solutions, and the columns represent the hyperparameter options/values.

``` r
pop[1:8,1:5]
```

    ##          [,1]     [,2]       [,3]        [,4]      [,5]
    ## [1,] 1.761609 32.30029 0.04165062 0.474724492  2.396861
    ## [2,] 1.965992 44.85667 0.01813760 0.680933105 19.275772
    ## [3,] 1.027920 24.69788 0.09137016 0.028275916 14.764121
    ## [4,] 1.023029 45.09783 0.99007558 0.006777912 14.408896
    ## [5,] 1.930154 44.40461 0.96061401 0.790546128 18.010609
    ## [6,] 1.647831 37.40090 0.11722288 0.073912338 14.594989
    ## [7,] 1.766471 44.23403 0.84576505 0.076909460 18.687608
    ## [8,] 1.557570 21.75314 0.80816630 0.249329502 16.969707

### Fit and compute the fitness of the randomly initialized population

``` r
# calculate fitness of initial population
fit=numeric(popsize)
for (i in 1:popsize)
{
    cors=try(fitness(pop[i,]))
    if('try-error' %in% class(cors)) fit[i]=-1
    else fit[i]=median(cors)
    if(is.na(fit[i])==T | fit[i]==Inf | fit[i]==-Inf) fit[i]=-1
}
```

### Each individual solution has its own fitness

``` r
fit
```

    ##  [1]  0.15394339  0.06655880  0.29718035  0.06418929 -0.01147416 -0.11926398
    ##  [7] -0.06059441 -0.05858093 -0.02996168  0.20857324

### Best solution in the initial population

``` r
cat(paste("initial population:",round(max(fit),3),"\n"))
```

    ## initial population: 0.297

### Run differential evolution. Make the plot for the evolution process. The dots represent the best solution of the population

``` r
# Save the max fit of all generations
fit_hist <- round(max(fit),3)
# Run DE
for (i in 1:numgen) rungen(i) 
```

### Plot the evolution history (max fitness of the population)

``` r
# Just a plot to see how fitness evolves - change boundaries depending on possible values
plot(fit_hist,xlim=c(0,numgen),ylim=c(0,1),xlab="generation",ylab="fitness",cex=0.8,col="blue",pch=20)
```

![](ToyExample-CNN_files/figure-markdown_github/unnamed-chunk-14-1.png) \#\# Results

### In this section, we will save the best MLP model to disk, evaluate its predictive performance, and print the evolved population with hyperparameter solutions.

``` r
index=which(fit==max(fit))[1]
cors=try(fitness(pop[index,],store=TRUE)) # save the final best model to disk
model_cnn = load_model_hdf5("finalModel",custom_objects = c(corvalid = corvalid)) # load model from disk
```

### Use the model to predict the training set

``` r
pred=as.vector(predict(model_cnn, geno)) # training fit
cor(pred,y)
```

    ## [1] 0.4459356

### Use the model to predict the validation set

``` r
pred=as.vector(predict(model_cnn, genov)) # testing fit
cat(paste("accuracy in validation dataset:",round(cor(pred,yv),3),"\n"))
```

    ## accuracy in validation dataset: 0.228

### Convert the numeric matrix *pop* to the actual hyperparameter solutions, and save them and their fitness to the disk.

``` r
sols=list()
size=0
for (sol in 1:nrow(pop))
{
    pars=pop[sol,]
    indexop=(nrow(constraints)+1):((nrow(constraints))+length(optimizers))
    indexac=(max(indexop)+1):((max(indexop)+length(activations)))

    # get values from candidate solution
    opti=optimizers[sort(pars[indexop],index.return=T,decreasing=T)$ix[1]]
    activ=activations[sort(pars[indexac],index.return=T,decreasing=T)$ix[1]]

    hp=numeric(nrow(constraints))
    for (i in 1:nrow(constraints)) 
    {
        hp[i]=pars[i]
        if(hp[i]<constraints[i,1]) hp[i]=constraints[i,1] # apply constraint - min
        if(hp[i]>constraints[i,2]) hp[i]=constraints[i,2] # apply constraint - max
    }
    hp[c(1,2,5,6)]=round(hp[c(1,2,5,6)]) # round HP that need integers
    names(hp)=rownames(constraints)

    neurons=hp[6] # get neurons per layer
    if (hp[1]>1) for (i in 2:hp[1]) neurons=c(neurons,round(neurons[i-1]*hp[7])) 
    for (i in 1:length(neurons)) # apply constraint on number of neurons in each layer
    {
        if(neurons[i]<constraints[6,1]) neurons[i]=constraints[6,1] # min
        if(neurons[i]>constraints[6,2]) neurons[i]=constraints[6,2] # max
    }
    sols[[sol]]=c(opti,activ,hp,neurons)
    if(length(sols[[sol]])>size) size=length(sols[[sol]])
}
out=matrix(NA,nrow(pop),size)
for (i in 1:nrow(pop)) out[i,1:length(sols[[i]])]=sols[[i]]
colnames(out)=c("optimizer","activation","nlayers","epochs","dropout","L2","batch","nneuron","dneuron",paste("layer",1:(size-9),sep=""))
out=cbind(fit,out)

save(out,pop,fit,file="Output_myDL.RData")
```

### Display the evolved population with hyperparameters and their fitness

``` r
as_tibble(out)
```

    ## # A tibble: 10 x 13
    ##    fit   optimizer activation nlayers epochs dropout L2    batch nneuron dneuron
    ##    <chr> <chr>     <chr>      <chr>   <chr>  <chr>   <chr> <chr> <chr>   <chr>  
    ##  1 0.27~ adagrad   elu        2       32     0.0416~ 0.16~ 11    102     1.1929~
    ##  2 0.29~ nadam     relu       2       22     0       0     20    128     1.9    
    ##  3 0.29~ nadam     elu        1       25     0.0913~ 0.02~ 15    53      1.8843~
    ##  4 0.35~ adam      elu        1       36     0.2062~ 0     10    74      1.1977~
    ##  5 0.19~ rmsprop   elu        1       25     0.4863~ 0.08~ 16    17      1.4861~
    ##  6 0.30~ adamax    selu       1       49     0.7082~ 0.21~ 4     103     1.9    
    ##  7 0.31~ nadam     elu        1       42     0.4540~ 0     19    43      1.3519~
    ##  8 0.22~ adamax    tanh       2       50     0       0     6     128     1.1794~
    ##  9 0.20~ adagrad   tanh       2       42     0.2314~ 0     8     80      1.5271~
    ## 10 0.23~ adam      linear     2       33     0.1967~ 0     2     78      1.1    
    ## # ... with 3 more variables: layer1 <chr>, layer2 <chr>, layer3 <chr>
