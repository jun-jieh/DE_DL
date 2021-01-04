MLP toy example {.title .toc-ignore}
===============

#### Junjie Han, Cedric Gondro, Kenneth Reid, and Juan Steibel {.author}

#### December 23, 2020 {.date}

DE for MLP
==========

This is an R Markdown document and we show the toy example for
differential evolution (DE) against multilayer perceptron (MLP) here.
For more information see
[https://github.com/jun-jieh/DE\_DL](https://github.com/jun-jieh/DE_DL).

Load libraries
--------------

``` {.r}
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

``` {.r}
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
index = 1:1000
# Define the validation set here
genov = geno[-index,]
yv = y[-index]
# Define the training set
geno = geno[index,]
y = y[index]
```

### Dimension of the genotype matrix for the training set

``` {.r}
dim(geno)
```

    ## [1] 1000 1100

### Summary of the response variable in the training set

``` {.r}
summary(y)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -26.1519  -4.3652  -0.2698  -0.2582   4.0539  18.5657

Hyperparameter space and constraints
------------------------------------

### We define the hyperparameter space here. It is up to the users to specify the preferred hyperparameter space. Also, feel free to add/delete the unwanted hyperparameters. Note: if a modified hyperparameter space is used, be sure to reflect the changes in `fitness` and `makesol` functions.

``` {.r}
shape = ncol(geno)
nobs = nrow(geno)

optimizers = c('sgd','adam','adagrad','rmsprop','adadelta','adamax','nadam')
activations = c('relu','elu','sigmoid','selu','softplus','linear','tanh')

nlayers = c(1,5) # min/max number of layers in neural network (depth of the network)
epochs = c(21,50) # min/max number of epochs (how many internal random samples and iterations in training set)
dropout=c(0,1) # min/max dropout rate
L2=c(0,1) # min/max kernel regularizer
batch=c(6,20) # min/max batch size 

nneuron = c(4,512) # min/max number of neurons in the neural network (width of the network)
dneuron =c(0.1,0.9) # min/max rate of decay in neuron number from one layer to the next (defines number of neurons in each layer)

# For non-categorical hyperparameters, we put constraints on them
constraints=rbind(nlayers,epochs,dropout,L2,batch,nneuron,dneuron)
colnames(constraints)=c("min","max")
rownames(constraints)=c("nlayers","epochs","dropout","L2","batch","nneuron","dneuron")
```

### Show the hyperparameters with numeric (continuous) values and their min/max values

``` {.r}
constraints
```

    ##          min   max
    ## nlayers  1.0   5.0
    ## epochs  21.0  50.0
    ## dropout  0.0   1.0
    ## L2       0.0   1.0
    ## batch    6.0  20.0
    ## nneuron  4.0 512.0
    ## dneuron  0.1   0.9

Functions
---------

`corvalid` is a customized metric in TensorFlow, which can be used for
monitering the model fitting process. In this study, we define the
'corvalid' as the correlation of the predicted and actual (simulated)
phenotype. This value is calculated in each epoch, where the internal
validation set is ramdomly selected.

`fitness` is a function to fit DL model and evaluate its predictive
performance. The input of `fitness` function is a hyperparameter set
from the defined space. If 'store'= T, the MLP model will be saved in
the disk.

`makesol` is a function to generate a hyperparameter solution to MLP.
The input of this function is the 'constrants' array defined in the last
section.

`challenge` is a function to create challengers for the next generation
(each title holder will have a corresponding challenger).

`rungen` is a function to execute differential evolution. The input of
this function is simply the number of generations.

``` {.r}
# Correlation of internal validation set
corvalid = custom_metric("corvalid", function(y_true, y_pred) 
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

# gets vector of hyperparameters, builds and runs DL model, returns correlation vector of epochs
fitness = function(pars,store=F) 
{
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

    # setup model as character vector
    model=paste("model_mlp %>% layer_dense(units=",neurons[1],", activation='",activ,"', input_shape=",shape,", kernel_regularizer=regularizer_l2(l=",hp[4],")) %>% layer_dropout(rate=",hp[3],") " ,sep="")
    addlayer=NULL # additional layers
    if (hp[1]>1) 
    {
        for (i in 2:hp[1]) addlayer=c(addlayer,paste("layer_dense(units=",neurons[i],", activation='",activ,"')",sep=""))
        addlayer=paste(addlayer,collapse=" %>% ")
        addlayer=paste("%>%",addlayer)
    }   
    model=paste(model,addlayer," %>% layer_dense(units = 1)",sep="")

    model_mlp = keras_model_sequential() # initialize
    eval(parse(text=model)) # build model
    
    model_mlp %>% compile(loss = 'mse',optimizer = opti, metrics = c(corvalid)) # compile

    # fit model
    run = model_mlp %>% fit(geno,y,epochs = hp[2],batch_size = hp[5],validation_split = 0.2,verbose=0,
        callbacks = list(callback_early_stopping(monitor = 'val_corvalid', min_delta = 0.01, patience = 10, verbose = 0, mode = "max", baseline = NULL, restore_best_weights = TRUE)))
    if(store==T) save_model_hdf5(model_mlp,filepath = "finalModel.h5")
    cors=run$metric$val_corvalid # correlation over epochs for internal validation 
    
    pred=as.vector(predict(model_mlp, genov)) # testing fit
    allacc<<-c(allacc,cor(pred,yv)) # just for testing purposes 
    k_clear_session()
    return(cors)
}

# Generate a hyperparameter solution
makesol=function(constraints)
{
    pars=numeric(nrow(constraints))
    for (i in 1:nrow(constraints)) pars[i]=runif(1,constraints[i,1],constraints[i,2])
    pars=c(pars,runif(length(optimizers)),runif(length(activations)))
    return(pars)
}

# Create a challenger for the next generation
challenge=function(x) 
{
  # choose 4 from population - 1 title holder, 2 template, 3 and 4 mutators
  index=matrix(unlist(lapply(1:popsize, function (x) sample(c(1:popsize)[-x],3))),popsize,3,byrow=T)
  # true or false for CR - generate random numbers for CR and then test
  crtf=matrix(CR>runif(allelesize*popsize),popsize,allelesize)
  # make a challenger from title holder
  challenger=pop 
  
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

# Connect to the TensorFlow backend (Python)
K = backend() 
```

Differential evolution (DE)
---------------------------

### In this section, we define some parameters for DE e.g. population size, number of generations, and cross-over rate. Also, it's up to the users to set the proper values for DE parameters.

``` {.r}
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

``` {.r}
pop[1:8,1:5]
```

    ##          [,1]     [,2]       [,3]      [,4]      [,5]
    ## [1,] 4.744978 29.32102 0.53252957 0.4752574  9.370750
    ## [2,] 2.957122 32.05607 0.43324368 0.5074603 18.181397
    ## [3,] 3.903120 33.70265 0.57077729 0.9125798 11.224832
    ## [4,] 3.816714 42.63256 0.05925843 0.3689464 15.316512
    ## [5,] 3.849996 21.88381 0.12277421 0.8926158 12.891539
    ## [6,] 1.422251 28.45584 0.78053757 0.4356041 13.700880
    ## [7,] 4.319811 36.28034 0.79980461 0.5092980 10.214003
    ## [8,] 2.325518 46.51608 0.70716436 0.8575749  9.193126

### Fit and compute the fitness of the randomly initialized population

``` {.r}
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

``` {.r}
fit
```

    ##  [1] -1.00000000  0.39459868  0.13238355 -1.00000000  0.08817151 -1.00000000
    ##  [7]  0.10110675  0.33783847 -1.00000000 -1.00000000

### Best solution in the initial population

``` {.r}
cat(paste("initial population:",round(max(fit),3),"\n"))
```

    ## initial population: 0.395

### Run differential evolution. Make the plot for the evolution process. The dots represent the best solution of the population

``` {.r}
# Save the max fit of all generations
fit_hist <- round(max(fit),3)
# Run DE
for (i in 1:numgen) rungen(i) 
```

### Plot the evolution history (max fitness of the population)

``` {.r}
# Just a plot to see how fitness evolves - change boundaries depending on possible values
plot(fit_hist,xlim=c(0,numgen),ylim=c(0,1),xlab="generation",ylab="fitness",cex=0.8,col="blue",pch=20)
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABUAAAAPACAMAAADDuCPrAAAA1VBMVEUAAAAAADoAAGYAAP8AOjoAOmYAOpAAZrY6AAA6OgA6Ojo6OmY6ZmY6ZpA6ZrY6kJA6kLY6kNtmAABmADpmOgBmOjpmZjpmZmZmZpBmkLZmkNtmtttmtv+QOgCQOjqQZgCQZjqQZmaQkDqQkJCQkLaQtraQttuQtv+Q2/+2ZgC2Zjq2kDq2kGa2tpC2tra2ttu225C227a229u22/+2///bkDrbkGbbtmbbtpDbtrbb25Db27bb29vb2//b/7bb////tmb/25D/27b/29v//7b//9v////1Mb+TAAAACXBIWXMAAB2HAAAdhwGP5fFlAAAgAElEQVR4nO3dfWMTV8KfYSkbwCFLWihZxQv0IW2hMUuUFuIFHpxiDNL3/0idkfwiWw7J/HSs0Zlc1x/BFvJEg+Xb83LmzGgOQGTU9wsAqJWAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCAkoAAhAQUICShASEABQgIKEBJQgJCAAoQEFCC02wEdARRTPlHFl1hQ3//awLAU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)

Results
-------

### In this section, we will save the best MLP model to disk, evaluate its predictive performance, and print the evolved population with hyperparameter solutions.

``` {.r}
index=which(fit==max(fit))[1]
cors=try(fitness(pop[index,],store=TRUE)) # save the final best model to disk
model_mlp = load_model_hdf5("finalModel.h5",custom_objects = c(corvalid = corvalid)) # load model from disk
```

### Use the model to predict the training set

``` {.r}
pred=as.vector(predict(model_mlp, geno)) # training fit
cor(pred,y)
```

    ## [1] 0.7628301

### Use the model to predict the validation set

``` {.r}
pred=as.vector(predict(model_mlp, genov)) # testing fit
cat(paste("accuracy in validation dataset:",round(cor(pred,yv),3),"\n"))
```

    ## accuracy in validation dataset: 0.367

### Convert the numeric matrix *pop* to the actual hyperparameter solutions, and save them and their fitness to the disk.

``` {.r}
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

``` {.r}
as_tibble(out)
```

    ## # A tibble: 10 x 15
    ##    fit   optimizer activation nlayers epochs dropout L2    batch nneuron dneuron
    ##    <chr> <chr>     <chr>      <chr>   <chr>  <chr>   <chr> <chr> <chr>   <chr>  
    ##  1 0.18~ nadam     selu       5       22     0.3054~ 0.47~ 7     407     0.2784~
    ##  2 0.39~ adagrad   linear     3       32     0.4332~ 0.50~ 18    493     0.7814~
    ##  3 0.31~ nadam     tanh       1       50     0.6608~ 0.99~ 20    102     0.1    
    ##  4 0.31~ adamax    relu       5       46     0.0592~ 0.50~ 8     481     0.8090~
    ##  5 0.38~ adamax    tanh       2       40     0.3861~ 0.72~ 20    122     0.9    
    ##  6 0.32~ nadam     linear     5       39     0       0.40~ 9     274     0.4184~
    ##  7 0.22~ sgd       tanh       2       50     0.2661~ 1     20    86      0.3253~
    ##  8 0.33~ adamax    linear     2       47     0.7071~ 0.85~ 9     389     0.8795~
    ##  9 0.37~ adagrad   relu       5       50     0       0     18    512     0.6411~
    ## 10 0.26~ nadam     softplus   1       50     0.8487~ 0.63~ 19    512     0.5590~
    ## # ... with 5 more variables: layer1 <chr>, layer2 <chr>, layer3 <chr>,
    ## #   layer4 <chr>, layer5 <chr>
