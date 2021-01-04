DE for MLP
==========

This is an R Markdown document and we show the toy example for differential evolution (DE) against multilayer perceptron (MLP) here. For more information see <https://github.com/jun-jieh/DE_DL>.

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
index = 1:1000
# Define the validation set here
genov = geno[-index,]
yv = y[-index]
# Define the training set
geno = geno[index,]
y = y[index]
```

### Dimension of the genotype matrix for the training set

``` r
dim(geno)
```

    ## [1] 1000 1100

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

``` r
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

`corvalid` is a customized metric in TensorFlow, which can be used for monitering the model fitting process. In this study, we define the 'corvalid' as the correlation of the predicted and actual (simulated) phenotype. This value is calculated in each epoch, where the internal validation set is ramdomly selected.

`fitness` is a function to fit DL model and evaluate its predictive performance. The input of `fitness` function is a hyperparameter set from the defined space. If 'store'= T, the MLP model will be saved in the disk.

`makesol` is a function to generate a hyperparameter solution to MLP. The input of this function is the 'constrants' array defined in the last section.

`challenge` is a function to create challengers for the next generation (each title holder will have a corresponding challenger).

`rungen` is a function to execute differential evolution. The input of this function is simply the number of generations.

``` r
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

    ##          [,1]     [,2]      [,3]       [,4]      [,5]
    ## [1,] 2.488452 41.24701 0.6415258 0.06286062 19.863928
    ## [2,] 1.596965 45.78305 0.1885239 0.33452888 14.879673
    ## [3,] 2.299722 49.55115 0.5136956 0.28085139  8.397406
    ## [4,] 1.178275 49.36193 0.6394086 0.84814818 11.258803
    ## [5,] 4.466154 39.84223 0.4502070 0.18104647  8.406804
    ## [6,] 4.833366 47.65153 0.6117084 0.39326018  6.829873
    ## [7,] 3.716013 22.23475 0.5701279 0.82316887 10.688471
    ## [8,] 1.062503 36.18117 0.3027687 0.13175703  6.538539

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

    ##  [1]  0.34279859  0.30845135 -1.00000000  0.29607496  0.29458606  0.28424267
    ##  [7]  0.01602922  0.34357938  0.33591715 -0.08590578

### Best solution in the initial population

``` r
cat(paste("initial population:",round(max(fit),3),"\n"))
```

    ## initial population: 0.344

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

![](ToyExample-MLP_files/figure-markdown_github/unnamed-chunk-14-1.png)

Results
-------

### In this section, we will save the best MLP model to disk, evaluate its predictive performance, and print the evolved population with hyperparameter solutions.

``` r
index=which(fit==max(fit))[1]
cors=try(fitness(pop[index,],store=TRUE)) # save the final best model to disk
model_mlp = load_model_hdf5("finalModel.h5",custom_objects = c(corvalid = corvalid)) # load model from disk
```

### Use the model to predict the training set

``` r
pred=as.vector(predict(model_mlp, geno)) # training fit
cor(pred,y)
```

    ## [1] 0.7729029

### Use the model to predict the validation set

``` r
pred=as.vector(predict(model_mlp, genov)) # testing fit
cat(paste("accuracy in validation dataset:",round(cor(pred,yv),3),"\n"))
```

    ## accuracy in validation dataset: 0.349

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

    ## # A tibble: 10 x 15
    ##    fit   optimizer activation nlayers epochs dropout L2    batch nneuron dneuron
    ##    <chr> <chr>     <chr>      <chr>   <chr>  <chr>   <chr> <chr> <chr>   <chr>  
    ##  1 0.35~ rmsprop   softplus   1       50     0       0.06~ 20    512     0.1962~
    ##  2 0.34~ adagrad   softplus   4       38     0.3064~ 0.02~ 20    512     0.1    
    ##  3 0.37~ nadam     linear     1       38     0.0109~ 0.28~ 12    173     0.1    
    ##  4 0.37~ adamax    linear     4       50     0       0     20    218     0.6257~
    ##  5 0.36~ adamax    linear     2       50     0       0.10~ 20    512     0.2590~
    ##  6 0.28~ adagrad   softplus   5       48     0.6117~ 0.39~ 7     359     0.1232~
    ##  7 0.27~ rmsprop   sigmoid    3       26     0.5701~ 0.04~ 14    144     0.4125~
    ##  8 0.34~ nadam     linear     1       36     0.3027~ 0.13~ 7     239     0.1467~
    ##  9 0.36~ adam      softplus   5       48     0.0219~ 0.00~ 20    283     0.3391~
    ## 10 0.34~ adamax    linear     1       50     0       0.06~ 12    390     0.8332~
    ## # ... with 5 more variables: layer1 <chr>, layer2 <chr>, layer3 <chr>,
    ## #   layer4 <chr>, layer5 <chr>
