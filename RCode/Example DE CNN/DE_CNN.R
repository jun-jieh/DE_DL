###########################################
#### Load librarys, files, and objects ####
###########################################

# Load Keras library for Deep Learning
library(keras)
# Load gtool library to combine rows with different lengths (smartbind())
library(gtools)
# Load ggplot2 to generate plots
library(ggplot2)
# Load cluster library to compute the medoid
library(cluster)
# Turn off the warning. If you want to turn warnings back on, use options(warn=0)
options(warn=-1)
# Set working directory. The data should be ready in this path
setwd("~/")
# Load dataset into R environment
load("Genotype.RData")
# Load the response variable
load("Adj_ph.RData")
# Split dataset into training and validation sets. index will be used for training set.
# For instance, here the first 728 records will be used for training. 
# It's up to the user to decide the training data
index <- 1:728
# Load the genotype matrix
geno <- geno[index,]
# Define the input shape of MLP
shape <- ncol(geno)
# Prepare the data format for CNN (2D array)
geno <- array_reshape(geno, list(nrow(geno), ncol(geno), 1))
# Load the phenotypical values
y <- y[index]


##################################
##### Define hyperparameters #####
##################################

# The sample size of the data
Nobs <- nrow(geno)
# Commonly used optimizers available in deep learning
optimizers <- c('sgd','adam','adagrad','rmsprop','adadelta','adamax','nadam')
# Number of unique categories in these optimizers
a <- length(optimizers)
# Commonly used activation functions available in deep learning
activations <- c('relu','elu','sigmoid','selu','softplus','linear','tanh')
# Number of unique categories in these activation functions
b <- length(activations)
# Kernel size
filter_size <- as.integer(2:20)
c <- length(filter_size)
pools <- c('layer_max_pooling_1d(pool_size = 2)','layer_average_pooling_1d(pool_size = 2)')
e <- length(pools)
# Define the number of layers in neural network (depth of the network)
nlayers <- c(1,2,3,4,5)
# How many options available in the number of layers
f <- length(nlayers)
# Define the number of epoches (how many random sample & iterations in training set)
epochs <- as.integer(21:50)
g <- length(epochs)
# Dropout and L2 will reserve one position(column) in the vector

# Define the fully connected layer after flatten layer of CNN
layer_mlp <- as.integer(4:512)
# The # options in fully connected layer
h <- length(layer_mlp)

# The number of filters as a hyperparameter
nfilters <- as.integer(4:128)
# Maximum value of #Filter
up_limit <- 128
# Minimum value of #Filter
low_limit <- 4
# The #options for #Filters
d <- length(nfilters)
# Just count a+b+c as a flag
count=a+b+c


############################################
###### Initialize a random population ######
############################################

# 50 candidate solution in the population. We'll choose best one in the end
popsize=50
# Count unique markers of hyperparameters
varsize <- a+b+c+d+e+f+g+h+3
# Randomly sample values from (0,1)
init <- runif(n = popsize*varsize,min=0,max = 1) 
# Generate matrix to evolve based on numeric values
pop=matrix(init,popsize,varsize)


#######################################
####  Custom metrics to monitor DL ####
#######################################

# Connect to the backend and create a custom metric
K <- backend()
# metric_cor() is a measurement of predicted y and observed y
metric_cor <- function(y_true, y_pred) {
  n <- K$sum(K$ones_like(y_true))
  sum_x <- K$sum(y_true)
  sum_y <- K$sum(y_pred)
  sum_x_sq <- K$sum(K$square(y_true))
  sum_y_sq <- K$sum(K$square(y_pred))
  psum <- K$sum(y_true * y_pred)
  num <- psum - (sum_x * sum_y / n)
  den <- K$sqrt((sum_x_sq - K$square(sum_x) / n) *  (sum_y_sq - K$square(sum_y) / n))
  
  num / den
}


#####################################
###### Set up parameters for DE######
#####################################

numgen=10000  # number of iterations
CR=0.5        # probability of crossover
mu=0.5        # mutation rate

fit=numeric(popsize)  # Fitness of individuals in the population
info_DL <- numeric(length = popsize) # This is to save the fitness of the population

# Compute the fitness of individuals at baseline
for (i in 1:popsize)
{
  HP <- get_hp(pop[i,])
  if(i==1)
  {
    initial_pop <- HP
  }
  
  if(i>1)
  {initial_pop <- smartbind(initial_pop,HP)}
  # Each candidate has a score
  fit_initial <- try(fitness_CNN(HP))
  if('try-error' %in% class(fit_initial))
  {
    info_DL[i] <- -1
    k_clear_session()
    gc(reset=TRUE)
    next
  }else{
    initial_pop[i,]$epoch <- fit_initial[[2]]
    info_DL[i] <- as.numeric(fit_initial[[1]])
    k_clear_session()
    gc(reset=TRUE)
  }
  
}


fit <- info_DL
initial_info <- info_DL
save(initial_info,pop,file="initial_info_Real_CNN.RData")

# Create a event log to moniter every single iteration for later analyses
event_log <- as.data.frame(matrix(data = NA,nrow = numgen,ncol = 36))
colnames(event_log) <- c('Target Fitness','optimizer','activation','nlayers', 'filter_size',
                         'pool_layer', 'epoch', 'dropout_rate','L2','bsize','layer_mlp',
                         'filters_layer1','filters_layer2','filters_layer3','filters_layer4','filters_layer5',
                         'Challenger Fitness','c_optimizer','c_activation','c_nlayers', 'c_filter_size',
                         'c_pool_layer', 'c_epoch', 'c_dropout_rate','c_L2','c_bsize','c_layer_mlp',
                         'c_filters_layer1','c_filters_layer2','c_filters_layer3','c_filters_layer4',
                         'c_filters_layer5','c_succeed','num_iter','train_time','mean_fit')

# This column indicates if the candidate is replaced by challenger
event_log[,33] <- FALSE

# Monitoring mesurement including correlation and sd of correlation among population 
moniter_fit <- NULL
sd_fit <- NULL


#########################################
###### Now we run the DE algorithm ######
#########################################

for (i in 1:numgen)
{
  Time <- Sys.time()
  # choose 4 from population - 1 title holder, 2 template, 3 and 4 mutators
  index=sample(popsize,4)
  # true or false for CR - generate random numbers for CR and then test
  crtf=CR>runif(varsize)
  # index of true CR 
  crtf=which(crtf==T) 
  
  challenged <- pop[index[1],]
  HP_0 <- get_hp(challenged)
  HP_0$bsize <- 32
  event_log$num_iter[i] <- i
  event_log[i,2:(2+length(HP_0)-1)] <- HP_0
  save(event_log,file="event_log_Real_CNN.RData")
  
  # make a challenger from title holder
  challenger=pop[index[1],]
  # modify challenger
  challenger[crtf]=pop[index[2],crtf]-mu*(pop[index[3],crtf]-pop[index[4],crtf])
  # get fitness for challenger
  HP <- get_hp(challenger)
  HP$bsize <- 32
  event_log[i,18:(18+length(HP)-1)] <- HP
  save(event_log,file="event_log_Real_CNN.RData")
  
  fit_0 <- try(fitness_CNN(HP_0))
  
  if('try-error' %in% class(fit_0))
  {
    pop[index[1],]=challenger
    fit[index[1]] <- -1
    k_clear_session()
    gc(reset=TRUE)
    next
  }else{
    HP_0$epoch <- fit_0[[2]]
    fit[index[1]] <- as.numeric(fit_0[[1]])
    k_clear_session()
    gc(reset=TRUE)
  }
  event_log[i,1] <- fit[index[1]]
  event_log[i,2:(2+length(HP_0)-1)] <- HP_0
  save(event_log,file="event_log_Real_CNN.RData")
  
  fit_1 <- try(fitness_CNN(HP))
  
  if('try-error' %in% class(fit_1))
  {
    fitchal <- -1
    k_clear_session()
    gc(reset=TRUE)
    next
  }else{
    HP$epoch <- fit_1[[2]]
    fitchal <- as.numeric(fit_1[[1]])
    k_clear_session()
    gc(reset=TRUE)
  }
  
  event_log[i,17] <- fitchal
  event_log[i,18:(18+length(HP)-1)] <- HP
  event_log$train_time[i] <- round(Sys.time()-Time,digits = 3)
  save(event_log,file="event_log_Real_CNN.RData")
  fit <- as.numeric(unlist(fit))
  
  # if fitness of challenger better than fitness of title holder, replace.
  if(!is.na(fitchal)&!is.na(fit[index[1]])&fitchal>fit[index[1]]) 
  {
    pop[index[1],]=challenger
    fit[index[1]]=fitchal
    event_log$c_succeed[i] <- TRUE
  }
  event_log[i,36] <- mean(fit,na.rm = T)
  
  if(i%%10==0)
  {
    moniter_fit <- c(moniter_fit,mean(fit,na.rm = T))
    sd_fit <- c(sd_fit,sd(fit,na.rm = T))
  }
  
  
  if (i%%10==0){
    for(j in 1:nrow(pop))
    {
      EP <- get_hp(pop[j,])
      if(j==1)
      { evolve_pop <- EP}
      if(j>1)
      { evolve_pop <- smartbind(evolve_pop,EP)}
      
    }
    
    save(evolve_pop,pop,info_DL,fit,event_log,moniter_fit,sd_fit,
         file = "DE_Real_CNN.RData")
  }
  
}  



