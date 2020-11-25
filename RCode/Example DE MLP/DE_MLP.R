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
# Suppress the warning. If you want to turn warnings back on, use options(warn=0)
options(warn=-1)
# Set working directory here. The data should be ready in this path
setwd("~/")
# Load the dataset into R environment 
load("Genotype.RData")
# Load the response variable
load("Adj_ph.RData")
# Split dataset into training and validation sets. index will be used for training set.
# For instance, here the first 728 records will be used for training. 
# It's up to the user to decide the training data
index <- 1:728
# Load the genotype matrix. 'geno' matrix is for training dataset
geno <- geno[index,]
# Define the input shape of MLP regarding the number of SNP markers
shape <- ncol(geno)
# Load the phenotypes for training set
y <- y[index]

##################################
##### Define hyperparameters #####
##################################


# The sample size (number of observations) of the data
Nobs <- nrow(geno)
# Commonly used optimizers available in deep learning
optimizers <- c('sgd','adam','adagrad','rmsprop','adadelta','adamax','nadam')
# Number of unique categories in these optimizers
a <- length(optimizers)
# Commonly used activation functions available in deep learning
activations <- c('relu','elu','sigmoid','selu','softplus','linear','tanh')
# Number of unique categories in these activation functions
b <- length(activations)
# Define the possible number of layers in neural network (depth of the network)
nlayers <- c(1,2,3,4,5)
# The number of options available in the number of layers
c <- length(nlayers)
# Define the number of epoches (how many internal random sample & iterations in training set)
epochs <- as.integer(21:50)
# The number of options available for epochs
f <- length(epochs)
# Define the number neurons in neural network (width of the network)
# The number of neurons can be any integer between 4 and 512
nneuron <- as.integer(4:512)
# The maximum value for nneuron
up_limit <- 512
# The minimum value for nneuron
low_limit <- 4
# The number of options available for number of neurons
d <- length(nneuron)
# Count how many possible options for activation, optimizer, # layers and # neurons
# It will be for later use
count=a+b+c+d

### Note: it is up to users to define the hyperparameter space ###


############################################
###### Initialize a random population ######
############################################

# 50 candidate solution in the population.
popsize=50 
# Count unique markers of hyperparameters. The 3 at the end represents L2, dropout and batch proportion
varsize <- a+b+c+d+f+3   
# Randomly sample values from (0,1)
init <- runif(n = popsize*varsize,min=0,max = 1) 
# Generate the population matrix to evolve based on numeric values
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

numgen=10000    # number of generations in DE
CR=0.5          # probability of crossover
mu=0.5          # mutation rate

fit=numeric(popsize)  # Fitness of individuals in the population
info_DL <- numeric(length = popsize) # This is to save the fitness of the population

# Compute the fitness of individuals at baseline
for (i in 1:popsize)
{
  # Get the hyperparameter set for individual i
  HP <- get_hp(pop[i,])
  # Start from the first one
  if(i==1)
  {
    initial_pop <- HP
  }
  
  # Keep adding records after first one
  if(i>1)
  {initial_pop <- smartbind(initial_pop,HP)}
  # Each candidate has a score. Use try function in case
  # we encounter some error
  fit_initial <- try(fitness_MLP(HP$optimizer,HP$activation,HP$nlayers,HP$epoch,
                                 HP$dropout_rate,HP$L2,HP$bsize,HP$neuron_layer1,HP$neuron_layer2,
                                 HP$neuron_layer3,HP$neuron_layer4,HP$neuron_layer5))
  
  # Assign value to info_DL[i] regarding if there was error
  if('try-error' %in% class(fit_initial))
  {
    # If there is error, penalize the hyperparameter
    info_DL[i] = -1
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

# Save the information for initial population
fit <- info_DL
initial_info <- info_DL
save(initial_info,pop,file="initial_info_Real_MLP.RData")

# Create a event log to moniter every single iteration for later analyses
event_log <- as.data.frame(matrix(data = NA,nrow = numgen,ncol = 30))
colnames(event_log) <- c('Target Fitness','optimizer','activation','nlayers','epoch',
                         'dropout_rate','L2','bsize','neuron_layer1',
                         'neuron_layer2','neuron_layer3','neuron_layer4','neuron_layer5',
                         'Challenger Fitness','c_optimizer','c_activation','c_nlayers',
                         'c_epoch','c_dropout_rate','c_L2','c_bsize',
                         'c_neuron_layer1','c_neuron_layer2','c_neuron_layer3',
                         'c_neuron_layer4','c_neuron_layer5','c_succeed','num_iter','train_time','mean_fit')

# This column indicates if the candidate is replaced by challenger
event_log[,27] <- FALSE

# Monitoring mesurement including correlation and sd of correlation among population 
moniter_fit <- NULL
sd_fit <- NULL

#########################################
###### Now we run the DE algorithm ######
#########################################


# Run numgen iterations and break certain iteration if necessary
for (i in 1:numgen)
{
  # Profile the start time of this iteration
  Time <- Sys.time()
  # choose 4 from population - 1 title holder, 2 template, 3 and 4 mutators
  index=sample(popsize,4)
  # true or false for CR - generate random numbers for CR and then test
  crtf=CR>runif(varsize)
  # index of true CR 
  crtf=which(crtf==T)
  # The candidate being challenged
  titleholder <- pop[index[1],]
  ##### Note: every time we sample the candidate, we re-calculate its fitness.
  ##### !!!!: this is because of the randomness of DL. In this way, we will lower the chance
  ##### for DL model being 'good' by chance
  HP_0 <- get_hp(titleholder)
  # The index of this iteration
  event_log[i,28] <- i
  # Record the hyperparameter set of the candidate
  event_log[i,2:(2+length(HP_0)-1)] <- HP_0
  # In case fitting DL model fails, we save the event log ahead of fitting
  save(event_log,file="event_log_Real_MLP.RData")
  
  # make a challenger from title holder
  challenger=pop[index[1],]
  # modify challenger
  challenger[crtf]=pop[index[2],crtf]-mu*(pop[index[3],crtf]-pop[index[4],crtf]) 
  # get fitness for challenger
  HP <- get_hp(challenger)
  # save the challenger's hyperparameter in event log
  event_log[i,15:(15+length(HP)-1)] <- HP
  # again, save the event log before fitting challenger's model
  save(event_log,file="event_log_Real_MLP.RData")
  
  fit_0 <- try(fitness_MLP(HP_0$optimizer,HP_0$activation,HP_0$nlayers,HP_0$epoch,
                           HP_0$dropout_rate,HP_0$L2,HP_0$bsize,HP_0$neuron_layer1,HP_0$neuron_layer2,
                           HP_0$neuron_layer3,HP_0$neuron_layer4,HP_0$neuron_layer5))
  
  if('try-error' %in% class(fit_0))
  {
    # if there is an error, there will be an -1 for the fitness
    # and the event log will show -1 as its fitness
    pop[index[1],]=challenger
    fit[index[1]] <- -1
    k_clear_session()
    gc(reset=TRUE)
    next
  }else{
    # save the actual epochs
    HP_0$epoch <- fit_0[[2]]
    # if there is not any error, save the fitness
    fit[index[1]] <- as.numeric(fit_0[[1]])
    k_clear_session()
    gc(reset=TRUE)
  }
  
  # Record the updated epoch of the candidate as early stopping may happen
  event_log[i,2:(2+length(HP_0)-1)] <- HP_0
  # Save the recalculated fitness in event log
  event_log[i,1] <- fit[index[1]]
  # In case fitting DL model fails, we save the event log ahead of fitting
  save(event_log,file="event_log_Real_MLP.RData")
  
  
  # Use try function in case encounter error
  fit_1 <- try(fitness_MLP(HP$optimizer,HP$activation,HP$nlayers,HP$epoch,
                           HP$dropout_rate,HP$L2,HP$bsize,HP$neuron_layer1,HP$neuron_layer2,
                           HP$neuron_layer3,HP$neuron_layer4,HP$neuron_layer5))
  
  if('try-error' %in% class(fit_1))
  {
    fitchal <- -1
    k_clear_session()
    gc(reset=TRUE)
    next
  }else{
    # save the actual epochs
    HP$epoch <- fit_1[[2]]
    # if there is not any error, save the fitness
    fitchal <- as.numeric(fit_1[[1]])
    k_clear_session()
    gc(reset=TRUE)
  }
  
  # Fitness of the challenger
  event_log[i,14] <- fitchal
  # Record the updated epoch of the candidate as early stopping may happen
  event_log[i,15:(15+length(HP)-1)] <- HP
  # Save the training time for fitting two models
  event_log[i,29] <- round(Sys.time()-Time,digits = 3)
  # In case fitting DL model fails, we save the event log ahead of fitting
  save(event_log,file="event_log_Real_MLP.RData")
  fit <- as.numeric(unlist(fit))
  
  # if fitness of challenger better than fitness of titleholder, replace.
  if(!is.na(fitchal)&!is.na(fit[index[1]])&fitchal>fit[index[1]]) 
  {
    # In the numeric matrix, replace the individual
    pop[index[1],]=challenger
    # In the fitness scores of population, replace the fitness
    fit[index[1]]=fitchal
    # Indicate whether the challenger won
    event_log[i,27] <- TRUE
  }
  
  event_log[i,30] <- mean(fit,na.rm = T)
  
  # Save the mean fitness and sd of fitness among the population
  # every 10 iterations
  if(i%%10==0)
  {
    moniter_fit <- c(moniter_fit,mean(fit,na.rm = T))
    sd_fit <- c(sd_fit,sd(fit,na.rm = T))
  }
  
  # Save all the relevant results every ten iterations
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
         file = "DE_Real_MLP.RData")
  }
  
}



