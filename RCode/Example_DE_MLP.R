#### 1.Library, file, and object ####
# Load Keras library for Deep Learning
library(keras)
# Load gtool library to combine rows with different lengths (smartbind())
library(gtools)
# Load ggplot2 to generate plots
library(ggplot2)
# Load cluster library to compute the medoid
library(cluster)
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
# Load the genotype matrix. 'geno' matrix is for training dataset
geno <- geno[index,]
# Define the input shape of MLP regarding the number of SNP markers
shape <- ncol(geno)
# Load the phenotypes for training set
y <- y[index]


##### 2. Define hyperparameters #####
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


###### 3. Initialize a random population ######
# 5 candidate solution in the population.
popsize=5 
# Count unique markers of hyperparameters. The 3 at the end represents L2, dropout and batch proportion
varsize <- a+b+c+d+f+3   
# Randomly sample values from (0,1)
init <- runif(n = popsize*varsize,min=0,max = 1) 
# Generate matrix to evolve based on numeric values
pop=matrix(init,popsize,varsize)


###### 4. A function for adaptive number of neurons according to number of layers ######
gen_neurons <- function(number_layers, lower_limit, upper_limit)
{
  # Tell the unction the range of the neurons
  nneuron <- as.integer(lower_limit:upper_limit)
  # Divide the neuron space equally into blocks, according to the number of layers
  stride <- as.integer((upper_limit-lower_limit+1)/number_layers)
  # Create a list to save possible options in that block
  neurons <- NULL
  # The number of options in the block
  layer_width <- NULL
  # A loop to get divided blocks by considering the number of layers
  for(i in 1:number_layers)
  {
    neurons[[i]] <- as.integer(upper_limit-i*stride+1):as.integer(upper_limit-(i-1)*stride)
    if(i==number_layers){neurons[[i]] <- lower_limit:as.integer(upper_limit-(i-1)*stride)}
    layer_width[i] <- length(neurons[[i]])
  }
  # Return the neurons options, the number of options in each block, and the stride
  return(list(neurons,layer_width,stride))
}


###### 5. Actual hyperparameter sampler ######
# Get hyperparameters from given population
# Note: the input is a numeric vector!!!
get_hp <- function(vec)
{
  # op represents optimizer. We sort first a values given the vector
  # a is the length of hyperparameter space for optimizer. Similarly for b,c,d,and f
  # We sort the values given vector, and the index of first value will be sampled
  op <- sort(vec[1:a],index.return=T)$ix[1]
  # Sort and get activation
  ac <- sort(vec[(a+1):(a+b)],index.return=T)$ix[1]
  # Sort and get number of layer
  layer <- sort(vec[(a+b+1):(a+b+c)],index.return=T)$ix[1]
  # Sort and get number of epochs
  epo <- sort(vec[(a+b+c+d+1):(a+b+c+d+f)],index.return=T)$ix[1]
  # Given the number of layers, we get the neurons adaptive to the #layers
  neuron_info <- gen_neurons(layer,low_limit,up_limit)
  
  # One layer scenario
  if(layer==1)
  {
    # Sort and get the number of neurons for the only hidden layer
    neuron_layer1 <- sort(vec[(a+b+c+1):(a+b+c+d)],index.return=T)$ix[1]
    # When #layer=1,return the hyperparameters with only one hidden layer
    # Similar for #layer=2,3,4,5 in the followed pieces
    HP <- data.frame(optimizers[op],activations[ac],nlayers[layer],epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 4))),nneuron[neuron_layer1])
    colnames(HP) <- c('optimizer','activation','nlayers','epoch',
                      'dropout_rate','L2','bsize','neuron_layer1')
  }
  
  # Two layer scenario
  if(layer==2)
  {
    
    
    stride <- neuron_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(neuron_info[[1]][i]))
      assign(paste('neuron_layer',i,sep = ''),
             sort(vec[(count-i*stride+1):(count-(i-1)*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('neuron_layer',i,sep = ''), 
               sort(vec[(a+b+c+1):as.integer(count-(i-1)*stride)],index.return=T)$ix[1])
      }
    }
    
    HP <- data.frame(optimizers[op],activations[ac],nlayers[layer],epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 4))),
                     HL1[neuron_layer1], HL2[neuron_layer2])
    colnames(HP) <- c('optimizer','activation','nlayers','epoch',
                      'dropout_rate','L2','bsize',
                      'neuron_layer1','neuron_layer2')
  }
  
  # Three layer scenario
  if(layer==3)
  {
    stride <- neuron_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(neuron_info[[1]][i]))
      assign(paste('neuron_layer',i,sep = ''),
             sort(vec[(count-i*stride+1):(count-(i-1)*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('neuron_layer',i,sep = ''), 
               sort(vec[(a+b+c+1):as.integer(count-(i-1)*stride)],index.return=T)$ix[1])
      }
    }
    HP <- data.frame(optimizers[op],activations[ac],layer,epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 4))),
                     HL1[neuron_layer1], HL2[neuron_layer2],HL3[neuron_layer3])
    colnames(HP) <- c('optimizer','activation','nlayers','epoch',
                      'dropout_rate','L2','bsize','neuron_layer1',
                      'neuron_layer2','neuron_layer3')
  }
  
  # Four layer scenario
  if(layer==4)
  {
    stride <- neuron_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(neuron_info[[1]][i]))
      assign(paste('neuron_layer',i,sep = ''),
             sort(vec[(count-i*stride+1):(count-(i-1)*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('neuron_layer',i,sep = ''),
               sort(vec[(a+b+c+1):as.integer(count-(i-1)*stride)],index.return=T)$ix[1])
      }
    }
    HP <- data.frame(optimizers[op],activations[ac],layer,epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 4))),
                     HL1[neuron_layer1], HL2[neuron_layer2],HL3[neuron_layer3],HL4[neuron_layer4])
    colnames(HP) <- c('optimizer','activation','nlayers','epoch',
                      'dropout_rate','L2','bsize','neuron_layer1',
                      'neuron_layer2','neuron_layer3','neuron_layer4')
  }
  
  # Five layer scenario
  if(layer==5)
  {
    stride <- neuron_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(neuron_info[[1]][i]))
      assign(paste('neuron_layer',i,sep = ''),
             sort(vec[(count-i*stride+1):(count-(i-1)*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('neuron_layer',i,sep = ''),
               sort(vec[(a+b+c+1):as.integer(count-(i-1)*stride)],index.return=T)$ix[1])
      }
    }
    
    HP <- data.frame(optimizers[op],activations[ac],layer,epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 4))),
                     HL1[neuron_layer1], HL2[neuron_layer2],HL3[neuron_layer3],
                     HL4[neuron_layer4], HL5[neuron_layer5])
    colnames(HP) <- c('optimizer','activation','nlayers','epoch',
                      'dropout_rate','L2','bsize','neuron_layer1',
                      'neuron_layer2','neuron_layer3','neuron_layer4','neuron_layer5')
  }
  
  # Keep in mind optimizer is character variable, otherwise keras will receive error
  HP[,1] <- as.character(HP[,1]); HP[,1] <- optimizers[op]
  # Activation function is also character variable
  HP[,2] <- as.character(HP[,2]); HP[,2] <- activations[ac]
  # The dropout rate and L2 rate cannot be larger than 1
  HP$dropout_rate <- HP$dropout_rate%%1
  HP$L2 <- HP$L2%%1
  return(HP)
}


#### 6. Custom metrics to monitor DL ####
# WARNING!!! THIS IS DEPRECATED IT SHOULD BE UPDATED
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



####### 7. Fitness function ########
# Parameters include optimizer, activation, number of layers,
# epochs, dropout rate, L2 rate, batch size, and neurons for each layer
# Note: As neurons for each layer are uncertainties, we just leave it at the end
fitness_MLP <- function(optimizer,activation,nlayers,epo,dropout_rate,L2,batch,neuron_layer1,
                        neuron_layer2=NULL,neuron_layer3=NULL,neuron_layer4=NULL,
                        neuron_layer5=NULL)
{
  
  # One layer scenario
  if(nlayers==1)
  {
    
    # Setup the Deep learning model. First, assign the model with hyperparameters
    # before executing the algorithm. The layers are connected by pipeline form
    model_mlp <- keras_model_sequential()
    model_mlp %>%
      layer_dense(units = neuron_layer1, activation=activation, input_shape = shape,
                  kernel_regularizer = regularizer_l2(l = L2)) %>%
      layer_dropout(rate=dropout_rate) %>%
      layer_dense(units = 1)
    
    # Compiling. Define the loss function, optimizer to update weights, and 
    # monitering metrics
    model_mlp %>%
      compile(loss = 'mse',
              optimizer = optimizer,
              metrics = c('mse','corr'=metric_cor))
    
    
    
    # Fit Model. Note: We set a early stopping rule if there is no change over 0.1
    # in 10 consecutive epochs
    history <- model_mlp %>%
      fit(geno,
          y,
          epochs = epo,
          batch_size = max(6,batch),
          validation_split = 0.2,
          verbose=0,
          callbacks = list(callback_early_stopping(monitor = 'val_corr', min_delta = 0.1,
                                                   patience = 10, verbose = 0, mode = "max",
                                                   baseline = NULL, restore_best_weights = TRUE))
      )
    
    
  }
  
  if(nlayers==2)
  {
    
    # Setup the Deep learning model
    model_mlp <- keras_model_sequential()
    model_mlp %>%
      layer_dense(units = neuron_layer1, activation=activation, input_shape = shape,
                  kernel_regularizer = regularizer_l2(l = L2)) %>%
      layer_dropout(rate=dropout_rate) %>%
      layer_dense(units = neuron_layer2, activation=activation) %>%
      layer_dense(units = 1)
    
    # Compiling
    model_mlp %>%
      compile(loss = 'mse',
              optimizer = optimizer,
              metrics = c('mse','corr'=metric_cor))
    
    
    # Fit Model
    history <- model_mlp %>%
      fit(geno,
          y,
          epochs = epo,
          batch_size = max(6,batch),
          validation_split = 0.2,
          verbose=0,
          callbacks = list(callback_early_stopping(monitor = 'val_corr', min_delta = 0.1,
                                                   patience = 10, verbose = 0, mode = "max",
                                                   baseline = NULL, restore_best_weights = TRUE))
      )
    
  }
  
  if(nlayers==3)
  {
    # Setup the Deep learning model
    model_mlp <- keras_model_sequential()
    model_mlp %>%
      layer_dense(units = neuron_layer1, activation=activation, input_shape = shape,
                  kernel_regularizer = regularizer_l2(l = L2)) %>%
      layer_dropout(rate=dropout_rate) %>%
      layer_dense(units = neuron_layer2, activation=activation) %>%
      layer_dense(units = neuron_layer3, activation=activation) %>%
      layer_dense(units = 1)
    
    # Compiling
    model_mlp %>%
      compile(loss = 'mse',
              optimizer = optimizer,
              metrics = c('mse','corr'=metric_cor))
    
    
    # Fit Model
    history <- model_mlp %>%
      fit(geno,
          y,
          epochs = epo,
          batch_size = max(6,batch),
          validation_split = 0.2,
          verbose=0,
          callbacks = list(callback_early_stopping(monitor = 'val_corr', min_delta = 0.1,
                                                   patience = 10, verbose = 0, mode = "max",
                                                   baseline = NULL, restore_best_weights = TRUE))
      )
    
    
  }
  if(nlayers==4)
  {
    # Setup the Deep learning model
    model_mlp <- keras_model_sequential()
    model_mlp %>%
      layer_dense(units = neuron_layer1, activation=activation, input_shape = shape,
                  kernel_regularizer = regularizer_l2(l = L2)) %>%
      layer_dropout(rate=dropout_rate) %>%
      layer_dense(units = neuron_layer2, activation=activation) %>%
      layer_dense(units = neuron_layer3, activation=activation) %>%
      layer_dense(units = neuron_layer4, activation=activation) %>%
      layer_dense(units = 1)
    
    # Compiling
    model_mlp %>%
      compile(loss = 'mse',
              optimizer = optimizer,
              metrics = c('mse','corr'=metric_cor))
    
    
    # Fit Model
    history <- model_mlp %>%
      fit(geno,
          y,
          epochs = epo,
          batch_size = max(6,batch),
          validation_split = 0.2,
          verbose=0,
          callbacks = list(callback_early_stopping(monitor = 'val_corr', min_delta = 0.1,
                                                   patience = 10, verbose = 0, mode = "max",
                                                   baseline = NULL, restore_best_weights = TRUE))
      )
    
    
  }
  if(nlayers==5)
  {
    # Setup the Deep learning model
    model_mlp <- keras_model_sequential()
    model_mlp %>%
      layer_dense(units = neuron_layer1, activation=activation, input_shape = shape,
                  kernel_regularizer = regularizer_l2(l = L2)) %>%
      layer_dropout(rate=dropout_rate) %>%
      layer_dense(units = neuron_layer2, activation=activation) %>%
      layer_dense(units = neuron_layer3, activation=activation) %>%
      layer_dense(units = neuron_layer4, activation=activation) %>%
      layer_dense(units = neuron_layer5, activation=activation) %>%
      layer_dense(units = 1)
    
    # Compiling
    model_mlp %>%
      compile(loss = 'mse',
              optimizer = optimizer,
              metrics = c('mse','corr'=metric_cor))
    
    
    # Fit Model
    history <- model_mlp %>%
      fit(geno,
          y,
          epochs = epo,
          # The minimum batch size for SimPig and SimCattle is 6 (0.1% of the training)
          # The minimum batch size for RealPig is 1 (0.1% of the training)
          batch_size = max(6,batch), 
          validation_split = 0.2,
          verbose=0,
          # The MLP is minimizing loss function mse. However, if the internal validation
          # (correlation) does not change over 0.1 for 10 epochs, we stop the model fitting
          callbacks = list(callback_early_stopping(monitor = 'val_corr', min_delta = 0.1,
                                                   patience = 10, verbose = 0, mode = "max",
                                                   baseline = NULL, restore_best_weights = TRUE))
      )
    
    
  }
  
  # The score is the correlation of internal validation
  score <- mean(history$metrics$val_corr, na.rm = T)
  # Check the history object and return the actual training epochs used considering 
  # early stopping rule
  Actual_epochs <- length(history$metrics$val_corr)
  # Clear the memory took up by KERAS DL object in case out of storage
  k_clear_session()
  # Clear R environment
  gc(reset=TRUE)
  # Sometimes we get infinity
  if(!is.na(score)&score==Inf | !is.na(score)&score==-Inf) {score <- -1}
  # Sometimes we get NA
  if(is.na(score)) {score <- -1}
  return(list(round(score,5),Actual_epochs))
}


###### 8. Set up parameters for DE######
numgen=100    # number of generations in DE
CR=0.5        # probability of crossover
mu=0.5        # mutation rate

fit=numeric(popsize)  # We want a pool of candidate solutions
info_DL <- numeric(length = popsize) # Initial fitness of the population

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
initial_info <- info_DL
fit <- info_DL
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

# Initial monitoring mesurement including correlation and sd of correlation among population 
moniter_fit <- NULL
sd_fit <- NULL


###### 9. Now we go to the evolutional algorithm ######
# Run numgen iterations and break certain iteration if necessary
for (i in 1:numgen)
{
  # Mark the start time of this iteration
  Time <- Sys.time()
  # choose 4 from population - 1 title holder, 2 template, 3 and 4 mutators
  index=sample(popsize,4)
  # true or false for CR - generate random numbers for CR and then test
  crtf=CR>runif(varsize)
  # index of true CR 
  crtf=which(crtf==T)
  # The candidate being challenged
  challenged <- pop[index[1],]
  ##### Note: every time we sample the candidate, we re-calculate its fitness.
  ##### !!!!: this is because of the randomness of DL. In this way, we will lower the chance
  ##### for DL model being 'good' by chance
  HP_0 <- get_hp(challenged)
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
    # if there is an error, there will be an -1 for the fitness
    # and the event log will show -1 as its fitness
    fit[index[1]] <- -1
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
  
  # if fitness of challenger better than fitness of title holder, replace.
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


###### 10. Visualize results ######
setwd("~/")
# Results from first DE run
load("DE_Real_MLP.RData")
# Mean fitness of the population over generations
acc_mean <- moniter_fit
# SD of the fitness of the population over generations
acc_sd <- sd_fit
# Plot the evolution curve (mean fitness)
x_axis = 10*(1:length(acc_mean))
ggplot(data= as.data.frame(acc_mean),aes(x=x_axis, y=acc_mean)) +
  geom_line(size=1.2) + 
  labs(subtitle="Accuracy vs iterations", 
       y="Mean fitness of population", 
       x="DE generations", 
       title="MLP on real pig data") +
  theme_bw()
# Plot the evolution curve (SD fitness)
ggplot(data= as.data.frame(acc_sd),aes(x=x_axis, y=acc_sd)) +
  geom_line(size=1.2) + 
  labs(subtitle="SD vs iterations", 
       y="SD fitness of population", 
       x="DE generations", 
       title="MLP on real pig data") +
  theme_bw()

# Hyperparameter distributions
cat("Evolved optimizer distribution: \n ")
cat("DE 1:")
table(evolve_pop$optimizer)

cat("\n \n Evolved activation function distribution: \n ")
cat("DE 1:")
table(evolve_pop$activation)

cat("\n \n #Layers distribution: \n ")
cat("DE 1:")
table(evolve_pop$nlayers)

cat("\n \n Summary of dropout: \n ")
cat("DE 1:")
summary(evolve_pop$dropout_rate)

cat("\n \n Summary of L2 Regularization: \n ")
cat("DE 1:")
summary(evolve_pop)

# Batch size
quantile(evolve_pop$bsize,probs = c(0.05,0.5,0.95))

# Epochs
quantile(evolve_pop$epoch,probs = c(0.05,0.5,0.95))

# Dropout
quantile(evolve_pop$dropout_rate,probs = c(0.05,0.5,0.95))

# L2
quantile(evolve_pop_DE1$L2,probs = c(0.05,0.5,0.95))


###### 11. Repeat Training individuals in the evolved population for 30 times ######
Nrepeat <- 30
corr_track <- as.data.frame(matrix(data = 0,nrow = Nrepeat*nrow(evolve_pop) ,ncol = 2))
colnames(corr_track) <- c("H_param","Fitness")
for(i in 1:nrow(evolve_pop))
{
  hp <- paste("HP",i)
  corr_track$H_param[((i-1)*Nrepeat+1):(i*Nrepeat)] <- hp
}


for(i in 1:nrow(evolve_pop))
{
  corr_fit <- numeric(length = Nrepeat)
  HP <- evolve_pop[i,]
  for(j in 1:Nrepeat)
  {
    k_clear_session()
    gc(reset=TRUE)
    
    run_fit <- try(fitness_MLP(HP$optimizer,HP$activation,HP$nlayers,HP$epoch,
                               HP$dropout_rate,HP$L2,HP$bsize,HP$neuron_layer1,HP$neuron_layer2,
                               HP$neuron_layer3,HP$neuron_layer4,HP$neuron_layer5))
    
    if('try-error' %in% class(run_fit))
    {
      # if there is an error, there will be an -1 for the fitness
      # and the event log will show -1 as its fitness
      corr_fit[j] <- -1
      k_clear_session()
      gc(reset=TRUE)
      next
    }else{
      # if there is not any error, save the fitness
      corr_fit[j] <- as.numeric(run_fit[[1]])
      k_clear_session()
      gc(reset=TRUE)
    }
  }
  corr_track$Fitness[((i-1)*Nrepeat+1):(i*Nrepeat)] <- corr_fit
}

mean_fit <- numeric(length = nrow(evolve_pop))
sd_fit <- numeric(length = nrow(evolve_pop))

for(i in 1:nrow(evolve_pop))
{
  mean_fit[i] <-  mean(corr_track$Fitness[((i-1)*Nrepeat+1):(i*Nrepeat)])
  sd_fit[i] <-  sd(corr_track$Fitness[((i-1)*Nrepeat+1):(i*Nrepeat)])
  
}

stats_dl <- cbind(mean_fit,sd_fit)
rownames(stats_dl) <- paste("HP",1:nrow(evolve_pop)) 
save(stats_dl,corr_track,file = "Real_Repeat_Stats.RData")


###### 12. Model selection from the evolved population ######
load("Real_Repeat_Stats.RData")
drop_index <- which(stats_dl[,1]==-1)
if(length(drop_index)>0)
{stats_dl <- stats_dl[-drop_index,]}
###### A function for evaluating and selecting best model given stats of repeated training ######
max_Euc_dist <- function(stats_dl)
{
  # Compute and find the medoid of the point clusters
  medoid <- pam(stats_dl,1)$medoids
  # Define a variable to save Euclidean distance between points and the center
  Euc_dist <- numeric(length = nrow(stats_dl)) 
  # Excute the computation
  for(i in 1:nrow(stats_dl))
  {
    Euc_dist[i] <- sqrt((stats_dl[i,1]-medoid[1])^2+(stats_dl[i,2]-medoid[2])^2)
  }
  # Combine information
  stats_dl <- cbind(stats_dl,Euc_dist)
  # We only focus on the 4th "quadrant" where the center is the origin
  quadrant_4th <- which(stats_dl[,1]>medoid[1]&stats_dl[,2]< medoid[2]) 
  selected_HP <- which(stats_dl[,3]==max(stats_dl[quadrant_4th,3]))
  return(selected_HP)
}

medoid <- pam(stats_dl,1)$medoids
selected_HP <- max_Euc_dist(stats_dl)
ggplot(as.data.frame(stats_dl), aes(x=mean_fit, y=sd_fit)) +
  geom_point()+
  geom_point(data = as.data.frame(medoid),
             aes(x=mean_fit, y=sd_fit,shape="Medoid"),col='red',size=3)+
  geom_point(data = as.data.frame(t(stats_dl[selected_HP,])),
             aes(x=mean_fit,y=sd_fit,shape="Selected HP"),col='red',size=3)+
  labs(subtitle="Mean fitness vs SD fitness",
       y="Standard Deviation",
       x="Mean fitness",
       title="Results for repeated training")+
  theme_bw()

cat("\n The top MLP Hyperparameter for DE1, Real Pig: \n ")
names(selected_HP)





