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
# Load the genotype matrix
geno <- geno[index,]
# Define the input shape of MLP
shape <- ncol(geno)
# Prepare the data format for CNN (2D array)
geno <- array_reshape(geno, list(nrow(geno), ncol(geno), 1))
# Load the phenotypical values
y <- y[index]

##### 2. Define hyperparameters #####
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
# Just a sum of a+b+c as a flag
count=a+b+c

###### 3. Initialize a random population ######
# 50 candidate solution in the population. We'll choose best one in the end
popsize=5
# Count unique markers of hyperparameters
varsize <- a+b+c+d+e+f+g+h+3
# Randomly sample values from (0,1)
init <- runif(n = popsize*varsize,min=0,max = 1) 
# Generate matrix to evolve based on numeric values
pop=matrix(init,popsize,varsize)

###### 4. A function for adaptive number of filters according to number of layers ######
# gen_filters() will return a list with three elements:
# [1]Filter options adaptive to #layers; [2]Possible network width(#filters) in each layer
# [3]The stride to divide #filter hyperparameter space given #layers
gen_filters <- function(number_layers, lower_limit, upper_limit)
{
  # How many options in total given a range
  nfilter <- as.integer(lower_limit:upper_limit)
  # Divide the range equally by #layers
  stride <- as.integer((upper_limit-lower_limit+1)/number_layers)
  filters <- NULL
  layer_width <- NULL
  for(i in 1:number_layers)
  {
    filters[[i]] <- as.integer(lower_limit+(i-1)*stride):as.integer(lower_limit+i*stride-1)
    if(i==number_layers){filters[[i]] <- as.integer(lower_limit+(i-1)*stride):upper_limit}
    layer_width[i] <- length(filters[[i]])
  }
  return(list(filters,layer_width,stride))
}

###### 5. Actual hyperparameter sampler ######
# Get hyperparameters from given population
get_hp <- function(vec)
{
  op <- sort(vec[1:a],index.return=T)$ix[1]
  ac <- sort(vec[(a+1):(a+b)],index.return=T)$ix[1]
  fsize <- sort(vec[(a+b+1):(a+b+c)],index.return=T)$ix[1]
  pool <- sort(vec[(a+b+c+d+1):(a+b+c+d+e)],index.return=T)$ix[1]
  layer <- sort(vec[(a+b+c+d+e+1):(a+b+c+d+e+f)],index.return=T)$ix[1]
  epo <- sort(vec[(a+b+c+d+e+f+1):(a+b+c+d+f+e+g)],index.return=T)$ix[1]
  mlp <- sort(vec[(a+b+c+d+e+f+g+1):(a+b+c+d+f+e+g+h)],index.return=T)$ix[1]
  
  filter_info <- gen_filters(layer,low_limit,up_limit)
  
  if(layer==1)
  {
    # for(i in 1:layer)
    # {
    #   assign(paste('neuron_layer',i), sort(vec[(count+1):(a+b+c+d)],index.return=T)$ix[1])
    # }
    
    filter_layer1 <- sort(vec[(a+b+c+1):(a+b+c+d)],index.return=T)$ix[1]
    HP <- data.frame(optimizers[op],activations[ac],nlayers[layer],filter_size[fsize],
                     pools[pool],epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 3))), layer_mlp[mlp],
                     nfilters[filter_layer1])
    colnames(HP) <- c('optimizer','activation','nlayers','filter_size','pool_layer','epoch',
                      'dropout_rate','L2','bsize','layer_mlp', 'filters_layer1')
  }
  if(layer==2)
  {
    
    
    stride <- filter_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(filter_info[[1]][i]))
      assign(paste('filters_layer',i,sep = ''),
             sort(vec[(count+(i-1)*stride+1):(count+i*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('filters_layer',i,sep = ''), sort(vec[as.integer(count+(i-1)*stride+1):(a+b+c+d)],index.return=T)$ix[1])
      }
    }
    
    HP <- data.frame(optimizers[op],activations[ac],nlayers[layer],filter_size[fsize],
                     pools[pool],epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 3))),layer_mlp[mlp],
                     HL1[filters_layer1], HL2[filters_layer2])
    colnames(HP) <- c('optimizer','activation','nlayers','filter_size','pool_layer','epoch',
                      'dropout_rate','L2','bsize','layer_mlp','filters_layer1','filters_layer2')
  }
  if(layer==3)
  {
    stride <- filter_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(filter_info[[1]][i]))
      assign(paste('filters_layer',i,sep = ''),
             sort(vec[(count+(i-1)*stride+1):(count+i*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('filters_layer',i,sep = ''), sort(vec[as.integer(count+(i-1)*stride+1):(a+b+c+d)],index.return=T)$ix[1])
      }
    }
    HP <- data.frame(optimizers[op],activations[ac],nlayers[layer],filter_size[fsize],
                     pools[pool],epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 3))),layer_mlp[mlp],
                     HL1[filters_layer1], HL2[filters_layer2],HL3[filters_layer3])
    colnames(HP) <- c('optimizer','activation','nlayers','filter_size','pool_layer','epoch',
                      'dropout_rate','L2','bsize','layer_mlp','filters_layer1','filters_layer2',
                      'filters_layer3')
  }
  if(layer==4)
  {
    stride <- filter_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(filter_info[[1]][i]))
      assign(paste('filters_layer',i,sep = ''),
             sort(vec[(count+(i-1)*stride+1):(count+i*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('filters_layer',i,sep = ''), sort(vec[as.integer(count+(i-1)*stride+1):(a+b+c+d)],index.return=T)$ix[1])
      }
    }
    HP <- data.frame(optimizers[op],activations[ac],nlayers[layer],filter_size[fsize],
                     pools[pool],epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 3))),layer_mlp[mlp],
                     HL1[filters_layer1], HL2[filters_layer2],HL3[filters_layer3],
                     HL4[filters_layer4])
    colnames(HP) <- c('optimizer','activation','nlayers','filter_size','pool_layer','epoch',
                      'dropout_rate','L2','bsize','layer_mlp','filters_layer1','filters_layer2',
                      'filters_layer3','filters_layer4')
  }
  if(layer==5)
  {
    stride <- filter_info[[3]]
    for(i in 1:layer)
    {
      assign(paste('HL',i,sep = ''),unlist(filter_info[[1]][i]))
      assign(paste('filters_layer',i,sep = ''),
             sort(vec[(count+(i-1)*stride+1):(count+i*stride)],index.return=T)$ix[1])
      if(i==layer)
      {
        assign(paste('filters_layer',i,sep = ''), sort(vec[as.integer(count+(i-1)*stride+1):(a+b+c+d)],index.return=T)$ix[1])
      }
    }
    HP <- data.frame(optimizers[op],activations[ac],nlayers[layer],filter_size[fsize],
                     pools[pool],epochs[epo],
                     abs(round(vec[varsize-2],digits = 3)),abs(round(vec[varsize-1],digits = 3)),
                     as.integer(Nobs*abs(round(vec[varsize]%%0.1,digits = 3))),layer_mlp[mlp],
                     HL1[filters_layer1], HL2[filters_layer2],HL3[filters_layer3],
                     HL4[filters_layer4],HL5[filters_layer5])
    colnames(HP) <- c('optimizer','activation','nlayers','filter_size','pool_layer','epoch',
                      'dropout_rate','L2','bsize','layer_mlp','filters_layer1','filters_layer2',
                      'filters_layer3','filters_layer4','filters_layer5')
  }
  
  HP[,1] <- as.character(HP[,1]); HP[,1] <- optimizers[op]
  HP[,2] <- as.character(HP[,2]); HP[,2] <- activations[ac]
  HP[,5] <- as.character(HP[,5])
  HP$dropout_rate <- HP$dropout_rate%%1
  HP$L2 <- HP$L2%%1
  return(HP)
}

#### 6. Custom metrics to monitor DL ####
#WARNING!!! THIS IS DEPRECATED IT SHOULD BE UPDATED
K <- backend()

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

#### 7. funciton for adaptive filter sizes for each layer ######
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

#### 8. Generate a CNN architecture ####
build_cnn <-  function(activation='relu',nlayers=1,filter_size=2,
                       pool_layer= 'layer_max_pooling_1d(pool_size = 2)',
                       dropout_rate=0.01, L2=0.01, layer_mlp=128,
                       filters_layer1=512,filters_layer2=NULL,filters_layer3=NULL,
                       filters_layer4=NULL,filters_layer5=NULL)
{
  k_clear_session()
  gc(reset=TRUE)
  Nfilters <- c(filters_layer1,filters_layer2,filters_layer3,
                filters_layer4,filters_layer5)
  kernel_size <- get_k_size(shape=shape, nlayers=nlayers, filter_size=filter_size)
  
  old<-file.exists("Model_example_DE1.R")
  if(old){
    warning("File Model_example_DE1.R exists and it will be deteled")
    file.remove("Model_example_DE1.R")
  }
  sink("Model_example_DE1.R")
  
  s0 <- "model_cnn <- keras_model_sequential() \n"
  cat(s0)
  s1<-"model_cnn %>% \n"
  cat(s1)
  s2<-paste("layer_conv_1d(filters = ",Nfilters[1],",kernel_size=",kernel_size[1], ",activation= '",activation,
            "', stride=",kernel_size[1]," ,input_shape = c(shape,1))%>% \n",sep="")
  cat(s2)
  cat("layer_zero_padding_1d() %>% \n")
  s3 <- paste(pool_layer," %>% \n")
  cat(s3)
  if (nlayers>1){
    for (i in 2:nlayers){
      si<-paste("layer_conv_1d(filters = ",Nfilters[i],",kernel_size=",kernel_size[i]," , activation= '",
                activation,"', stride=",kernel_size[i]," )%>% \n",sep="")  
      cat(si)
      cat("layer_zero_padding_1d() %>% \n")
      sip <- paste(pool_layer," %>% \n")
      cat(sip)
    }
  }
  flatten <- "layer_flatten() %>% \n"
  cat(flatten)
  s4 <- paste("layer_dropout(rate=",dropout_rate,") %>% \n", sep = "")
  cat(s4)
  s5 <- paste("layer_dense(units=",layer_mlp,",activation='",activation,
              "',kernel_regularizer = regularizer_l2(l =",L2,")) %>% \n", sep = "")
  cat(s5)
  sf<-"layer_dense(units = 1,activation = 'linear') \n \n"
  cat(sf)
  sink()
}

####### 9. Fitness function ########
fitness_CNN <- function(Hyper)
{
  build_cnn(activation=Hyper$activation,nlayers=Hyper$nlayers,filter_size=Hyper$filter_size,
            pool_layer= Hyper$pool_layer,
            dropout_rate=Hyper$dropout_rate, L2=Hyper$L2, layer_mlp=Hyper$layer_mlp,
            filters_layer1=Hyper$filters_layer1,filters_layer2=Hyper$filters_layer2,
            filters_layer3=Hyper$filters_layer3,filters_layer4=Hyper$filters_layer4,
            filters_layer5=Hyper$filters_layer5)
  
  source("Model_example_DE1.R")
  
  
  
  # Compiling
  model_cnn %>%
    compile(loss = 'mse',
            optimizer = Hyper$optimizer,
            metrics = c('mse','corr'=metric_cor)) 
  
  
  # Fit Model
  history <- model_cnn %>%
    fit(geno,
        y,
        epochs = Hyper$epoch,
        batch_size = 32,
        validation_split = 0.2,
        verbose=0,
        # The CNN is minimizing loss function mse. However, if the internal validation
        # (correlation) does not change over 0.1 for 10 epochs, we stop the model fitting
        callbacks = list(callback_early_stopping(monitor = 'val_corr', min_delta = 0.1,
                                                 patience = 10, verbose = 0, mode = "max",
                                                 baseline = NULL, restore_best_weights = TRUE))
    )
  
  score <- mean(history$metrics$val_corr,na.rm = T)
  Actual_epochs <- length(history$metrics$val_corr) 
  k_clear_session()
  gc(reset=TRUE)
  if(!is.na(score)&score==Inf | !is.na(score)&score==-Inf) {score <- -1}
  if(is.na(score)) {score <- -1}
  return(list(round(score,5),Actual_epochs))
}

###### 10. Set up parameters for DE######
numgen=100  # number of iterations
CR=0.5        # probability of crossover
mu=0.5        # mutation rate

fit=numeric(popsize)  # We want a pool of candidate solutions
info_DL <- numeric(length = popsize)

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
  fit <- try(fitness_CNN(HP))
  if('try-error' %in% class(fit))
  {
    info_DL[i] <- -1
    k_clear_session()
    gc(reset=TRUE)
    next
  }else{
    initial_pop[i,]$epoch <- fit[[2]]
    info_DL[i] <- as.numeric(fit[[1]])
    k_clear_session()
    gc(reset=TRUE)
  }
  
}

initial_info <- info_DL
fit <- info_DL
save(initial_info,pop,file="initial_info_Real_CNN.RData")

event_log <- as.data.frame(matrix(data = NA,nrow = numgen,ncol = 36))
colnames(event_log) <- c('Target Fitness','optimizer','activation','nlayers', 'filter_size',
                         'pool_layer', 'epoch', 'dropout_rate','L2','bsize','layer_mlp',
                         'filters_layer1','filters_layer2','filters_layer3','filters_layer4','filters_layer5',
                         'Challenger Fitness','c_optimizer','c_activation','c_nlayers', 'c_filter_size',
                         'c_pool_layer', 'c_epoch', 'c_dropout_rate','c_L2','c_bsize','c_layer_mlp',
                         'c_filters_layer1','c_filters_layer2','c_filters_layer3','c_filters_layer4',
                         'c_filters_layer5','c_succeed','num_iter','train_time','mean_fit')
event_log[,33] <- FALSE


moniter_fit <- NULL
sd_fit <- NULL

###### 11. Now we go to the evolutional algorithm ######
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


###### 10. Visualize results ######
setwd("~/")
# Results from first DE run
load("DE_Real_CNN.RData")
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
       title="CNN on real pig data") +
  theme_bw()
# Plot the evolution curve (SD fitness)
ggplot(data= as.data.frame(acc_sd),aes(x=x_axis, y=acc_sd)) +
  geom_line(size=1.2) + 
  labs(subtitle="SD vs iterations", 
       y="SD fitness of population", 
       x="DE generations", 
       title="CNN on real pig data") +
  theme_bw()

cat("Evolved optimizer distribution: \n ")
cat("DE 1:")
table(evolve_pop$optimizer)

cat("\n \n Evolved activation function distribution: \n ")
cat("DE 1:")
table(evolve_pop$activation)

cat("\n \n #Layers distribution: \n ")
cat("DE 1:")
table(evolve_pop$nlayers)

cat("\n \n Pooling distribution: \n ")
cat("DE 1:")
table(evolve_pop$pool_layer)

# Filter size
quantile(evolve_pop$filter_size,probs = c(0.05,0.5,0.95))
# Fully connected layer
quantile(evolve_pop$layer_mlp,probs = c(0.05,0.5,0.95))
# Epochs
quantile(evolve_pop$epoch,probs = c(0.05,0.5,0.95))
# Dropout
quantile(evolve_pop$dropout_rate,probs = c(0.05,0.5,0.95))
# L2
quantile(evolve_pop$L2,probs = c(0.05,0.5,0.95))


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
    
    run_fit <- try(fitness_CNN(HP))
    
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

