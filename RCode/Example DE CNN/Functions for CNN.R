###################################################
###### A function for adaptive number of     ######  
###### filters according to number of layers ######
###################################################

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

#################################
###### Random key for CNNs ######
#################################

# Get hyperparameters from given individual vector
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

#############################################################
#### Funciton for adaptive filter sizes for each layer ######
#############################################################


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

######################################
##### Specify a CNN architecture #####
######################################

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


#################################
####### Fitness function ########
#################################

### Note: this function fits and computes the fitness of the given hyperparameter ###

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



