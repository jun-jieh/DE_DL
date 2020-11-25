###################################################
###### A function for adaptive number of     ######
###### neurons according to number of layers ######
###################################################

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


#################################
###### Random key for MLPs ######
#################################

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

##################################
#######  Fitness function ########
##################################

### Note: this function fits and computes the fitness of the given hyperparameter ###

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
