library(tensorflow)
library(keras)

finalModel="finalModel"
# Set working directory. The data should be ready in this path
setwd("D:/DE_DL/Cedric's code/")
geno=readRDS("genotypes.rds")
geno=t(geno)
pheno=read.table("SimPheno100QTL.txt",header=T,sep="\t")
y=pheno$y
allacc=NULL # store accuracy for all calls - just for testing/eval purposes

# subset into training/testing set
index = 1:1000
genov = geno[-index,]
yv = y[-index]
geno = geno[index,]
y = y[index]

################################################
##### hyperparameter info and constraints #####
##############################################
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

constraints=rbind(nlayers,epochs,dropout,L2,batch,nneuron,dneuron)
colnames(constraints)=c("min","max")
rownames(constraints)=c("nlayers","epochs","dropout","L2","batch","nneuron","dneuron")

################################################
#### functions ################################
##############################################
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

fitness = function(pars,store=F) # gets vector of hyperparameters, builds and runs DL model, returns correlation vector of epochs
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

makesol=function(constraints)
{
	pars=numeric(nrow(constraints))
	for (i in 1:nrow(constraints)) pars[i]=runif(1,constraints[i,1],constraints[i,2])
	pars=c(pars,runif(length(optimizers)),runif(length(activations)))
	return(pars)
}

K = backend() # connect

##########################################
### differential evolution ##############
########################################

# DE parameters
popsize=50 # population size
numgen=100 # number of generations to iterate
CR=0.5 # probability of crossover
FR=0.5 # mutation rate
noise=0.3 # add random noise to crank up mutations occasionally

# make initial population of solutions
pop=NULL
for (i in 1:popsize) pop=rbind(pop,makesol(constraints))
allelesize=ncol(pop) # number of parameters to optimize

# calculate fitness of initial population
fit=numeric(popsize)
for (i in 1:popsize)
{
	cors=try(fitness(pop[i,]))
	if('try-error' %in% class(cors)) fit[i]=-1
	else fit[i]=median(cors)
	if(is.na(fit[i])==T | fit[i]==Inf | fit[i]==-Inf) fit[i]=-1
	print(fit[i])
}

cat(paste("initial population:",round(max(fit),3),"\n"))
plot(max(fit),xlim=c(0,numgen),ylim=c(mean(fit),1),xlab="generation",ylab="fitness",cex=0.8,col="blue",pch=20) # just a plot to see how fitness evolves - change boundaries depending on possible values

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

rungen=function(x) # call challenge function, calculate fitness of new solutions, evaluate against previous solutions
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
  points(x,max(fit),cex=0.8,col="blue",pch=20)
  cat(paste("generation",x,":",round(max(fit),3),"\n"))
}
for (i in 1:numgen) rungen(i) # run DE

##########################################
######## results ########################
########################################
index=which(fit==max(fit))[1]
cors=try(fitness(pop[index,],store=TRUE)) # save the final best model to disk

model_mlp = load_model_hdf5("finalModel",custom_objects = c(corvalid = corvalid)) # load model from disk

# evaluate
pred=as.vector(predict(model_mlp, geno)) # training fit
cor(pred,y)
pred=as.vector(predict(model_mlp, genov)) # testing fit
cat(paste("accuracy in validation dataset:",round(cor(pred,yv),3),"\n"))



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

#pars=makesol(constraints) # a random solution
#cors=try(fitness(pars))

