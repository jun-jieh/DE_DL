###############################
###### Visualize results ######
###############################

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

# Hyperparameter distributions of optimizer
cat("Evolved optimizer distribution: \n ")
table(evolve_pop$optimizer)

# Hyperparameter distributions of activation
cat("\n \n Evolved activation function distribution: \n ")
table(evolve_pop$activation)

# Hyperparameter distributions for number of layers
cat("\n \n #Layers distribution: \n ")
table(evolve_pop$nlayers)

# Hyperparameter distributions of pooling
cat("\n \n Pooling distribution: \n ")
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

#########################################################
###### Repeat Training individuals in the evolved  ######
###### population for 30 times                     ######
#########################################################

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

#############################
###### Model selection ######
#############################

# Load data from the previous step
load("Real_Repeat_Stats.RData")
drop_index <- which(stats_dl[,1]==-1)
if(length(drop_index)>0)
{stats_dl <- stats_dl[-drop_index,]}

###### A function for evaluating and selecting best  ######
###### model given stats of repeated training        ######
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

cat("\n The top MLP Hyperparameter, Real Pig: \n ")
names(selected_HP)


