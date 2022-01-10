library(caret)


create.tuning.criterion.storage <- function(param.set.list){
  n.param <- length(param.set.list)
  n.comb <- prod(sapply(param.set.list, length))
  storage <- matrix(NA, nrow = n.comb, ncol = n.param + 1 ) # last column is for criterion values
  storage <- data.frame(storage)
  colnames(storage) <- c(names(param.set.list), "criterion")
  return(storage)
}

get.gmc.oversample <-function(gmc.model.pos, data.gswsvm.pos, oversample.ratio, tuning.ratio){
  # generate synthetic samples from the learned Gaussian mixture model,
  # add them to the training dataset,
  # split the training dataset into a tuning dataset and a training dataset, using stratified sampling
  # (stratified in terms of synthetic/original, and inside synthetic samples, stratified in terms of groups)
  
  G <- gmc.model.pos$G; #learned groups. The number of groups is determined by 
  d <- gmc.model.pos$d; #dimension of the x variable. In our case, d = 2.
  prob <- gmc.model.pos$parameters$pro # learned group membership probabilities
  means <- gmc.model.pos$parameters$mean #learned group means
  vars <- gmc.model.pos$parameters$variance$sigma #learned group variances
  
  
  #1. Oversample using the learned GMC model
  n.oversample <- round(length(data.gswsvm.pos$y) * oversample.ratio)
  
  gmc.index <- sample(x = 1:G, size = n.oversample, replace = T, prob = prob) #randomly assign group, according to the learned group membership probability.
  data.gmc <- data.frame(matrix(NA, n.oversample, d + 2)) #initialize the oversampled data storing matrix
  colnames(data.gmc) <- c("x1", "x2", "group", "y")
  
  
  #2. generate samples
  for(i in 1 : n.oversample) {
    data.gmc[i,c("x1", "x2")] <- rmvnorm(1, mean = means[ , gmc.index[i]],sigma=vars[,,gmc.index[i]])
    data.gmc[i,"group"] <- gmc.index[i]
    data.gmc[i, "y"] <- "pos"
  }
  data.gmc$y <- factor(data.gmc$y, levels = c("neg", "pos")) # turn the y variable into a factor
  
  # 3. split into training and tuning set, stratified w.r.t. group membership
  idx.split.gmc <- createDataPartition(data.gmc$group, p = tuning.ratio)
  data.gmc.train <- data.gmc[-idx.split.gmc$Resample1, ]
  data.gmc.tune <- data.gmc[idx.split.gmc$Resample1, ]
  
  data.gmc.train <- data.gmc.train[c("x1", "x2", "y")]  # remove group variable
  data.gmc.tune <- data.gmc.tune[c("x1", "x2", "y")]   # remove group variable
  return(list("data.gmc.train" = data.gmc.train, "data.gmc.tune" = data.gmc.tune))
} # end of the funciton get.gmc.oversample
 
split.with.oversample.stratified <- function(data.oversample.pos, tuning.ratio){
  #3. split the training data into training and tuning set the specified tuning set ratio
  ### 
  
  ## 3.1.  split the synthetic positive samples

  
  ## 3.2. split the original samples
  idx.split.gswsvm <- createDataPartition(data.original$y, p = tuning.ratio)
  data.plus.oversample.train <- data.original[-idx.split.gswsvm$Resample1, ]
  data.plus.oversample.tune <- data.original[idx.split.gswsvm$Resample1, ]
  
  ## 3.3. combine original and synthetic
  data.plus.oversample.train <- rbind(data.oversample.pos.train, data.plus.oversample.train)
  data.plus.oversample.tune <- rbind(data.oversample.pos.tune, data.plus.oversample.tune)
  
  return(list("data.plus.oversample.train" = data.plus.oversample.train, "data.plus.oversample.tune" = data.plus.oversample.tune))
} # end of the funciton split.with.oversample.stratified




