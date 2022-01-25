library(caret)


create.tuning.criterion.storage <- function(param.set.list){
  n.param <- length(param.set.list)
  n.comb <- prod(sapply(param.set.list, length))
  storage <- matrix(NA, nrow = n.comb, ncol = n.param + 1 ) # last column is for criterion values
  storage <- data.frame(storage)
  colnames(storage) <- c(names(param.set.list), "criterion")
  return(storage)
}

get.gmc.oversample <-function(gmc.model.pos, data.gswsvm.pos, oversample.ratio){
  # generate synthetic samples from the learned Gaussian mixture model,
  # add them to the training dataset,
  
  #1. store useful values from the GMC result
  G <- gmc.model.pos$G; #learned groups. The number of groups is determined by 
  d <- gmc.model.pos$d; #dimension of the x variable. In our case, d = 2.
  prob <- gmc.model.pos$parameters$pro # learned group membership probabilities
  means <- gmc.model.pos$parameters$mean #learned group means
  vars <- gmc.model.pos$parameters$variance$sigma #learned group variances
  
  #2. Oversample using the learned GMC model
  n.oversample <- round(length(data.gswsvm.pos$y) * oversample.ratio)
  
  gmc.index <- sample(x = 1:G, size = n.oversample, replace = T, prob = prob) #randomly assign group, according to the learned group membership probability.
  data.gmc <- data.frame(matrix(NA, n.oversample, d + 1)) #initialize the oversampled data storing matrix
  variables.x <- colnames(data.gswsvm.pos)[-(d+1)]
  colnames(data.gmc) <- c(variables.x ,c("y"))
  
  #3. generate samples
  for(i in 1 : n.oversample) {
    data.gmc[i,variables.x] <- rmvnorm(1, mean = means[ , gmc.index[i]],sigma=vars[,,gmc.index[i]])
    data.gmc[i, "y"] <- "pos"
  }
  data.gmc$y <- factor(data.gmc$y, levels = c("neg", "pos")) # turn the y variable into a factor
  data.gmc.train <- data.gmc
  
  return(list("data.gmc.train" = data.gmc.train))
} # end of the funciton get.gmc.oversample
 

smote.and.split <- function(data.train.og, smote.samples, oversample.ratio, tuning.ratio){
  n.smote.samples <- dim(smote.samples)[1]
  n.pos.og <- sum( ( (data.train.og$y) == 'pos' ) )
  n.oversample <- round(n.pos.og * oversample.ratio) #calculate desired oversample size
  
  #1. randomly select n.oversample elements from the smote samples
  smote.samples.selected <- smote.samples[ sample(1:n.smote.samples, n.oversample, replace = FALSE), ]
  smote.samples.selected[,3] <- factor(smote.samples.selected[,3], levels = c("neg", "pos")); #smote function changes the datatype and name of the target variable; So we fix them.
  colnames(smote.samples.selected) <- c("x1","x2","y") 

  #2. split the original samples into a training set and a tuning set
  idx.split.og <- createDataPartition(data.train.og$y, p = tuning.ratio)
  data.train.og.train <- data.train.og[ -idx.split.og$Resample1, ]
  data.train.og.tune  <- data.train.og[ idx.split.og$Resample1, ]
  
  #3. combine the training set(**NOT the tuning set**) and synthetic samples 
  data.train.train.smoted <- rbind(smote.samples.selected, data.train.og.train)

  return(list("data.train.smoted" = data.train.train.smoted, "data.train.tune" = data.train.og.tune ))
  
  }









