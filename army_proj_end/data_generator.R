library(mvtnorm)

generate.mvn.mixture <- function(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, n.sample){
  # gernerate samples from mixtures of bivariate normal distribution(checkerboard)

  # vectorized algorithm to boost up the speed
  # input variables:
  #   p.mus = c_1 * 2 size matrix. Each row represents the center of each positive cluster. c_1 equals the number of positive clusters.
  #   n.mus = c_2 * 2 size matrix. Each row represents the center of each negative cluster. c_2 equals the number of negative clusters.
  #   imbalance.ratio = number of negative samples / number of positive samples.
  #   n.sample = number of total samples(both positive and negative).


  # implement prior probabilities defined by imbalance ratio
  #   index 0 means positive class, index 1:imbalance.ratio means nagative class
  p.pos = 1 / (imbalance.ratio + 1)
  
  # specify the number of samples for each class
  p.nsample <- sum(runif(n.sample)<p.pos) #number of positive samples, according to the imbalance ratio
  n.nsample <- n.sample - p.nsample #number of negative samples, according to the imbalance ratio
  print(p.nsample); print(n.nsample)
  # matrices for storing x variable(two columns, since it's bivariate normal)
  p.x.train <- matrix(0, p.nsample, 2)
  n.x.train <- matrix(0, n.nsample, 2)
  
  # generate bivariate normal mixture data(x1, x2) with uniform mixing probability

  # 1. For each sample, assign cluster membership with equal probabilities
  p.mu.indices <- sample(1 : nrow(p.mus), p.nsample, replace = TRUE, prob = NULL) # equal probability for each positive cluster
  n.mu.indices <- sample(1 : nrow(n.mus), n.nsample, replace = TRUE, prob = NULL) # equal probability for each negative cluster
  
  p.mu.indices.count <- table(p.mu.indices) # sum up the membership vector to get the membership frequency 
  n.mu.indices.count <- table(n.mu.indices)
  
  count.sum <- 0 # this will be used to keep track of the starting position of current cluster in the storage matrix.
  for (index in names(p.mu.indices.count)){ # for each positive cluster:
    count.now <- as.numeric(p.mu.indices.count[index]) #number of samples of this cluster
    mu <- p.mus[as.numeric(index), ] # get the center vector of this cluster
    sample <- rmvnorm(n = count.now, mean = mu, sigma = p.sigma) # generate random samples for this cluster
    p.x.train[(count.sum + 1) : (count.sum + count.now), ] <- sample # store the genereated sample in the matrix
    count.sum <- count.sum + count.now #update the starting position  
  }

  #do the same thing on the negative samples
  count.sum <- 0
  for (index in names(n.mu.indices.count)){
    count.now <- as.numeric(n.mu.indices.count[index])
    mu <- n.mus[as.numeric(index), ]
    sample <- rmvnorm(n = count.now, mean = mu, sigma = n.sigma)
    n.x.train[(count.sum + 1) : (count.sum + count.now), ] <- sample
    count.sum <- count.sum + count.now
  }
  x.train = rbind(p.x.train, n.x.train)
  
  # generate label(y) data
  p.y.train <- rep("pos", p.nsample)
  n.y.train <- rep("neg", n.nsample)
  y.train <- factor(c(p.y.train, n.y.train), levels = c("neg", "pos"))
  
  #combine x and y and return the dataset
  train <- data.frame(x.train, y.train)
  colnames(train) <- c("x1", "x2", "y")
  return(train)
}










