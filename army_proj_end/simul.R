setwd("/Users/mac/Documents/GitHub/gswsvm/army_proj_end")
getwd()
library(WeightSVM)
library(mclust)
library(mvtnorm)
library(caret)
library(e1071)
library(SwarmSVM)
library(rayshader)

source("data_generator.R")
source("bayes_rule.R")

#################################
# Step 1. parameter setting
#################################
set.seed(1)
## note: In our paper, positive class = minority and negative class = majority.
## 1.1. misclassification cost ratio
c.neg <- 4
c.pos <- 1
cost.ratio <- c.neg / c.pos

## 1.2. data generation parameters
### 1.2.1. data generation imbalance ratio
imbalance.ratio <- 6
pi.pos <- 1 / (1 + imbalance.ratio) # probability of a positive sample being generated
pi.neg <- 1 - pi.pos # probability of a negative sample being generated

### 1.2.2. sampling imbalance ratio(i.e. imbalance ratio after SMOTE)
### since the performance may vary w.r.t to this quantity,
### we treat this as s hyperparameter and
pi.s.pos <- c(pi.pos*1.25, pi.pos*1.5, pi.pos*1.75, pi.pos*2)
pi.s.neg <- 1 - pi.s.pos
imbalance.ratio.s <-  pi.s.neg / pi.s.pos
oversample.ratio = (pi.s.pos / pi.pos) - 1

L.pos = c.neg * pi.s.neg * pi.pos #vector
L.neg = c.pos * pi.s.pos * pi.neg #vector



#data generation
n.samples = 1000

#checkerboard data
p.mean1 <- c(-2,-5);
p.mean2 <- c(8,-5);
p.mean3 <- c(-7,0);
p.mean4 <- c(3,0);
p.mean5 <- c(-2,5);
p.mean6 <- c(8,5);
p.mus <- rbind(p.mean1, p.mean2, p.mean3, p.mean4, p.mean5, p.mean6)
p.sigma <- matrix(c(2.5,0,0,2.5),2,2)

n.mean1 <- c(-7,-5)
n.mean2 <- c(3,-5);
n.mean3 <- c(-2,0);
n.mean4 <- c(8,0);
n.mean5 <- c(-7,5);
n.mean6 <- c(2,5);



n.mus <- rbind(n.mean1,n.mean2,n.mean3,n.mean4, n.mean5, n.mean6)
n.sigma <- matrix(c(4,0,0,4),2,2)

param.set.c = 2^(-5 : 5); 
param.set.gamma = 2^(-5 : 5);

#simulation parameters
replication <- 2
n.method <- 6

#################################
# Step 2. simulation(monte carlo)
#################################

# setting storing variable
svm.acc <- matrix(NA, replication,n.method)
svm.sen <- matrix(NA, replication,n.method)
svm.pre <- matrix(NA, replication,n.method)
svm.spe <- matrix(NA, replication,n.method)
svm.gme <- matrix(NA, replication,n.method)

svm.cost <- matrix(NA, replication,n.method);
svm.gamma <- matrix(NA, replication,n.method);



for (rep in 1:replication){# why start with 3?
## 2.1. Prepare a dataset

### 2.1.1.generate full dataset
data.full = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.samples)
data.range <- list(x1.max = max(data.full[1]), x1.min = min(data.full[1]), x2.max = max(data.full[2]), x2.min = min(data.full[2]))

### 2.1.2. split the dataset into training set and testing set by 8:2 strafitied sampling
idx.split.test <- createDataPartition(data.full$y, p = 1/4)
data.train <- data.full[-idx.split.test$Resample1, ]
data.test <- data.full[idx.split.test$Resample1, ]

# 2.1.3 try different methods


#################################################################################
# Method 1. GS-WSVM
#################################################################################
n.model=1

# 1. Apply gmc-smote to the positive class

## 1.1. copy the training dataset, since GS-WSVM procedure would modify the training dataset.
data.gswsvm <- data.train 
data.gswsvm.pos <- data.gswsvm[data.gswsvm$y == 'pos', ]
data.gswsvm.neg <- data.gswsvm[data.gswsvm$y == 'neg', ]

## 1.2. Learn Gaussian Mixture Cluster model
gmc.model.pos <- Mclust(data.gswsvm.pos[,-3]) # 
G <- gmc.model.pos$G; #learned groups. The number of groups is determined by 
d <- gmc.model.pos$d; #dimension of the x variable. In our case, d=2.
prob <- gmc.model.pos$parameters$pro # learned group membership probabilities
means <- gmc.model.pos$parameters$mean #learned group means
vars <- gmc.model.pos$parameters$variance$sigma #learned group variances

## 1.3. TUNING
tuning.criterion.values = data.frame()
for (k in 1:length(pi.s.pos)){ #loop over pi.s.pos
  
  ### 1.3.1 Oversample using the learned GMC model
  n.oversample <- round(length(data.gswsvm.pos$y) * oversample.ratio[k])
  gmc.index <- sample(x = 1:G, size = n.oversample, replace = T, prob = prob)
  data.gmc.x <- matrix(0, n.oversample, d + 1)
  
  for(i in 1 : n.oversample) {
    data.gmc.x[i,1:2] <- rmvnorm(1, mean = means[ , gmc.index[i]],sigma=vars[,,gmc.index[i]])
    data.gmc.x[i,3] <- gmc.index[i]
  }
  
  data.gmc <- data.frame(data.gmc.x, rep("pos", n.oversample))
  colnames(data.gmc) <- c("x1", "x2", "group", "y")
  
  ### 1.3.2. split the training data into training and tuning set by 3:1 stratified sampling
  ### stratified in terms of both synthetic/original and positive/negative
  
  #### 1.3.2.1.  split the synthetic positive samples
  print("1.3.2.1")
  idx.split.gmc <- createDataPartition(data.gmc$group, p = 3/4)
  data.gmc.train <- data.gmc[idx.split.gmc$Resample1, ]
  data.gmc.tune <- data.gmc[-idx.split.gmc$Resample1, ]
  
  data.gmc.train = data.gmc.train[-3]  # remove group variable
  data.gmc.tune = data.gmc.tune[-3]   # remove group variable
  
  #### 1.3.2.2. split the original samples
  print("1.3.2.2")
  idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = 3/4)
  data.gswsvm.train <- data.gswsvm[idx.split.gswsvm$Resample1, ]
  data.gswsvm.tune <- data.gswsvm[-idx.split.gswsvm$Resample1, ]
  
  #### 1.3.2.3. combine original and synthetic
  print("1.3.2.3")
  data.gswsvm.train <- rbind(data.gswsvm.train, data.gmc.train)
  data.gswsvm.tune <- rbind(data.gswsvm.tune, data.gmc.tune)
  
  L.vector.tune = (data.gswsvm.tune$y == "pos") * L.pos[k] + (data.gswsvm.tune$y == "neg") * L.neg[k]
  L.vector.train = (data.gswsvm.train$y == "pos") * L.pos[k] + (data.gswsvm.train$y == "neg") * L.neg[k]

  
  tuning.criterion.values.now <- matrix(NA, nrow= length(param.set.gamma)*length(param.set.c), ncol = 4 )
  for (i in 1:length(param.set.gamma)){ #loop over gamma
    for (j in 1:length(param.set.c)){ #loop over c
      row.idx.now <- (i-1) * length(param.set.c) + j

      gamma.now <- param.set.gamma[i]
      c.now <- param.set.c[j]
      
      model.now <- wsvm(y ~ ., weight = L.vector.train, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.gswsvm.train)
      y.pred.now <- predict(model.now, data.gswsvm.tune[1:2]) #f(x_i) value
      tuning.criterion.values.now[row.idx.now, 1:3] = c(c.now, gamma.now, pi.s.pos[k])
      tuning.criterion.values.now[row.idx.now, 4] <- sum((y.pred.now != data.gswsvm.tune$y) * L.vector.tune)/(length(data.gswsvm.tune$y))
      }} #end for two for loops
  
  tuning.criterion.values.now <- data.frame(tuning.criterion.values.now)
  colnames(tuning.criterion.values.now) <- c("c","gamma","pi.s.pos","criterion")
  tuning.criterion.values <- rbind(tuning.criterion.values, tuning.criterion.values.now)
  } #end of pi.s.pos loop

idx.sorting <- order(tuning.criterion.values$criterion, tuning.criterion.values$c, tuning.criterion.values$gamma)
tuning.criterion.values <- tuning.criterion.values[idx.sorting, ]

# get the best parameters
param.gswsvm.c <- tuning.criterion.values[1,1]
param.gswsvm.gamma <- tuning.criterion.values[1,2]
pi.s.pos.gswsvm <- tuning.criterion.values[1,3]


# 1.4. with the best hyperparameter, fit the gs-wsvm
print("1.4")
# 1.4.1. set parameters

pi.s.neg.gswsvm <- 1 - pi.s.pos.gswsvm
oversample.ratio.gswsvm <- (pi.s.pos.gswsvm / pi.pos) - 1
n.oversample.gswsvm <- round(length(data.gswsvm.pos$y) * oversample.ratio.gswsvm)

L.pos.gswsvm <- c.neg * pi.s.neg.gswsvm * pi.pos #vector
L.neg.gswsvm <- c.pos * pi.s.pos.gswsvm * pi.neg #vector

### 1.4.2 Oversample using the learned GMC model
gmc.index <- sample(x = 1:G, size = n.oversample.gswsvm, replace = T, prob = prob)
data.gmc.x <- matrix(0, n.oversample.gswsvm, d + 1)

for(i in 1 : n.oversample.gswsvm) {
  data.gmc.x[i,1:2] <- rmvnorm(1, mean = means[ , gmc.index[i]],sigma=vars[,,gmc.index[i]])
  data.gmc.x[i,3] <- gmc.index[i]
}

data.gmc <- data.frame(data.gmc.x, rep("pos", n.oversample.gswsvm))
colnames(data.gmc) <- c("x1", "x2", "group", "y")

### 1.4.3. split the training data into training and tuning set by 3:1 stratified sampling
### stratified in terms of both synthetic/original and positive/negative

#### 1.4.3.1.  split the synthetic positive samples
idx.split.gmc <- createDataPartition(data.gmc$group, p = 3/4)
data.gmc.train <- data.gmc[idx.split.gmc$Resample1, ]
data.gmc.tune <- data.gmc[-idx.split.gmc$Resample1, ]

data.gmc.train <- data.gmc.train[-3]  # remove group variable
data.gmc.tune <- data.gmc.tune[-3]   # remove group variable

#### 1.4.3.2. split the original samples
idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = 3/4)
data.gswsvm.train <- data.gswsvm[idx.split.gswsvm$Resample1, ]
data.gswsvm.tune <- data.gswsvm[-idx.split.gswsvm$Resample1, ]

#### 1.4.3.3. combine original and synthetic
data.gswsvm.train <- rbind(data.gswsvm.train, data.gmc.train)
data.gswsvm.tune <- rbind(data.gswsvm.tune, data.gmc.tune)

L.vector.tune = (data.gswsvm.tune$y == "pos") * L.pos.gswsvm + (data.gswsvm.tune$y == "neg") * L.neg.gswsvm
L.vector.train = (data.gswsvm.train$y == "pos") * L.pos.gswsvm + (data.gswsvm.train$y == "neg") * L.neg.gswsvm

gswsvm.model <- wsvm(y ~ ., weight = L.vector.train, gamma = param.gswsvm.gamma, cost = param.gswsvm.c, kernel="radial", scale = FALSE, data = data.gswsvm.train)
gswsvm.pred <- predict(gswsvm.model, data.test[1:2])

svm.cmat=t(table(gswsvm.pred, data.test$y))
svm.cmat


svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])

# method 2 - x do not involve resampling,
# so they share the same training and tuning set.
#################################################################################
# Method 2. Standard SVM
#################################################################################
n.model = 2
# 2.1. split the training data into training and tuning set by 3:1 stratified sampling
data.svm <- data.train 
idx.split.svm <- createDataPartition(data.svm$y, p = 3/4)
data.svm.train <- data.svm[idx.split.svm$Resample1, ]
data.svm.tune <- data.svm[-idx.split.svm$Resample1, ]

# 2.2. hyperparameter tuning w.r.t. g-mean
tuning.criterion.values.svm <- matrix(NA, nrow = length(param.set.c)*length(param.set.gamma), ncol = 3 )
tuning.criterion.values.svm <- data.frame(tuning.criterion.values.svm)
colnames(tuning.criterion.values.svm) <- c("c", "gamma", "criterion")
for (i in 1:length(param.set.c)){ #loop over gamma 
  for (j in 1:length(param.set.gamma)){ #loop over c
    row.idx.now <- (i-1) * length(param.set.c) + j

    c.now <- param.set.c[i]
    gamma.now <- param.set.gamma[j]
    
    model.now <- svm(y ~ ., gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.svm.train)
    
    y.pred.now <- predict(model.now, data.svm.tune[1:2]) #f(x_i) value
    cmat <- t(table(y.pred.now, data.svm.tune$y))
    sen <- cmat[2,2]/sum(cmat[2,])
    spe <- cmat[1,1]/sum(cmat[1,])
    gme <- sqrt(sen*spe)
    
    
    tuning.criterion.values.svm[row.idx.now, 1:2] <- c(c.now, gamma.now)
    tuning.criterion.values.svm[row.idx.now, 3] <- gme
      }} #end for two for loops

idx.sorting <- order(-tuning.criterion.values.svm$criterion, tuning.criterion.values.svm$c, tuning.criterion.values.svm$gamma)
tuning.criterion.values.svm <- tuning.criterion.values.svm[idx.sorting, ]
param.svm.c <- tuning.criterion.values.svm[1,1]
param.svm.gamma <- tuning.criterion.values.svm[1,2]

#fit and evalutate performance on the test set
svm.model <- svm(y~., data=data.svm.train, kernel="radial", gamma=param.svm.gamma, cost=param.svm.c)

svm.pred <- predict(svm.model, data.test[1:2])

svm.cmat=t(table(svm.pred, data.test$y))
svm.cmat
svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])

svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])


#################################################################################
# Method 3. SVMDC
#################################################################################
# Reference: K. Veropoulos, C. Campbell, and N. Cristianini, 
# “Controlling the sensi-tivity of support vector machines,”
# in Proc. Int. Joint Conf. Artif. Intell.,Stockholm, Sweden, 1999, pp. 55–60.

n.model = 3
# 3.1. The class weights are given based on the method of the paper above:
# negative class : # of positive training samples / # of total training samples
# positive class : # of negative training samples / # of total training samples
weight.svmdc.neg <- sum(data.svm.train$y == 'pos') / length(data.svm.train$y)
weight.svmdc.pos <- sum(data.svm.train$y == 'neg') / length(data.svm.train$y)

weight.svmdc <- weight.svmdc.pos * (data.svm.train$y == 'pos') + weight.svmdc.neg * (data.svm.train$y == 'neg')

# 3.2. hyperparameter tuning w.r.t. g-mean
tuning.criterion.values.svmdc <- matrix(NA, nrow = length(param.set.c)*length(param.set.gamma), ncol = 3 )
tuning.criterion.values.svmdc <- data.frame(tuning.criterion.values.svmdc)
colnames(tuning.criterion.values.svmdc) <- c("c", "gamma", "criterion")
for (i in 1:length(param.set.c)){ #loop over gamma
  for (j in 1:length(param.set.gamma)){ #loop over c
    row.idx.now <- (i-1) * length(param.set.c) + j
    
    c.now <- param.set.c[i]
    gamma.now <- param.set.gamma[j]
    
    model.now <- wsvm(y ~ .,  weight = weight.svmdc, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.svm.train)
    
    y.pred.now <- predict(model.now, data.svm.tune[1:2]) #f(x_i) value
    cmat <- t(table(y.pred.now, data.svm.tune$y))
    sen <- cmat[2,2]/sum(cmat[2,])
    spe <- cmat[1,1]/sum(cmat[1,])
    gme <- sqrt(sen*spe)
    
    tuning.criterion.values.svmdc[row.idx.now, 1:2] <- c(c.now, gamma.now)
    tuning.criterion.values.svmdc[row.idx.now, 3] <- gme
  }} #end for two for loops

idx.sorting <- order(-tuning.criterion.values.svmdc$criterion, tuning.criterion.values.svmdc$c, tuning.criterion.values.svmdc$gamma)
tuning.criterion.values.svmdc <- tuning.criterion.values.svmdc[idx.sorting, ]
param.svmdc.c <- tuning.criterion.values.svmdc[1,1]
param.svmdc.gamma <- tuning.criterion.values.svmdc[1,2]

#fit and evaluate performance on the test set
svmdc.model <- wsvm(y~., weight = weight.svmdc, data = data.svm.train, kernel="radial", gamma=param.svmdc.gamma, cost=param.svmdc.c)

svmdc.pred <- predict(svmdc.model, data.test[1:2])

svm.cmat <- table(data.test$y, svmdc.pred)
svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])

svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])

#################################################################################
# Method 4. ClusterSVM
#################################################################################
# Reference: Gu, Q., & Han, J, 
# “Clustered support vector machines”.
# Journal of Machine Learning Research, 2013, 31, 307-315.

n.model = 4

# 4.1. The number of cluster should be provided: we use the result of GMC model learned.
param.clusterSVM.k <- G

# 4.2. hyperparameter tuning w.r.t. g-mean
param.set.clusteSVM.lambda <- c(1,5,10,20,50,100) #same as the reference

tuning.criterion.values.clusterSVM <- matrix(NA, nrow = length(param.set.c)*length(param.set.clusteSVM.lambda), ncol = 3 )
tuning.criterion.values.clusterSVM <- data.frame(tuning.criterion.values.clusterSVM)
colnames(tuning.criterion.values.clusterSVM) <- c("c", "lambda", "criterion")
for (i in 1:length(param.set.c)){ #loop over gamma 
  for (j in 1:length(param.set.clusteSVM.lambda)){ #loop over c
    row.idx.now <- (i-1) * length(param.set.c) + j
    
    c.now <- param.set.c[i]
    lambda.now <- param.set.clusteSVM.lambda[j]
    
    model.now <- clusterSVM(x = data.svm.train[1:2], y = data.svm.train$y, lambda = lambda.now, cost = c.now, centers = param.clusterSVM.k, seed = 512, verbose = 0) 
    
    y.pred.now = predict(model.now, data.svm.tune[1:2])$predictions
    cmat <- table(truth = data.svm.tune$y, pred = y.pred.now)
    sen <- cmat[2,2]/sum(cmat[2,])
    spe <- cmat[1,1]/sum(cmat[1,])
    gme <- sqrt(sen*spe)
    
    tuning.criterion.values.clusterSVM[row.idx.now, 1:2] <- c(c.now, lambda.now)
    tuning.criterion.values.clusterSVM[row.idx.now, 3] <- gme
  }} #end for two for loops

idx.sorting <- order(-tuning.criterion.values.clusterSVM$criterion, tuning.criterion.values.clusterSVM$c, tuning.criterion.values.clusterSVM$lambda)
tuning.criterion.values.clusterSVM <- tuning.criterion.values.clusterSVM[idx.sorting, ]
param.clusterSVM.c <- tuning.criterion.values.clusterSVM[1,1]
param.clusterSVM.lambda <- tuning.criterion.values.clusterSVM[1,2]

#fit and evaluate performance on the test set
clusterSVM.model <- clusterSVM(x = data.svm.train[1:2], y = data.svm.train$y, lambda = param.clusterSVM.lambda, cost = param.clusterSVM.c, centers = param.clusterSVM.k, seed = 512, verbose = 0) 
clusterSVM.pred = predict(clusterSVM.model, data.test[1:2])$predictions

svm.cmat=t(table(clusterSVM.pred, data.test$y))
svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])

svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])




} 


#replication bracket

#plot.basic <- draw.basic(data.train, col.p = "blue", col.n = "red", alpha.p = 0.3, alpha.n = 0.3)
#plot.bayes <-draw.bayes.rule(data.test, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
#plot.wsvm <- draw.svm.rule(data.test, gswsvm.model, color = 'green')
#plot.svm <- draw.svm.rule(data.test, svm.model, color = 'orange')

#plot.basic + plot.bayes + plot.wsvm

#plot.basic + plot.bayes


gswsvm.decision.function.grid <- get.svm.decision.values.grid(data.range, gswsvm.model)
faithful_dd <- ggplot(gswsvm.decision.function.grid, aes(x1, x2)) +
  geom_raster(aes(fill = z))
