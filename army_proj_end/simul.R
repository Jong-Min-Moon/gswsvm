setwd("/Users/mac/Documents/GitHub/gswsvm/army_proj_end")
getwd()
library(WeightSVM) # for svm with instatnce-wise differing C
library(mclust) # for Gaussian mixture model
library(mvtnorm)
library(caret) # for data splitting
library(e1071) # for svm
library(SwarmSVM) # for clusterSVM
library(smotefamily) # for smote algorithms


source("data_generator.R")
source("bayes_rule.R")
source("zsvm.R")
source("simplifier.R")
#################################
# Step 1. parameter setting
#################################

# 1.1. simulation parameters
replication <- 30
n.method <- 10
use.method <- list("gswsvm3"= 1, "gswsvm" = 0, "svm" = 1, "svmdc" = 0, "clusterSVM" = 0, "zsvm" = 0, "smotesvm" = 1)
set.seed(2021)
tuning.ratio <- 1/4

## note: In our paper, positive class = minority and negative class = majority.
## 1.1. misclassification cost ratio
c.neg <- 4
c.pos <- 1
cost.ratio <- c.neg / c.pos
cost.ratio.og.syn <- 3
## 1.2. data generation parameters
n.samples = 1000
### 1.2.1. data generation imbalance ratio
imbalance.ratio <- 6
pi.pos <- 1 / (1 + imbalance.ratio) # probability of a positive sample being generated
pi.neg <- 1 - pi.pos # probability of a negative sample being generated

### 1.2.2. sampling imbalance ratio(i.e. imbalance ratio after SMOTE)
### since the performance may vary w.r.t to this quantity,
### we treat this as s hyperparameter and
pi.s.pos <- c(pi.pos*1.25, pi.pos*1.5, pi.pos*1.75, pi.pos*2, pi.pos*2.25, pi.pos*2.5)
pi.s.neg <- 1 - pi.s.pos
imbalance.ratio.s <-  pi.s.neg / pi.s.pos
oversample.ratio = (pi.s.pos / pi.pos) - 1

synthetic.within.pos.ratio <- oversample.ratio / (1 + oversample.ratio)
c.syn <- c.neg / (cost.ratio.og.syn - cost.ratio.og.syn * synthetic.within.pos.ratio + synthetic.within.pos.ratio)
c.og <- cost.ratio.og.syn * c.syn

L.pos <- c.neg * pi.s.neg * pi.pos #vector
L.neg <- c.pos * pi.s.pos * pi.neg #vector

L.syn <- c.syn * pi.s.neg * pi.pos
L.og <- c.og * pi.s.neg * pi.pos




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
  print(cat(rep, "th run"))
## 2.1. Prepare a dataset

### 2.1.1.generate full dataset
data.full = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.samples)
data.range <- list(x1.max = max(data.full[1]), x1.min = min(data.full[1]), x2.max = max(data.full[2]), x2.min = min(data.full[2]))

### 2.1.2. split the dataset into training set and testing set by 8:2 strafitied sampling
idx.split.test <- createDataPartition(data.full$y, p = 1/4)
data.train <- data.full[ -idx.split.test$Resample1, ] # 1 - 1/4
data.test  <- data.full[  idx.split.test$Resample1, ] # 1/4

# 2.1.3 try different methods


#################################################################################
# Method 1. GS-WSVM3
#################################################################################
if (use.method$"gswsvm3"){ #use this method or NOT
n.model <- 1

# 1.1. Apply gmc-smote to the positive class

## 1.1.1 copy the training dataset, since oversampling would result to modified dataset.
data.gswsvm <- data.train 
data.gswsvm.pos <- data.gswsvm[data.gswsvm$y == 'pos', ]

## 1.1.2. Learn Gaussian Mixture Cluster model
gmc.model.pos <- Mclust(data.gswsvm.pos[,-3]) 

# 1.2. TUNING
tuning.criterion.values.gswsvm <- create.tuning.criterion.storage(list("pi.s.pos" = pi.s.pos, "c" = param.set.c, "gamma" = param.set.gamma))

for (k in 1 : length(pi.s.pos)){ #loop over pi.s.pos
  
## 1.2.1. Oversample using the learned GMC model, and split it into the training set and the tuning set
  data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm.pos, oversample.ratio[k], tuning.ratio)

### 1.2.1.1. split the original samples
  idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = tuning.ratio)
  data.gswsvm.train <- data.gswsvm[-idx.split.gswsvm$Resample1, ] # 1 - 1/4
  data.gswsvm.tune  <- data.gswsvm[ idx.split.gswsvm$Resample1, ] # 1/4
  
  L.vector.tune = (data.gswsvm.tune$y == "pos") * L.og[k] + (data.gswsvm.tune$y == "neg") * L.neg[k] #L function in Lin et al. paper
  L.vector.train = (data.gswsvm.train$y == "pos") * L.og[k] + (data.gswsvm.train$y == "neg") * L.neg[k] 

  ### 1.2.1.2. combine original and synthetic
  data.gswsvm.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm.train)
  data.gswsvm.tune <- rbind(data.gmc$"data.gmc.tune", data.gswsvm.tune)
  
  L.vector.tune  <- c(rep(L.syn[k], length(data.gmc$"data.gmc.tune"$y )), L.vector.tune)
  L.vector.train <- c(rep(L.syn[k], length(data.gmc$"data.gmc.train"$y)), L.vector.train)

#### 1.3.2.4 loop over c and gamma
  for (i in 1:length(param.set.c)){ #loop over c
    for (j in 1:length(param.set.gamma)){ #loop over gamma
      row.idx.now <- (k-1) * length(param.set.gamma)*length(param.set.c) + (i-1) * length(param.set.gamma) + j #set row index
      
      
      c.now <- param.set.c[i]
      gamma.now <- param.set.gamma[j]
      
      model.now <- wsvm(y ~ ., weight = L.vector.train, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.gswsvm.train)# fit weighted svm model
      
      y.pred.now <- predict(model.now, data.gswsvm.tune[c("x1", "x2")]) #fitted value for tuning dataset
      tuning.criterion.values.gswsvm[row.idx.now, c("pi.s.pos", "c", "gamma")] <- c(pi.s.pos[k], c.now, gamma.now)
      tuning.criterion.values.gswsvm[row.idx.now, "criterion"] <- sum((y.pred.now != data.gswsvm.tune$y) * L.vector.tune)/(length(data.gswsvm.tune$y)) #tuning criterion introduced in Lin et
      }} #end for two for loops
  } #end of pi.s.pos loop

#### 1.3.2.5. get the best parameters
idx.sorting <- order(tuning.criterion.values.gswsvm$criterion, tuning.criterion.values.gswsvm$c, tuning.criterion.values.gswsvm$gamma)
tuning.criterion.values.gswsvm <- tuning.criterion.values.gswsvm[idx.sorting, ]
param.best <- tuning.criterion.values.gswsvm[1,]
param.gswsvm.c <- param.best$"c"
param.gswsvm.gamma <- param.best$"gamma"
param.gswsvm.pi.s.pos <- param.best$"pi.s.pos"

# 1.4. with the best hyperparameter, fit the gs-wsvm

## 1.4.1. set parameters
param.gswsvm.pi.s.neg <- 1 - param.gswsvm.pi.s.pos
oversample.ratio.gswsvm <- (param.gswsvm.pi.s.pos / pi.pos) - 1

synthetic.within.pos.ratio.gsvm <- oversample.ratio.gswsvm / (1 + oversample.ratio.gswsvm)
c.syn.gsvm <- c.neg / (cost.ratio.og.syn - cost.ratio.og.syn * synthetic.within.pos.ratio.gsvm + synthetic.within.pos.ratio.gsvm)
c.og.gsvm <- cost.ratio.og.syn * c.syn.gsvm

L.syn.gswsvm <- c.syn.gsvm * param.gswsvm.pi.s.neg * pi.pos #vector
L.og.gswsvm <- c.og.gsvm * param.gswsvm.pi.s.neg * pi.pos #vector
L.neg.gswsvm <- c.pos * param.gswsvm.pi.s.pos * pi.neg #vector

### 1.4.2 Oversample using the learned GMC model
data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm.pos, oversample.ratio.gswsvm, tuning.ratio)

#### 1.4.3.2. split the original samples
idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = tuning.ratio)
data.gswsvm.train <- data.gswsvm[-idx.split.gswsvm$Resample1, ]
data.gswsvm.tune  <- data.gswsvm[ idx.split.gswsvm$Resample1, ]

L.vector.train = (data.gswsvm.train$y == "pos") * L.og.gswsvm + (data.gswsvm.train$y == "neg") * L.neg.gswsvm

#### 1.4.3.3. combine original and synthetic
data.gswsvm.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm.train)
data.gswsvm.tune <- rbind(data.gmc$"data.gmc.tune", data.gswsvm.tune)
L.vector.train <- c(rep(L.syn[k], length(data.gmc$"data.gmc.train"$y)), L.vector.train)

gswsvm.model <- wsvm(y ~ ., weight = L.vector.train, gamma = param.gswsvm.gamma, cost = param.gswsvm.c, kernel="radial", scale = FALSE, data = data.gswsvm.train)
gswsvm.pred <- predict(gswsvm.model, data.test[c("x1", "x2")])

svm.cmat=t(table(gswsvm.pred, data.test$y))
svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
}# use this method or NOT


#################################################################################
# Method 2. GS-WSVM
#################################################################################
if (use.method$"gswsvm"){ #use this method or NOT
  n.model <- 2
  
  # 1. Apply gmc-smote to the positive class
  
  ## 1.1. copy the training dataset, since oversampling would result to modified dataset.
  data.gswsvm <- data.train 
  data.gswsvm.pos <- data.gswsvm[data.gswsvm$y == 'pos', ]
  
  ## 1.2. Learn Gaussian Mixture Cluster model
  gmc.model.pos <- Mclust(data.gswsvm.pos[,-3]) # 
  
  
  ## 1.3. TUNING
  tuning.criterion.values.gswsvm <- create.tuning.criterion.storage(list("pi.s.pos" = pi.s.pos, "c" = param.set.c, "gamma" = param.set.gamma))
  
  for (k in 1 : length(pi.s.pos)){ #loop over pi.s.pos
    
    ### 1.3.1.1 Oversample using the learned GMC model, and split it into the training set and the tuning set
    data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm.pos, oversample.ratio[k], tuning.ratio)
    
    #### 1.3.1.2. split the original samples
    idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = tuning.ratio)
    data.gswsvm.train <- data.gswsvm[-idx.split.gswsvm$Resample1, ] # 1- 1/4
    data.gswsvm.tune  <- data.gswsvm[ idx.split.gswsvm$Resample1, ] # 1/4
    
    #### 1.3.2.3. combine original and synthetic
    data.gswsvm.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm.train)
    data.gswsvm.tune <- rbind(data.gmc$"data.gmc.tune", data.gswsvm.tune)
    
    L.vector.tune  <- (data.gswsvm.tune$y == "pos") * L.pos[k] + (data.gswsvm.tune$y == "neg") * L.neg[k]
    L.vector.train <- (data.gswsvm.train$y == "pos") * L.pos[k] + (data.gswsvm.train$y == "neg") * L.neg[k]
    
    #### 1.3.2.4 loop over c and gamma
    for (i in 1:length(param.set.c)){ #loop over c
      for (j in 1:length(param.set.gamma)){ #loop over gamma
        row.idx.now <- (k-1) * length(param.set.gamma)*length(param.set.c) + (i-1) * length(param.set.gamma) + j #set row index
        
        
        c.now <- param.set.c[i]
        gamma.now <- param.set.gamma[j]
        
        model.now <- wsvm(y ~ ., weight = L.vector.train, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.gswsvm.train)# fit weighted svm model
        
        y.pred.now <- predict(model.now, data.gswsvm.tune[c("x1", "x2")]) #fitted value for tuning dataset
        tuning.criterion.values.gswsvm[row.idx.now, c("pi.s.pos", "c", "gamma")] <- c(pi.s.pos[k], c.now, gamma.now)
        tuning.criterion.values.gswsvm[row.idx.now, "criterion"] <- sum((y.pred.now != data.gswsvm.tune$y) * L.vector.tune)/(length(data.gswsvm.tune$y)) #tuning criterion introduced in Lin et
      }} #end for two for loops
  } #end of pi.s.pos loop
  
  #### 1.3.2.5. get the best parameters
  idx.sorting <- order(tuning.criterion.values.gswsvm$criterion, tuning.criterion.values.gswsvm$c, tuning.criterion.values.gswsvm$gamma)
  tuning.criterion.values.gswsvm <- tuning.criterion.values.gswsvm[idx.sorting, ]
  param.best <- tuning.criterion.values.gswsvm[1,]
  param.gswsvm.c <- param.best$"c"
  param.gswsvm.gamma <- param.best$"gamma"
  param.gswsvm.pi.s.pos <- param.best$"pi.s.pos"
  
  # 1.4. with the best hyperparameter, fit the gs-wsvm
  
  ## 1.4.1. set parameters
  param.gswsvm.pi.s.neg <- 1 - param.gswsvm.pi.s.pos
  oversample.ratio.gswsvm <- (param.gswsvm.pi.s.pos / pi.pos) - 1
  
  L.pos.gswsvm <- c.neg * param.gswsvm.pi.s.neg * pi.pos #vector
  L.neg.gswsvm <- c.pos * param.gswsvm.pi.s.pos * pi.neg #vector
  
  ### 1.4.2 Oversample using the learned GMC model
  data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm.pos, oversample.ratio.gswsvm, tuning.ratio)
  
  #### 1.4.3.2. split the original samples
  idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = tuning.ratio)
  data.gswsvm.train <- data.gswsvm[-idx.split.gswsvm$Resample1, ]
  data.gswsvm.tune  <- data.gswsvm[ idx.split.gswsvm$Resample1, ]
  
  #### 1.4.3.3. combine original and synthetic
  data.gswsvm.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm.train)
  data.gswsvm.tune <- rbind(data.gmc$"data.gmc.tune", data.gswsvm.tune)
  
  L.vector.train = (data.gswsvm.train$y == "pos") * L.pos.gswsvm + (data.gswsvm.train$y == "neg") * L.neg.gswsvm
  
  gswsvm.model <- wsvm(y ~ ., weight = L.vector.train, gamma = param.gswsvm.gamma, cost = param.gswsvm.c, kernel="radial", scale = FALSE, data = data.gswsvm.train)
  gswsvm.pred <- predict(gswsvm.model, data.test[c("x1", "x2")])
  
  svm.cmat=t(table(gswsvm.pred, data.test$y))
  svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
  svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
  svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
  svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
}# use this method or NOT


# method 2 - x do not involve resampling,
# so they share the same training and tuning set.
#################################################################################
# Method 3. Standard SVM
#################################################################################
if (use.method$"svm"){ #use this method or NOT
n.model <- 3
# 2.1. split the training data into training and tuning set by 3:1 stratified sampling
data.svm <- data.train 
idx.split.svm <- createDataPartition(data.svm$y, p = tuning.ratio)
data.svm.train <- data.svm[-idx.split.svm$Resample1, ] # 1 - 1/4
data.svm.tune  <- data.svm[ idx.split.svm$Resample1, ] # 1/4

# 2.2. hyperparameter tuning w.r.t. g-mean
tuning.criterion.values.svm <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))

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
svm.model <- svm(y~., data = data.svm.train, kernel="radial", gamma=param.svm.gamma, cost=param.svm.c)

svm.pred <- predict(svm.model, data.test[1:2])

svm.cmat=t(table(svm.pred, data.test$y))
svm.cmat
svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])

svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
} #use this method or NOT
  

#################################################################################
# Method 4. SVMDC
#################################################################################
# Reference: K. Veropoulos, C. Campbell, and N. Cristianini, 
# “Controlling the sensi-tivity of support vector machines,”
# in Proc. Int. Joint Conf. Artif. Intell.,Stockholm, Sweden, 1999, pp. 55–60.
if (use.method$"svmdc"){ #use this method or NOT
n.model <- 4
# 3.1. The class weights are given based on the method of the paper above:
# negative class : # of positive training samples / # of total training samples
# positive class : # of negative training samples / # of total training samples
weight.svmdc.neg <- sum(data.svm.train$y == 'pos') / length(data.svm.train$y)
weight.svmdc.pos <- sum(data.svm.train$y == 'neg') / length(data.svm.train$y)

weight.svmdc <- weight.svmdc.pos * (data.svm.train$y == 'pos') + weight.svmdc.neg * (data.svm.train$y == 'neg')

# 3.2. hyperparameter tuning w.r.t. g-mean
tuning.criterion.values.svmdc <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))

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
} #use this method or NOT

#################################################################################
# Method 5. ClusterSVM
#################################################################################
# Reference: Gu, Q., & Han, J, 
# “Clustered support vector machines”.
# Journal of Machine Learning Research, 2013, 31, 307-315.

if (use.method$"clusterSVM"){ #use this method or NOT
n.model <- 5

# 4.1. The number of cluster should be provided: we use the result of GMC model learned.
param.clusterSVM.k <- gmc.model.pos$G

# 4.2. hyperparameter tuning w.r.t. g-mean
param.set.clusteSVM.lambda <- c(1,5,10,20,50,100) #same as the reference

tuning.criterion.values.clusterSVM <- create.tuning.criterion.storage(list("c" = param.set.c, "lambda" = param.set.clusteSVM.lambda))

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



}#use this method or NOT


#################################################################################
# Method 6. z-SVM
#################################################################################
# Reference: Imam, T., Ting, K. M., Kamruzzaman, J. (2006).
# Z-SVM: An SVM for improved classification of imbalanced data. 
# Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 4304 LNAI.
n.model <- 6

if (use.method$"zsvm"){ #use this method or NOT
  
  data.zsvm.train <- data.svm.train
  
  # libsvm library used in e1071 sets the label of the first element as +1.
  # z-svm only scales up the Lagrange multipliers of the +1 class,
  # so we should a positive sample as the first instance of the training set 
  while ( (data.zsvm.train$y)[1] == "neg" ){
    data.zsvm.train <- data.zsvm.train[sample(1:length(data.zsvm.train$y), replace=FALSE) , ]
  } 
  
  
  param.set.zsvm.z <- c(1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6)
  tuning.criterion.values.zsvm <- matrix(NA, nrow = length(param.set.zsvm.z)* length(param.set.c)*length(param.set.gamma), ncol = 4 )
  tuning.criterion.values.zsvm <- data.frame(tuning.criterion.values.zsvm)
  colnames(tuning.criterion.values.zsvm) <- c("z", "c", "gamma", "criterion")
  
  for (i in 1:length(param.set.c)){ #loop over c
    for (j in 1:length(param.set.gamma)){ #loop over gamma
      c.now <- param.set.c[i]
      gamma.now <- param.set.gamma[j]
      model.now <- svm(y ~ ., data = data.svm.train, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)
      for (k in 1:length(param.set.zsvm.z)){ #loop over z
        row.idx.now <- (i-1) * length(param.set.zsvm.z)*length(param.set.c) + (j-1) * length(param.set.c) + k
        z.now = param.set.zsvm.z[k]
        
        y.pred.now <- zsvm.predict(data.svm.tune[1:2], model.now, gamma.now, z.now)
        cmat <- t(table(y.pred.now, data.svm.tune$y))
        sen <- cmat[2,2]/sum(cmat[2,])
        spe <- cmat[1,1]/sum(cmat[1,])
        gme <- sqrt(sen*spe)
        
        tuning.criterion.values.zsvm[row.idx.now, 1:3] <- c(z.now, c.now, gamma.now)
        tuning.criterion.values.zsvm[row.idx.now, 4] <- gme
      }}} #hyperparameter tuning bracket

  idx.sorting <- order(-tuning.criterion.values.zsvm$criterion, tuning.criterion.values.zsvm$c, tuning.criterion.values.zsvm$gamma, tuning.criterion.values.zsvm$z)
  tuning.criterion.values.zsvm <- tuning.criterion.values.zsvm[idx.sorting, ]
  param.best <- tuning.criterion.values.zsvm[1,]
  param.zsvm.z <- param.best$z
  param.zsvm.c <- param.best$c
  param.zsvm.gamma <- param.best$gamma
  
  #fit and evalutate performance on the test set
  zsvm.model <- svm(y ~., data = data.svm.train, gamma = param.zsvm.gamma, cost=param.zsvm.c, kernel="radial", scale = FALSE)
  zsvm.pred <- zsvm.predict(data.test[1:2], zsvm.model, param.zsvm.gamma, param.zsvm.z)
  svm.cmat=t(table(zsvm.pred, data.test$y))
  svm.cmat
  svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
  svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
  svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
  
  svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
  } #use this method or NOT bracket

} #replication bracket





#################################################################################
# Method 7. SMOTE
#################################################################################
# Reference: N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer,
# “SMOTE: Synthetic minority over-sampling technique,”
# J. Artif. Intell.Res., vol. 16, no. 1, pp. 321–357, 2002.
n.model <- 7

if (use.method$"smote"){ #use this method or NOT
  # 1. Apply smote to the positive class
  
  ## 1.1. copy the training dataset, since oversampling would result to modified dataset.
  data.smotesvm <- data.train 
  data.smotesvm.pos.idx <- rownames(data.train)[(data.smotesvm$y)=="pos"]
  
  ## 1.2. TUNING
  tuning.criterion.values.smotesvm <- create.tuning.criterion.storage(list("pi.s.pos" = pi.s.pos, "c" = param.set.c, "gamma" = param.set.gamma))
  
  for (k in 1 : length(pi.s.pos)){ #loop over pi.s.pos
    
    ### 1.2.1. Oversample using smote, and split into training and tuning set
    smote.sample = SMOTE(data.train[c("x1", "x2")], data.train["y"], dup_size = 0)

    data.plus.smote <- smote.and.split(data.smotesvm, smote.sample$syn_data, oversample.ratio[k], tuning.ratio)
    data.smotesvm.train <- data.plus.smote$"data.train.og.train"
    data.smotesvm.tune <- data.plus.smote$"data.train.og.tune"
    
    ### 1.2.2. loop over c and gamma
    for (i in 1:length(param.set.c)){ #loop over c
      for (j in 1:length(param.set.gamma)){ #loop over gamma
        row.idx.now <- (k-1) * length(param.set.gamma)*length(param.set.c) + (i-1) * length(param.set.gamma) + j #set row index
        
        c.now <- param.set.c[i]
        gamma.now <- param.set.gamma[j]
        
        model.now <- svm(y ~ ., gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.smotesvm.train)# fit weighted svm model
        
        y.pred.now <- predict(model.now, data.smotesvm.tune[c("x1", "x2")]) #fitted value for tuning dataset
        
        cmat <- table(data.smotesvm.tune$y, y.pred.now)
        sen <- cmat[2,2]/sum(cmat[2,])
        spe <- cmat[1,1]/sum(cmat[1,])
        gme <- sqrt(sen*spe)
        
        tuning.criterion.values.smotesvm[row.idx.now, c("pi.s.pos", "c", "gamma")] <- c(pi.s.pos[k], c.now, gamma.now)
        tuning.criterion.values.smotesvm[row.idx.now, "criterion"] <- gme
      }} #end for two for loops
  } #end of pi.s.pos loop
  
  #### 1.2.3. get the best parameters
  idx.sorting <- order(-tuning.criterion.values.smotesvm$criterion, tuning.criterion.values.smotesvm$c, tuning.criterion.values.smotesvm$gamma)
  tuning.criterion.values.smotesvm <- tuning.criterion.values.smotesvm[idx.sorting, ]
  param.best <- tuning.criterion.values.smotesvm[1,]
  param.smotesvm.c <- param.best$"c"
  param.smotesvm.gamma <- param.best$"gamma"
  param.smotesvm.pi.s.pos <- param.best$"pi.s.pos"
  
  # 1.4. with the best hyperparameter, fit the svm
  
  ## 1.4.1. set parameters
  param.smotesvm.pi.s.neg <- 1 - param.smotesvm.pi.s.pos
  oversample.ratio.smotesvm <- (param.smotesvm.pi.s.pos / pi.pos) - 1
  
  ### 1.4.2 Oversample using the learned GMC model
  data.plus.smote <- smote.and.split(data.smotesvm, smote.sample$syn_data, oversample.ratio.smotesvm, tuning.ratio)
  data.smotesvm.train <- data.plus.smote$"data.train.og.train"
  data.smotesvm.tune <- data.plus.smote$"data.train.og.tune"
  
  smotesvm.model <- svm(y ~ ., gamma = param.smotesvm.gamma, cost = param.smotesvm.c, kernel="radial", scale = FALSE, data = data.smotesvm.train)
  smotesvm.pred <- predict(smotesvm.model, data.test[c("x1", "x2")])
  
  svm.cmat=t(table(smotesvm.pred, data.test$y))
  svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
  svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
  svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
  svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
}

write.table(svm.gme,"gmeresult.csv")


#plot.basic <- draw.basic(data.test, col.p = "blue", col.n = "red", alpha.p = 0.3, alpha.n = 0.3)
#plot.bayes <-draw.bayes.rule(data.test, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
#plot.wsvm <- draw.svm.rule(data.test, gswsvm.model, color = 'green', cutoff = 0)
#plot.svm <- draw.svm.rule(data.test, svm.model, color = 'orange')

#plot.basic + plot.bayes + plot.wsvm
#plot.basic + plot.bayes


#################################################################################
# Bayes Rule
#################################################################################
#pred.bayes <- bayes.predict(data.test, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
#model.eval.bayes <- model.eval(test.y = data.test$y, pred.y = pred.bayes)
# 
# gswsvm.grid <- get.svm.decision.values.grid(data.range, gswsvm.model)
# gswsvm.heatmap <- ggplot(gswsvm.grid, aes(x1, x2)) +
#   geom_raster(aes(fill = z))
# plot_gg(gswsvm.heatmap, multicore = TRUE, width = 8, height = 8, scale = 300,
#         zoom = 0.6, phi = 60,
#         background = "#afceff",shadowcolor = "#3a4f70")
