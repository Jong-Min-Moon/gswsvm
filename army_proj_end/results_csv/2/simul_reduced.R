setwd("/Users/mac/Documents/GitHub/gswsvm/army_proj_end")
getwd()
library(WeightSVM) # for svm with instatnce-wise differing C
library(mclust) # for Gaussian mixture model
library(mvtnorm)
library(caret) # for data splitting
library(e1071) # for svm
library(SwarmSVM) # for clusterSVM
library(smotefamily) # for smote algorithms
library(gridExtra) 

source("data_generator.R")
source("bayes_rule.R")
source("simplifier.R")
#################################
# Step 1. parameter setting
############))#####################
start_time <- Sys.time() 

# 1.1. simulation parameters
replication <- 100
n.method <- 10
use.method <- list("gswsvm3"= 1, "gswsvm" = 0, "svm" = 0, "svmdc" = 0, "clusterSVM" = 0, "smotesvm" = 0, "blsmotesvm"= 0, "dbsmotesvm" = 0, "smotedc" = 1)

tuning.ratio <- 1/5
test.ratio <- 1/5

## note: In our paper, positive class = minority and negative class = majority.
## 1.1. misclassification cost ratio
# 여러 상황으로 실험 중...

## 1.2. data generation parameters
n.samples = 1000
### 1.2.1. data generation imbalance ratio

imbalance.ratios <-  c(10)

# saving matrices
imbal.gme <- matrix(NA, nrow = length(imbalance.ratios), ncol = n.method)
rownames(imbal.gme) <- imbalance.ratios
colnames(imbal.gme) <- c("gswsvm3", "gswsvm" , "svm" , "svmdc" , "clusterSVM" , "smotesvm" , "blsmotesvm", "dbsmotesvm", "smotedc", "bayes")

imbal.sen <- imbal.gme
imbal.spe <- imbal.gme
imbal.acc <- imbal.gme
imbal.pre <- imbal.gme


imbal.gme.sd <- imbal.gme
imbal.spe.sd <- imbal.gme
imbal.sen.sd <- imbal.gme
imbal.acc.sd <- imbal.gme
imbal.pre.sd <- imbal.gme

for (imbalance.ratio in imbalance.ratios){
cat("imbalance.ratio :",imbalance.ratio, "\n")
#imbalance.ratio <- 30 ## MAJOR PARAMETER


pi.pos <- 1 / (1 + imbalance.ratio) # probability of a positive sample being generated
pi.neg <- 1 - pi.pos # probability of a negative sample being generated

c.neg <- imbalance.ratio / 2
c.pos <- 1
cost.ratio <- c.neg / c.pos
cost.ratio.og.syn <- cost.ratio
### 1.2.2. sampling imbalance ratio(i.e. imbalance ratio after SMOTE)
### since the performance may vary w.r.t to this quantity,
### we treat this as s hyperparameter and
imbalance.ratio.s <- imbalance.ratio /4


pi.s.pos  <- 1 / (1 + imbalance.ratio.s)
pi.s.neg <- 1 - pi.s.pos


n.pos <- n.samples * pi.pos
n.neg <- n.pos * imbalance.ratio
n.neg.s <- n.neg

#n.neg.s/n.pos.s = imbalance.ratio.s
#n.pos.s = n.neg.s / imbalance.ratio.s

#n.pos.s <- n.pos * (1 + oversample.ratio)

oversample.ratio <- n.neg.s / imbalance.ratio.s / n.pos - 1
rSyn <- oversample.ratio / (1 + oversample.ratio)
c.syn <- c.neg / (cost.ratio.og.syn - cost.ratio.og.syn * rSyn + rSyn)
c.og <- cost.ratio.og.syn * c.syn

L.pos <- c.neg * pi.s.neg * pi.pos 
L.neg <- c.pos * pi.s.pos * pi.neg 

L.syn <- c.syn * pi.s.neg * pi.pos
L.og <- c.og * pi.s.neg * pi.pos




#checkerboard data
p.mean1 <- c(0.5,-5);
#p.mean2 <- c(8,-5);
p.mean3 <- c(-4.5,0);
p.mean4 <- c(5.5,0);
p.mean5 <- c(.5,5);
#p.mean6 <- c(8,5);
p.mus <- rbind(p.mean1, p.mean3, p.mean4, p.mean5)
p.sigma <- matrix(c(2.5,0,0,2.5),2,2)

n.mean1 <- c(-4.5,-5)
n.mean2 <- c(5.5,-5);
n.mean3 <- c(.5,0);
#n.mean4 <- c(8,0);
n.mean5 <- c(-4.5,5);
n.mean6 <- c(4.5,5);



n.mus <- rbind(n.mean1,n.mean2,n.mean3, n.mean5, n.mean6)
n.sigma <- matrix(c(3,0,0,3),2,2) 

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

models.learned <- list("gswsvm3"= NULL, "gswsvm" = NULL, "svm" = NULL, "svmdc" = NULL, "clusterSVM" = NULL, "smotesvm" = NULL, "blsmotesvm"= NULL, "dbsmotesvm" = NULL, "smotedc" = NULL)
cmat <- list("gswsvm3"= NULL, "gswsvm" = NULL, "svm" = NULL, "svmdc" = NULL, "clusterSVM" = NULL,  "smotesvm" = NULL, "blsmotesvm"= NULL, "dbsmotesvm" = NULL, "smotedc" = NULL)

for (rep in 1:replication){# why start with 3?
  cat(rep, "th run")
  set.seed(rep)
## 2.1. Prepare a dataset

### 2.1.1.generate full dataset
data.full = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.samples)
data.range <- list(x1.max = max(data.full[1]), x1.min = min(data.full[1]), x2.max = max(data.full[2]), x2.min = min(data.full[2]))

### 2.1.2. split the dataset into training set and testing set by 8:2 strafitied sampling
idx.split.test <- createDataPartition(data.full$y, p = test.ratio)
data.train <- data.full[ -idx.split.test$Resample1, ] # 1 - 1/4
data.test  <- data.full[  idx.split.test$Resample1, ] # 1/4

# 2.1.3 try different methods


#################################################################################
# Method 1. GS-WSVM3
#################################################################################

if (use.method$"gswsvm3"){ #use this method or NOT
n.model <- 1
set.seed(rep) # for reproducible result

## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods.
data.gswsvm3 <- data.train 

## 2. Hyperparamter tuning procedure

### 2.1. prepare a data.frame for storing the hyperparamter tuning results
tuning.criterion.values.gswsvm3 <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))

### 2.2. Split the original samples into a training set and a tuning set
idx.split.gswsvm3 <- createDataPartition(data.gswsvm3$y, p = tuning.ratio)
data.gswsvm3.train <- data.gswsvm3[-idx.split.gswsvm3$Resample1, ] # 1 - tuning.ratio
data.gswsvm3.tune  <- data.gswsvm3[ idx.split.gswsvm3$Resample1, ] # tuning.ratio
  
### 2.3. leran GMC model on the positive data
data.gswsvm3.train.pos <- data.gswsvm3.train[data.gswsvm3.train$y == "pos", ]
gmc.model.pos <- Mclust(data.gswsvm3.train.pos[c("x1", "x2")]) 
data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm3.train.pos, oversample.ratio)

## 2.3. L function in Lin et al.'s paper  
L.vector.tune.gswsvm3 = (data.gswsvm3.tune$y == "pos") * L.og + (data.gswsvm3.tune$y == "neg") * L.neg
L.vector.train.gswsvm3 = (data.gswsvm3.train$y == "pos") * L.og + (data.gswsvm3.train$y == "neg") * L.neg 

## 2.4. Combine original positive and synthetic positive
data.gswsvm3.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm3.train)
L.vector.train.gswsvm3 <- c(rep(L.syn, length(data.gmc$"data.gmc.train"$y)), L.vector.train.gswsvm3) # add L function values for synthetic samples
  
## 2.5. loop over c and gamma and calculate the tuning criterion(sample expected misclassification cost in Lin et al.'s paper)
for (i in 1:length(param.set.c)){ #loop over c
  for (j in 1:length(param.set.gamma)){ #loop over gamma
    row.idx.now <- (i-1) * length(param.set.gamma) + j #set row index
      
    c.now <- param.set.c[i]
    gamma.now <- param.set.gamma[j]
      
    model.now <- wsvm(data = data.gswsvm3.train, y ~ ., weight = L.vector.train.gswsvm3, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
      
    y.pred.now <- predict(model.now, data.gswsvm3.tune[c("x1", "x2")]) #fitted value for tuning dataset
    tuning.criterion.values.gswsvm3[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
    tuning.criterion.values.gswsvm3[row.idx.now, "criterion"] <- sum((y.pred.now != data.gswsvm3.tune$y) * L.vector.tune.gswsvm3)/(length(data.gswsvm3.tune$y)) #tuning criterion introduced in Lin et
      
      }} # end of two for loops

#### 1.2.6. get the best parameters
idx.sorting <- order(tuning.criterion.values.gswsvm3$criterion, tuning.criterion.values.gswsvm3$c, tuning.criterion.values.gswsvm3$gamma)
tuning.criterion.values.gswsvm3 <- tuning.criterion.values.gswsvm3[idx.sorting, ]
param.best.gswsvm3 <- tuning.criterion.values.gswsvm3[1,]

# 1.3. with the best hyperparameter, fit the gs-wsvm
svm.model.gswsvm3 <- wsvm(data = data.gswsvm3.train, y ~ .,
                                weight = L.vector.train.gswsvm3,
                                gamma = param.best.gswsvm3$"gamma",
                                cost = param.best.gswsvm3$"c",
                                kernel="radial", scale = FALSE)

pred.gswsvm3 <- predict(svm.model.gswsvm3, data.test[c("x1", "x2")])

svm.cmat <- table("truth" = data.test$y, "pred.gswsvm3" = pred.gswsvm3)
svm.acc[rep,n.model] <- (svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat) #accuracy
svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # sensitivity
svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model] * svm.spe[rep,n.model])

cmat.gswsvm3 <- svm.cmat
#cmat$"gswsvm3" <- svm.cmat #save confusion matrix for this model, for future analysis
}# use this method or NOT


#

#################################################################################
# Method 2. GS-WSVM
#################################################################################
if (use.method$"gswsvm"){ #use this method or NOT
n.model <- 2
set.seed(rep) # for reproducible result

## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods.
data.gswsvm <- data.train 

## 2. Hyperparamter tuning procedure

### 2.1. prepare a data.frame for storing the hyperparamter tuning results
tuning.criterion.values.gswsvm <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))

### 2.2. Split the original samples into a training set and a tuning set
idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = tuning.ratio)
data.gswsvm.train <- data.gswsvm[-idx.split.gswsvm$Resample1, ] # 1 - tuning.ratio
data.gswsvm.tune  <- data.gswsvm[ idx.split.gswsvm$Resample1, ] # tuning.ratio
  
### 2.3. leran GMC model on the positive data
data.gswsvm.train.pos <- data.gswsvm.train[data.gswsvm.train$y == "pos", ]
gmc.model.pos <- Mclust(data.gswsvm.train.pos[c("x1", "x2")]) 
data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm.train.pos, oversample.ratio)

## 2.3. L function in Lin et al.'s paper  
L.vector.tune.gswsvm = (data.gswsvm.tune$y == "pos") * L.og + (data.gswsvm.tune$y == "neg") * L.neg
L.vector.train.gswsvm = (data.gswsvm.train$y == "pos") * L.og + (data.gswsvm.train$y == "neg") * L.neg 

## 2.4. Combine original positive and synthetic positive
data.gswsvm.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm.train)
L.vector.train.gswsvm <- c(rep(L.syn, length(data.gmc$"data.gmc.train"$y)), L.vector.train.gswsvm) # add L function values for synthetic samples
  
## 2.5. loop over c and gamma and calculate the tuning criterion(sample expected misclassification cost in Lin et al.'s paper)
for (i in 1:length(param.set.c)){ #loop over c
  for (j in 1:length(param.set.gamma)){ #loop over gamma
    row.idx.now <- (i-1) * length(param.set.gamma) + j #set row index
      
    c.now <- param.set.c[i]
    gamma.now <- param.set.gamma[j]
      
    model.now <- wsvm(data = data.gswsvm.train, y ~ ., weight = L.vector.train.gswsvm, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
      
    y.pred.now <- predict(model.now, data.gswsvm.tune[c("x1", "x2")]) #fitted value for tuning dataset
    tuning.criterion.values.gswsvm[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
    tuning.criterion.values.gswsvm[row.idx.now, "criterion"] <- sum((y.pred.now != data.gswsvm.tune$y) * L.vector.tune.gswsvm)/(length(data.gswsvm.tune$y)) #tuning criterion introduced in Lin et
      
      }} # end of two for loops

#### 1.2.6. get the best parameters
idx.sorting <- order(tuning.criterion.values.gswsvm$criterion, tuning.criterion.values.gswsvm$c, tuning.criterion.values.gswsvm$gamma)
tuning.criterion.values.gswsvm <- tuning.criterion.values.gswsvm[idx.sorting, ]
param.best.gswsvm <- tuning.criterion.values.gswsvm[1,]

# 1.3. with the best hyperparameter, fit the gs-wsvm
svm.model.gswsvm <- wsvm(data = data.gswsvm.train, y ~ .,
                                weight = L.vector.train.gswsvm,
                                gamma = param.best.gswsvm$"gamma",
                                cost = param.best.gswsvm$"c",
                                kernel="radial", scale = FALSE)

pred.gswsvm <- predict(svm.model.gswsvm, data.test[c("x1", "x2")])

svm.cmat <- table("truth" = data.test$y, "pred.gswsvm" = pred.gswsvm)
svm.acc[rep,n.model] <- (svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat) #accuracy
svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # sensitivity
svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model] * svm.spe[rep,n.model])

cmat.gswsvm <- svm.cmat
}# use this method or NOT


# method 2 - x do not involve resampling,
# so they share the same training and tuning set.

#################################################################################
# Method 3. Standard SVM
#################################################################################
if (use.method$"svm"){ #use this method or NOT
n.model <- 3
set.seed(rep)

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
set.seed(rep)

# 3.1. The class weights are given based on the method of the paper above:
# negative class : # of positive training samples / # of total training samples
# positive class : # of negative training samples / # of total training samples
weight.svmdc.pos <- imbalance.ratio
weight.svmdc.neg <- 1


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

svm.cmat <- table("truth" = data.test$y, "svmdc.pred" = svmdc.pred)
svm.acc[rep,n.model] <- (svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model] <- svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model] <- svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model] <- svm.cmat[1,1]/sum(svm.cmat[1,])
svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
} #use this method or NOT

#################################################################################
# Method 5. ClusterSVM
#################################################################################
# Reference: Gu, Q., & Han, J, 
# “Clustered support vector machines”.
# Journal of Machine Learning Research, 2013, 31, 307-315.

if (use.method$"clusterSVM"){ #use this method or NOT
n.model <- 5
set.seed(rep)

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
# Method 6. SMOTE
#################################################################################
# Reference: N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer,
# “SMOTE: Synthetic minority over-sampling technique,”
# J. Artif. Intell.Res., vol. 16, no. 1, pp. 321–357, 2002.
n.model <- 6
set.seed(rep)

if (use.method$"smotesvm"){ #use this method or NOT
  # 1. Apply smote to the positive class
  
  ## 1.1. copy the training dataset, since oversampling would result to modified dataset.
  data.smotesvm <- data.train 
  data.smotesvm.pos.idx <- rownames(data.train)[(data.smotesvm$y)=="pos"]
  
  ## 1.2. TUNING
  tuning.criterion.values.smotesvm <- create.tuning.criterion.storage(list("pi.s.pos" = pi.s.pos, "c" = param.set.c, "gamma" = param.set.gamma))
  
  for (k in 1 : length(pi.s.pos)){ #loop over pi.s.pos
    
    ### 1.2.1. Oversample using smote, and split into training and tuning set
    smote.sample = SMOTE(data.train[c("x1", "x2")], data.train["y"], dup_size = 0)

    data.plus.smote <- smote.and.split(data.smotesvm, smote.sample$syn_data, oversample.ratio, tuning.ratio)
    data.smotesvm.train <- data.plus.smote$"data.train.smoted"
    data.smotesvm.tune <- data.plus.smote$"data.train.tune"
    
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
        
        tuning.criterion.values.smotesvm[row.idx.now, c("pi.s.pos", "c", "gamma")] <- c(pi.s.pos, c.now, gamma.now)
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
  data.smotesvm.train <- data.plus.smote$"data.train.smoted"
  data.smotesvm.tune <- data.plus.smote$"data.train.tune"
  
  smotesvm.model <- svm(y ~ ., gamma = param.smotesvm.gamma, cost = param.smotesvm.c, kernel="radial", scale = FALSE, data = data.smotesvm.train)
  smotesvm.pred <- predict(smotesvm.model, data.test[c("x1", "x2")])
  
  svm.cmat=t(table(smotesvm.pred, data.test$y))
  svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
  svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
  svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
  svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
  }#use this method or NOT


#################################################################################
# Method 7. Borderline SMOTE-SVM
#################################################################################
# Reference: 

n.model <- 7
set.seed(rep)

if (use.method$"blsmotesvm"){ #use this method or NOT
  # 1. Apply smote to the positive class
  
  ## 1.1. copy the training dataset, since oversampling would result to modified dataset.
  data.blsmotesvm <- data.train 
  data.blsmotesvm.pos.idx <- rownames(data.train)[(data.blsmotesvm$y)=="pos"]
  
  ## 1.2. TUNING
  tuning.criterion.values.blsmotesvm <- create.tuning.criterion.storage(list("pi.s.pos" = pi.s.pos, "c" = param.set.c, "gamma" = param.set.gamma))
  
  for (k in 1 : length(pi.s.pos)){ #loop over pi.s.pos
    
    ### 1.2.1. Oversample using smote, and split into training and tuning set
    smote.sample = BLSMOTE(data.train[c("x1", "x2")], data.train["y"], dupSize = 0)
    
    data.plus.smote <- smote.and.split(data.blsmotesvm, smote.sample$syn_data, oversample.ratio, tuning.ratio)
    data.blsmotesvm.train <- data.plus.smote$"data.train.smoted"
    data.blsmotesvm.tune <- data.plus.smote$"data.train.tune"
    
    ### 1.2.2. loop over c and gamma
    for (i in 1:length(param.set.c)){ #loop over c
      for (j in 1:length(param.set.gamma)){ #loop over gamma
        row.idx.now <- (k-1) * length(param.set.gamma)*length(param.set.c) + (i-1) * length(param.set.gamma) + j #set row index
        
        c.now <- param.set.c[i]
        gamma.now <- param.set.gamma[j]
        
        model.now <- svm(y ~ ., gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.blsmotesvm.train)# fit weighted svm model
        
        y.pred.now <- predict(model.now, data.blsmotesvm.tune[c("x1", "x2")]) #fitted value for tuning dataset
        
        cmat <- table(data.blsmotesvm.tune$y, y.pred.now)
        sen <- cmat[2,2]/sum(cmat[2,])
        spe <- cmat[1,1]/sum(cmat[1,])
        gme <- sqrt(sen*spe)
        
        tuning.criterion.values.blsmotesvm[row.idx.now, c("pi.s.pos", "c", "gamma")] <- c(pi.s.pos, c.now, gamma.now)
        tuning.criterion.values.blsmotesvm[row.idx.now, "criterion"] <- gme
      }} #end for two for loops
  } #end of pi.s.pos loop
  
  #### 1.2.3. get the best parameters
  idx.sorting <- order(-tuning.criterion.values.blsmotesvm$criterion, tuning.criterion.values.blsmotesvm$c, tuning.criterion.values.blsmotesvm$gamma)
  tuning.criterion.values.blsmotesvm <- tuning.criterion.values.blsmotesvm[idx.sorting, ]
  param.best <- tuning.criterion.values.blsmotesvm[1,]
  param.blsmotesvm.c <- param.best$"c"
  param.blsmotesvm.gamma <- param.best$"gamma"
  param.blsmotesvm.pi.s.pos <- param.best$"pi.s.pos"
  
  # 1.4. with the best hyperparameter, fit the svm
  
  ## 1.4.1. set parameters
  param.blsmotesvm.pi.s.neg <- 1 - param.blsmotesvm.pi.s.pos
  oversample.ratio.blsmotesvm <- (param.blsmotesvm.pi.s.pos / pi.pos) - 1
  
  ### 1.4.2 Oversample using the learned GMC model
  data.plus.smote <- smote.and.split(data.blsmotesvm, smote.sample$syn_data, oversample.ratio.blsmotesvm, tuning.ratio)
  data.blsmotesvm.train <- data.plus.smote$"data.train.smoted"
  data.blsmotesvm.tune <- data.plus.smote$"data.train.tune"
  
  blsmotesvm.model <- svm(y ~ ., gamma = param.blsmotesvm.gamma, cost = param.blsmotesvm.c, kernel="radial", scale = FALSE, data = data.blsmotesvm.train)
  blsmotesvm.pred <- predict(blsmotesvm.model, data.test[c("x1", "x2")])
  
  svm.cmat=t(table(blsmotesvm.pred, data.test$y))
  svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
  svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
  svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
  svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
}#use this method or NOT


#################################################################################
# Method 8. DB SMOTE-SVM
#################################################################################
# Reference: 

n.model <- 8
set.seed(rep)

if (use.method$"dbsmotesvm"){ #use this method or NOT
  # 1. Apply smote to the positive class
  
  ## 1.1. copy the training dataset, since oversampling would result to modified dataset.
  data.dbsmotesvm <- data.train 
  data.dbsmotesvm.pos.idx <- rownames(data.train)[(data.dbsmotesvm$y)=="pos"]
  
  ## 1.2. TUNING
  tuning.criterion.values.dbsmotesvm <- create.tuning.criterion.storage(list("pi.s.pos" = pi.s.pos, "c" = param.set.c, "gamma" = param.set.gamma))
  
  for (k in 1 : length(pi.s.pos)){ #loop over pi.s.pos
    
    ### 1.2.1. Oversample using smote, and split into training and tuning set
    smote.sample = DBSMOTE(data.train[c("x1", "x2")], data.train["y"], dupSize = 0)
    
    data.plus.smote <- smote.and.split(data.dbsmotesvm, smote.sample$syn_data, oversample.ratio, tuning.ratio)
    data.dbsmotesvm.train <- data.plus.smote$"data.train.smoted"
    data.dbsmotesvm.tune <- data.plus.smote$"data.train.tune"
    
    ### 1.2.2. loop over c and gamma
    for (i in 1:length(param.set.c)){ #loop over c
      for (j in 1:length(param.set.gamma)){ #loop over gamma
        row.idx.now <- (k-1) * length(param.set.gamma)*length(param.set.c) + (i-1) * length(param.set.gamma) + j #set row index
        
        c.now <- param.set.c[i]
        gamma.now <- param.set.gamma[j]
        
        model.now <- svm(y ~ ., gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.dbsmotesvm.train)# fit weighted svm model
        
        y.pred.now <- predict(model.now, data.dbsmotesvm.tune[c("x1", "x2")]) #fitted value for tuning dataset
        
        cmat <- table(data.dbsmotesvm.tune$y, y.pred.now)
        sen <- cmat[2,2]/sum(cmat[2,])
        spe <- cmat[1,1]/sum(cmat[1,])
        gme <- sqrt(sen*spe)
        
        tuning.criterion.values.dbsmotesvm[row.idx.now, c("pi.s.pos", "c", "gamma")] <- c(pi.s.pos, c.now, gamma.now)
        tuning.criterion.values.dbsmotesvm[row.idx.now, "criterion"] <- gme
      }} #end for two for loops
  } #end of pi.s.pos loop
  
  #### 1.2.3. get the best parameters
  idx.sorting <- order(-tuning.criterion.values.dbsmotesvm$criterion, tuning.criterion.values.dbsmotesvm$c, tuning.criterion.values.dbsmotesvm$gamma)
  tuning.criterion.values.dbsmotesvm <- tuning.criterion.values.dbsmotesvm[idx.sorting, ]
  param.best <- tuning.criterion.values.dbsmotesvm[1,]
  param.dbsmotesvm.c <- param.best$"c"
  param.dbsmotesvm.gamma <- param.best$"gamma"
  param.dbsmotesvm.pi.s.pos <- param.best$"pi.s.pos"
  
  # 1.4. with the best hyperparameter, fit the svm
  
  ## 1.4.1. set parameters
  param.dbsmotesvm.pi.s.neg <- 1 - param.dbsmotesvm.pi.s.pos
  oversample.ratio.dbsmotesvm <- (param.dbsmotesvm.pi.s.pos / pi.pos) - 1
  
  ### 1.4.2 Oversample using the learned GMC model
  data.plus.smote <- smote.and.split(data.dbsmotesvm, smote.sample$syn_data, oversample.ratio.dbsmotesvm, tuning.ratio)
  data.dbsmotesvm.train <- data.plus.smote$"data.train.smoted"
  data.dbsmotesvm.tune <- data.plus.smote$"data.train.tune"
  
  dbsmotesvm.model <- svm(y ~ ., gamma = param.dbsmotesvm.gamma, cost = param.dbsmotesvm.c, kernel="radial", scale = FALSE, data = data.dbsmotesvm.train)
  dbsmotesvm.pred <- predict(dbsmotesvm.model, data.test[c("x1", "x2")])
  
  svm.cmat=t(table(dbsmotesvm.pred, data.test$y))
  svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
  svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
  svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
  svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
}#use this method or NOT


#################################################################################
# Method 9. SMOTEDC
#################################################################################

if (use.method$"smotedc"){ #use this method or NOT, for flexible comparison
n.model <- 9
set.seed(rep) # for reproducible result    

## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods
data.smotedc <- data.train 

## 2. Hyperparamter tuning procedure

### 2.1. prepare a data.frame for storing the hyperparamter tuning results
tuning.criterion.values.smotedc <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
    
### 2.2. Oversample positive samples using SMOTE, and split into training and tuning set
### first do SMOTE to the positive samples as much as possible, and randomly select samples of designated size
### this is due to the limit of the implementation of smotefamily package: it cannot specifiy the oversample size.
### also, split the dataset into a training set and a tuning set.
### synthetic samples are only added to the training set.
### this process is done by custom function smote.and.split.
smote.sample = SMOTE(data.smotedc[c("x1", "x2")], data.smotedc["y"], dup_size = 0) #do SMOTE as much as possible
data.plus.smote <- smote.and.split(data.smotedc, smote.sample$syn_data, oversample.ratio, tuning.ratio) #sample and split
data.smotedc.train <- data.plus.smote$"data.train.smoted"
data.smotedc.tune <- data.plus.smote$"data.train.tune"

### 2.3. specify "svm error costs" as suggested in Akbani et al.
weight.smotedc.pos <- imbalance.ratio
weight.smotedc.neg <- 1
weight.smotedc <- weight.smotedc.pos * (data.smotedc.train$y == 'pos') + weight.smotedc.neg * (data.smotedc.train$y == 'neg')
      
### 2.4. loop over c and gamma
for (i in 1:length(param.set.c)){ #loop over c
  for (j in 1:length(param.set.gamma)){ #loop over gamma
    row.idx.now <- (i-1) * length(param.set.c) + j #set row index
          
    c.now <- param.set.c[i]
    gamma.now <- param.set.gamma[j]
          
    model.now <- wsvm(data = data.smotedc.train, y ~ ., weight = weight.smotedc, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
          
    y.pred.now <- predict(model.now, data.smotedc.tune[c("x1", "x2")]) #fitted value for tuning dataset
          
    cmat <- table("truth" = data.smotedc.tune$y, "pred" = y.pred.now)
    sen <- cmat[2,2] / sum(cmat[2,])
    spe <- cmat[1,1] / sum(cmat[1,])
    gme <- sqrt(sen*spe)
          
    tuning.criterion.values.smotedc[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
    tuning.criterion.values.smotedc[row.idx.now, "criterion"] <- gme
}} #end for two for loops

#### 2.5. get the best parameters
idx.sorting <- order(-tuning.criterion.values.smotedc$criterion, tuning.criterion.values.smotedc$c, tuning.criterion.values.smotedc$gamma)
tuning.criterion.values.smotedc <- tuning.criterion.values.smotedc[idx.sorting, ]
param.best.smotedc <- tuning.criterion.values.smotedc[1,]

# 3. with the best hyperparameter, fit the svm
smotedc.model <- wsvm(data = data.smotedc.train, y ~ ., weight = weight.smotedc, gamma = param.best.smotedc$"gamma", cost = param.best.smotedc$"c", kernel="radial", scale = FALSE)
smotedc.pred <- predict(smotedc.model, data.test[c("x1", "x2")])
    
svm.cmat <- table("truth" = data.test$y, "pred" = smotedc.pred)
svm.acc[rep,n.model] <-(svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat)
svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
  }#use this method or NOT
  
  
#################################################################################
# Baseline. Bayes Rule
#################################################################################
n.model <- n.method
pred.bayes <- bayes.predict(data.test, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
model.eval.bayes <- model.eval(test.y = data.test$y, pred.y = pred.bayes)

svm.acc[rep,n.model] <- model.eval.bayes$acc
svm.sen[rep,n.model] <- model.eval.bayes$sen
svm.pre[rep,n.model] <- model.eval.bayes$pre
svm.spe[rep,n.model] <- model.eval.bayes$spe
svm.gme[rep,n.model] <- model.eval.bayes$gme


} #replication bracket

# save all replications
write.table(svm.gme,
            paste(
              "/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/gme_result",
              imbalance.ratio, ".csv", sep = "") )
write.table(svm.sen,
            paste(
            "/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/sen_result",
            imbalance.ratio, ".csv", sep = "") )
write.table(svm.spe,
            paste(
            "/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/spe_result",
            imbalance.ratio, ".csv", sep = "") )
write.table(svm.acc,
            paste(
              "/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/acc_result",
              imbalance.ratio, ".csv", sep = "") )
write.table(svm.pre,
            paste(
              "/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/pre_result",
              imbalance.ratio, ".csv", sep = "") )

imbal.gme[as.character(imbalance.ratio), ] <-apply(svm.gme, 2, mean)
imbal.spe[as.character(imbalance.ratio), ] <-apply(svm.spe, 2, mean)
imbal.sen[as.character(imbalance.ratio), ] <-apply(svm.sen, 2, mean)
imbal.acc[as.character(imbalance.ratio), ] <-apply(svm.acc, 2, mean)
imbal.pre[as.character(imbalance.ratio), ] <-apply(svm.pre, 2, mean)

imbal.gme.sd[as.character(imbalance.ratio), ] <-apply(svm.gme, 2, sd)
imbal.spe.sd[as.character(imbalance.ratio), ] <-apply(svm.spe, 2, sd)
imbal.sen.sd[as.character(imbalance.ratio), ] <-apply(svm.sen, 2, sd)
imbal.acc.sd[as.character(imbalance.ratio), ] <-apply(svm.acc, 2, sd)
imbal.pre.sd[as.character(imbalance.ratio), ] <-apply(svm.pre, 2, sd)



sink(file = "/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/output.txt", append = TRUE)
print("---------------------------------")
print("imbalance ratio")
print(imbalance.ratio)
print("gme")
print(apply(svm.gme, 2, mean))
print("sen")
print(apply(svm.sen, 2, mean))
print("spe")
print(apply(svm.spe, 2, mean))
print("sen sd")
print(apply(svm.sen, 2, sd))

end_time <- Sys.time()
("time elapsed")
print(end_time - start_time)

sink(file = NULL)
} #imbalance ratio trials

write.table(imbal.gme,"/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/imbal_gme_result.csv")
write.table(imbal.spe,"/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/imbal_spe_result.csv")
write.table(imbal.sen,"/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/imbal_sen_result.csv")
write.table(imbal.gme.sd,"/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/imbal_gme_sd_eresult.csv")
write.table(imbal.spe.sd,"/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/imbal_spe_sd_eresult.csv")
write.table(imbal.sen.sd,"/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/2/imbal_sen_sd_result.csv")

