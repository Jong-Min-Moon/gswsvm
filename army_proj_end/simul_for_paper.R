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
trial.number <- 18
direc <- paste0("/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/", trial.number)

start_time <- Sys.time() 

# 1.1. simulation parameters
replication <- 100
n.method <- 8
use.method <- list("gswsvm3"= 1,  "svm" = 1, "svmdc" = 1, "clusterSVM" = 1, "smotesvm" = 1, "blsmotesvm"= 1, "dbsmotesvm" = 1)

tuning.ratio <- 1/4
test.ratio <- 1/4

## note: In our paper, positive class = minority and negative class = majority.
## 1.1. misclassification cost ratio
# 여러 상황으로 실험 중...

## 1.2. data generation parameters
n.samples = 2000
### 1.2.1. data generation imbalance ratio

imbalance.ratios <-  seq(10,40,5)

# saving matrices
imbal.gme <- matrix(NA, nrow = length(imbalance.ratios), ncol = n.method)
rownames(imbal.gme) <- imbalance.ratios
colnames(imbal.gme) <- c("gswsvm3", "svm" , "svmdc" , "clusterSVM" , "smotesvm" , "blsmotesvm", "dbsmotesvm", "bayes")

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

c.neg <- imbalance.ratio/2
c.pos <- 1
cost.ratio <- c.neg / c.pos
cost.ratio.og.syn <- cost.ratio
### 1.2.2. sampling imbalance ratio(i.e. imbalance ratio after SMOTE)
### since the performance may vary w.r.t to this quantity,
### we treat this as s hyperparameter and
imbalance.ratio.s <- imbalance.ratio / 3


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
p.mean1 <- c(-2,-5);
p.mean2 <- c(8,-5);
p.mean3 <- c(-7,0);
p.mean4 <- c(3,0);
p.mean5 <- c(-2,5);
p.mean6 <- c(8,5);
p.mus <- rbind(p.mean1, p.mean2, p.mean3, p.mean4, p.mean5, p.mean6)
p.sigma <- matrix(c(2,0,0,2),2,2)

n.mean1 <- c(-7,-5)
n.mean2 <- c(3,-5);
n.mean3 <- c(-2,0);
n.mean4 <- c(8,0);
n.mean5 <- c(-7,5);
n.mean6 <- c(2,5);



n.mus <- rbind(n.mean1,n.mean2,n.mean3,n.mean4, n.mean5, n.mean6)
n.sigma <- matrix(c(2,0,0,2),2,2)

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
gmc.model.pos <- Mclust(data.gswsvm3.train.pos[ -which(colnames(data.gswsvm3.train.pos) == "y") ])#data without y
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
      
    y.pred.now <- predict(model.now, data.gswsvm3.tune[ -which(colnames(data.gswsvm3.tune) == "y") ]) #fitted value for tuning dataset
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

pred.gswsvm3 <- predict(svm.model.gswsvm3, data.test[ -which(colnames(data.test) == "y") ])

svm.cmat <- table("truth" = data.test$y, "pred.gswsvm3" = pred.gswsvm3)
svm.acc[rep,n.model] <- (svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat) #accuracy
svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # sensitivity
svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model] * svm.spe[rep,n.model])

cmat.gswsvm3 <- svm.cmat
#cmat$"gswsvm3" <- svm.cmat #save confusion matrix for this model, for future analysis
}# use this method or NOT



#################################################################################
# Method 2. Standard SVM
#################################################################################
if (use.method$"svm"){ #use this method or NOT
n.model <- 2
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
# Method 3. SVMDC
#################################################################################
# Reference: K. Veropoulos, C. Campbell, and N. Cristianini, 
# “Controlling the sensi-tivity of support vector machines,”
# in Proc. Int. Joint Conf. Artif. Intell.,Stockholm, Sweden, 1999, pp. 55–60.
if (use.method$"svmdc"){ #use this method or NOT
n.model <- 3
set.seed(rep)

## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods
data.svmdc <- data.train 

## 2. Hyperparamter tuning procedure

### 2.1. prepare a data.frame for storing the hyperparamter tuning results
tuning.criterion.values.svmdc <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
    
### 2.2 split the dataset into a training set and a tuning set.
idx.split.og <- createDataPartition(data.svmdc$y, p = tuning.ratio)
data.svmdc.train <- data.svmdc[-idx.split.og$Resample1, ] # 1 - tuning ratio
data.svmdc.tune  <- data.svmdc[ idx.split.og$Resample1, ] # tuning ratio


### 2.4. specify "svm error costs" as suggested in Akbani et al.

weight.svmdc <- imbalance.ratio * (data.svmdc.train$y == 'pos') + 1 * (data.svmdc.train$y == 'neg')
      
### 2.4. loop over c and gamma
for (i in 1:length(param.set.c)){ #loop over c
  for (j in 1:length(param.set.gamma)){ #loop over gamma
    row.idx.now <- (i-1) * length(param.set.c) + j #set row index
          
    c.now <- param.set.c[i]
    gamma.now <- param.set.gamma[j]
          
    model.now <- wsvm(data = data.svmdc.train, y ~ ., weight = weight.svmdc, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
          
    y.pred.now <- predict(model.now, data.svmdc.tune[ -which(colnames(data.svmdc.tune) == "y") ]) 
          
    cmat <- table("truth" = data.svmdc.tune$y, "pred" = y.pred.now)
    sen <- cmat[2,2] / sum(cmat[2,])
    spe <- cmat[1,1] / sum(cmat[1,])
    gme <- sqrt(sen*spe)
          
    tuning.criterion.values.svmdc[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
    tuning.criterion.values.svmdc[row.idx.now, "criterion"] <- gme
}} #end for two for loops

#### 2.5. get the best parameters
idx.sorting <- order(-tuning.criterion.values.svmdc$criterion, tuning.criterion.values.svmdc$c, tuning.criterion.values.svmdc$gamma)
tuning.criterion.values.svmdc <- tuning.criterion.values.svmdc[idx.sorting, ]
param.best.svmdc <- tuning.criterion.values.svmdc[1,]

# 3. with the best hyperparameter, fit the svm
svmdc.model <- wsvm(data = data.svmdc.train, y ~ ., weight = weight.svmdc, gamma = param.best.svmdc$"gamma", cost = param.best.svmdc$"c", kernel="radial", scale = FALSE)
svmdc.pred  <- predict(svmdc.model, data.test[ -which(colnames(data.test) == "y") ])
    
svm.cmat <- table("truth" = data.test$y, "pred" = svmdc.pred)
svm.acc[rep,n.model] <-(svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat)
svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
  }#use this method or NOT

#################################################################################
# Method 4. ClusterSVM
#################################################################################
# Reference: Gu, Q., & Han, J, 
# “Clustered support vector machines”.
# Journal of Machine Learning Research, 2013, 31, 307-315.

if (use.method$"clusterSVM"){ #use this method or NOT
n.model <- 4
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
    
    model.now <- clusterSVM(x = data.svm.train[-which(colnames(data.svm.train) == "y")], y = data.svm.train$y, lambda = lambda.now, cost = c.now, centers = param.clusterSVM.k, seed = 512, verbose = 0) 
    
    y.pred.now = predict(model.now, data.svm.tune[-which(colnames(data.svm.tune) == "y")])$predictions
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
clusterSVM.model <- clusterSVM(x = data.svm.train[-which(colnames(data.svm.train) == "y")], y = data.svm.train$y, lambda = param.clusterSVM.lambda, cost = param.clusterSVM.c, centers = param.clusterSVM.k, seed = 512, verbose = 0) 
clusterSVM.pred = predict(clusterSVM.model, data.test[-which(colnames(data.test) == "y")])$predictions

svm.cmat=t(table(clusterSVM.pred, data.test$y))
svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])

svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])

}#use this method or NOT




#################################################################################
# Method 5. SMOTE SVM
#################################################################################
# Reference: N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer,
# “SMOTE: Synthetic minority over-sampling technique,”
# J. Artif. Intell.Res., vol. 16, no. 1, pp. 321–357, 2002.

#################################################################################
# SMOTE 계열의 template
#################################################################################

if (use.method$"smotesvm"){ #use this method or NOT, for flexible comparison
  n.model <- 5
  set.seed(rep) # for reproducible result    
  
  ## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods
  data.smotesvm <- data.train 
  
  ## 2. Hyperparamter tuning procedure
  
  ### 2.1. prepare a data.frame for storing the hyperparamter tuning results (using a user-defined function "create.tuning.criterion.storage")
  tuning.criterion.values.smotesvm <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
  
  ### 2.2 split the dataset into a training set and a tuning set.
  idx.split.og <- createDataPartition(data.smotesvm$y, p = tuning.ratio)
  data.smotesvm.og.train <- data.smotesvm[-idx.split.og$Resample1, ] # 1 - tuning ratio
  data.smotesvm.og.tune  <- data.smotesvm[ idx.split.og$Resample1, ] # tuning ratio
  
  ### 2.2. Oversample positive samples using SMOTE, and split into training and tuning set
  ### first do SMOTE to the positive samples as much as possible, and randomly select samples of designated size.
  ### this is due to the limit of the implementation of smotefamily package: it cannot specifiy the oversample size.
  ### this process is done by custom function smote.and.split.
  n.oversample.smotesvm <- round( sum(data.smotesvm.og.train$y == "pos") * oversample.ratio) #calculate desired oversample size
  
  # smote function of smotefamly requires that we provide the *entire* training set, including negative samples.
  # dup = 0 option ensures that only the positive samples will be oversampled.
  
  # 2.2.1. First, do a SMOTE once.
  smote.samples = SMOTE(
    X = data.smotesvm.og.train[ -which(colnames(data.smotesvm.og.train) == "y") ],
    target = data.smotesvm.og.train["y"],
    dup_size = 0)$syn_data
  
  # 2.2.2. Then, we concatenate several SMOTE results.
  for (i in 1:ceiling(oversample.ratio) ){    
    smote.samples <- rbind(
      smote.samples,
      SMOTE(X = data.smotesvm.og.train[-which(colnames(data.smotesvm.og.train) == "y")],
            target = data.smotesvm.og.train["y"], dup_size = 0)$syn_data)
  } 
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n.oversample.smotesvm, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("neg", "pos")); 
  colnames(smote.samples.selected) <- c( colnames(data.smotesvm.og.train) )  
  
  ### 2.3. synthetic samples are only added to the training set.
  data.smotesvm.train <- rbind(smote.samples.selected, data.smotesvm.og.train)
  data.smotesvm.tune <- data.smotesvm.og.tune
  
  
  ### 2.4. loop over c and gamma
  for (i in 1:length(param.set.c)){ #loop over c
    for (j in 1:length(param.set.gamma)){ #loop over gamma
      row.idx.now <- (i-1) * length(param.set.c) + j #set row index
      
      c.now <- param.set.c[i]
      gamma.now <- param.set.gamma[j]
      
      model.now <- svm(data = data.smotesvm.train, y ~ ., gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
      
      y.pred.now <- predict(model.now, data.smotesvm.tune[ -which(colnames(data.smotesvm.tune) == "y") ]) 
      
      cmat <- table("truth" = data.smotesvm.tune$y, "pred" = y.pred.now)
      sen <- cmat[2,2] / sum(cmat[2,])
      spe <- cmat[1,1] / sum(cmat[1,])
      gme <- sqrt(sen*spe)
      
      tuning.criterion.values.smotesvm[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
      tuning.criterion.values.smotesvm[row.idx.now, "criterion"] <- gme
    }} #end for two for loops
  
  #### 2.5. get the best parameters
  idx.sorting <- order(-tuning.criterion.values.smotesvm$criterion, tuning.criterion.values.smotesvm$c, tuning.criterion.values.smotesvm$gamma)
  tuning.criterion.values.smotesvm <- tuning.criterion.values.smotesvm[idx.sorting, ]
  param.best.smotesvm <- tuning.criterion.values.smotesvm[1,]
  
  # 3. with the best hyperparameter, fit the svm
  smotesvm.model <- svm(data = data.smotesvm.train, y ~ ., gamma = param.best.smotesvm$"gamma", cost = param.best.smotesvm$"c", kernel="radial", scale = FALSE)
  smotesvm.pred  <- predict(smotesvm.model, data.test[ -which(colnames(data.test) == "y") ])
  
  svm.cmat <- table("truth" = data.test$y, "pred" = smotesvm.pred)
  svm.acc[rep,n.model] <-(svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat)
  svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
  svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
  svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
}#use this method or NOT


#################################################################################
# Method 6. Borderline SMOTE-SVM
#################################################################################
# Reference: 

if (use.method$"blsmotesvm"){ #use this method or NOT, for flexible comparison
  n.model <- 6
  set.seed(rep) # for reproducible result    
  
  ## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods
  data.blsmotesvm <- data.train 
  
  ## 2. Hyperparamter tuning procedure
  
  ### 2.1. prepare a data.frame for storing the hyperparamter tuning results (using a user-defined function "create.tuning.criterion.storage")
  tuning.criterion.values.blsmotesvm <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
  
  ### 2.2 split the dataset into a training set and a tuning set.
  idx.split.og <- createDataPartition(data.blsmotesvm$y, p = tuning.ratio)
  data.blsmotesvm.og.train <- data.blsmotesvm[-idx.split.og$Resample1, ] # 1 - tuning ratio
  data.blsmotesvm.og.tune  <- data.blsmotesvm[ idx.split.og$Resample1, ] # tuning ratio
  
  ### 2.2. Oversample positive samples using SMOTE, and split into training and tuning set
  ### first do SMOTE to the positive samples as much as possible, and randomly select samples of designated size.
  ### this is due to the limit of the implementation of smotefamily package: it cannot specifiy the oversample size.
  ### this process is done by custom function smote.and.split.
  n.oversample.blsmotesvm <- round( sum(data.blsmotesvm.og.train$y == "pos") * oversample.ratio) #calculate desired oversample size
  
  # smote function of smotefamly requires that we provide the *entire* training set, including negative samples.
  # dup = 0 option ensures that only the positive samples will be oversampled.
  
  # 2.2.1. First, do a SMOTE once.
  smote.samples = BLSMOTE(
    X = data.blsmotesvm.og.train[ -which(colnames(data.blsmotesvm.og.train) == "y") ],
    target = data.blsmotesvm.og.train["y"],
    dupSize = 0, K = 5, C = ceiling(n.pos / 4)
    )$syn_data
  
  # 2.2.2. Then, we concatenfate several SMOTE results.
  for (i in 1:ceiling(oversample.ratio) ){    
    smote.samples <- rbind(
      smote.samples,
      BLSMOTE(X = data.blsmotesvm.og.train[-which(colnames(data.blsmotesvm.og.train) == "y")],
            target = data.blsmotesvm.og.train["y"], dupSize = 0, K = 5, C = ceiling(n.pos / 4))$syn_data)
  } 
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n.oversample.blsmotesvm, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("neg", "pos")); 
  colnames(smote.samples.selected) <- c( colnames(data.blsmotesvm.og.train) )  
  
  ### 2.3. synthetic samples are only added to the training set.
  data.blsmotesvm.train <- rbind(smote.samples.selected, data.blsmotesvm.og.train)
  data.blsmotesvm.tune <- data.blsmotesvm.og.tune
  
  
  ### 2.4. loop over c and gamma
  for (i in 1:length(param.set.c)){ #loop over c
    for (j in 1:length(param.set.gamma)){ #loop over gamma
      row.idx.now <- (i-1) * length(param.set.c) + j #set row index
      
      c.now <- param.set.c[i]
      gamma.now <- param.set.gamma[j]
      
      model.now <- svm(data = data.blsmotesvm.train, y ~ ., gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
      
      y.pred.now <- predict(model.now, data.blsmotesvm.tune[ -which(colnames(data.blsmotesvm.tune) == "y") ]) 
      
      cmat <- table("truth" = data.blsmotesvm.tune$y, "pred" = y.pred.now)
      sen <- cmat[2,2] / sum(cmat[2,])
      spe <- cmat[1,1] / sum(cmat[1,])
      gme <- sqrt(sen*spe)
      
      tuning.criterion.values.blsmotesvm[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
      tuning.criterion.values.blsmotesvm[row.idx.now, "criterion"] <- gme
    }} #end for two for loops
  
  #### 2.5. get the best parameters
  idx.sorting <- order(-tuning.criterion.values.blsmotesvm$criterion, tuning.criterion.values.blsmotesvm$c, tuning.criterion.values.blsmotesvm$gamma)
  tuning.criterion.values.blsmotesvm <- tuning.criterion.values.blsmotesvm[idx.sorting, ]
  param.best.blsmotesvm <- tuning.criterion.values.blsmotesvm[1,]
  
  # 3. with the best hyperparameter, fit the svm
  blsmotesvm.model <- svm(data = data.blsmotesvm.train, y ~ ., gamma = param.best.blsmotesvm$"gamma", cost = param.best.blsmotesvm$"c", kernel="radial", scale = FALSE)
  blsmotesvm.pred  <- predict(blsmotesvm.model, data.test[ -which(colnames(data.test) == "y") ])
  
  svm.cmat <- table("truth" = data.test$y, "pred" = blsmotesvm.pred)
  svm.acc[rep,n.model] <-(svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat)
  svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # same as the recall
  svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
  svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
  svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
}#use this method or NOT

#################################################################################
# Method 8. DB SMOTE-SVM
#################################################################################
# Reference: 

if (use.method$"dbsmotesvm"){ #use this method or NOT, for flexible comparison
  n.model <- 7
  set.seed(rep) # for reproducible result    
  
  ## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods
  data.dbsmotesvm <- data.train 
  
  ## 2. Hyperparamter tuning procedure
  
  ### 2.1. prepare a data.frame for storing the hyperparamter tuning results (using a user-defined function "create.tuning.criterion.storage")
  tuning.criterion.values.dbsmotesvm <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
  
  ### 2.2 split the dataset into a training set and a tuning set.
  idx.split.og <- createDataPartition(data.dbsmotesvm$y, p = tuning.ratio)
  data.dbsmotesvm.og.train <- data.dbsmotesvm[-idx.split.og$Resample1, ] # 1 - tuning ratio
  data.dbsmotesvm.og.tune  <- data.dbsmotesvm[ idx.split.og$Resample1, ] # tuning ratio
  
  ### 2.2. Oversample positive samples using SMOTE, and split into training and tuning set
  ### first do SMOTE to the positive samples as much as possible, and randomly select samples of designated size.
  ### this is due to the limit of the implementation of smotefamily package: it cannot specifiy the oversample size.
  ### this process is done by custom function smote.and.split.
  n.oversample.dbsmotesvm <- round( sum(data.dbsmotesvm.og.train$y == "pos") * oversample.ratio) #calculate desired oversample size
  
  # smote function of smotefamly requires that we provide the *entire* training set, including negative samples.
  # dup = 0 option ensures that only the positive samples will be oversampled.
  
  # 2.2.1. First, do a SMOTE once.
  
  
      smote.samples = DBSMOTE(
        X = data.dbsmotesvm.og.train[ -which(colnames(data.dbsmotesvm.og.train) == "y") ],
        target = data.dbsmotesvm.og.train["y"],
        dupSize = 0)$syn_data
    
  
  
  
  
  # 2.2.2. Then, we concatenate several SMOTE results.
  for (i in 1:ceiling(oversample.ratio) ){  
    
        smote.samples_new = DBSMOTE(
          X = data.dbsmotesvm.og.train[ -which(colnames(data.dbsmotesvm.og.train) == "y") ],
          target = data.dbsmotesvm.og.train["y"],
          dupSize = 0)$syn_data
     
    smote.samples <- rbind(smote.samples, smote.samples_new)
  } 
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n.oversample.dbsmotesvm, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("neg", "pos")); 
  colnames(smote.samples.selected) <- c( colnames(data.dbsmotesvm.og.train) )  
  
  ### 2.3. synthetic samples are only added to the training set.
  data.dbsmotesvm.train <- rbind(smote.samples.selected, data.dbsmotesvm.og.train)
  data.dbsmotesvm.tune <- data.dbsmotesvm.og.tune
  
  
  ### 2.4. loop over c and gamma
  for (i in 1:length(param.set.c)){ #loop over c
    for (j in 1:length(param.set.gamma)){ #loop over gamma
      row.idx.now <- (i-1) * length(param.set.c) + j #set row index
      
      c.now <- param.set.c[i]
      gamma.now <- param.set.gamma[j]
      
      model.now <- svm(data = data.dbsmotesvm.train, y ~ ., gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
      
      y.pred.now <- predict(model.now, data.dbsmotesvm.tune[ -which(colnames(data.dbsmotesvm.tune) == "y") ]) 
      
      cmat <- table("truth" = data.dbsmotesvm.tune$y, "pred" = y.pred.now)
      sen <- cmat[2,2] / sum(cmat[2,])
      spe <- cmat[1,1] / sum(cmat[1,])
      gme <- sqrt(sen*spe)
      
      tuning.criterion.values.dbsmotesvm[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
      tuning.criterion.values.dbsmotesvm[row.idx.now, "criterion"] <- gme
    }} #end for two for loops
  
  #### 2.5. get the best parameters
  idx.sorting <- order(-tuning.criterion.values.dbsmotesvm$criterion, tuning.criterion.values.dbsmotesvm$c, tuning.criterion.values.dbsmotesvm$gamma)
  tuning.criterion.values.dbsmotesvm <- tuning.criterion.values.dbsmotesvm[idx.sorting, ]
  param.best.dbsmotesvm <- tuning.criterion.values.dbsmotesvm[1,]
  
  # 3. with the best hyperparameter, fit the svm
  dbsmotesvm.model <- svm(data = data.dbsmotesvm.train, y ~ ., gamma = param.best.dbsmotesvm$"gamma", cost = param.best.dbsmotesvm$"c", kernel="radial", scale = FALSE)
  dbsmotesvm.pred  <- predict(dbsmotesvm.model, data.test[ -which(colnames(data.test) == "y") ])
  
  svm.cmat <- table("truth" = data.test$y, "pred" = dbsmotesvm.pred)
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
write.csv(svm.gme, paste0(direc, "/gme_result", imbalance.ratio, ".csv"))
write.csv(svm.spe, paste0(direc, "/spe_result", imbalance.ratio, ".csv"))
write.csv(svm.sen, paste0(direc, "/sen_result", imbalance.ratio, ".csv"))
write.csv(svm.acc, paste0(direc, "/acc_result", imbalance.ratio, ".csv"))
write.csv(svm.pre, paste0(direc, "/pre_result", imbalance.ratio, ".csv"))


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



sink(file = paste0(direc, "/output.txt"), append = TRUE)
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

write.csv(imbal.gme, paste0(direc, "/imbal_gme_result.csv"))
write.csv(imbal.spe, paste0(direc, "/imbal_spe_result.csv"))
write.csv(imbal.sen, paste0(direc, "/imbal_sen_result.csv"))
write.csv(imbal.acc, paste0(direc, "/imbal_acc_result.csv"))
write.csv(imbal.pre, paste0(direc, "/imbal_spe_result.csv"))

write.csv(imbal.gme.sd, paste0(direc, "/imbal_gme_sd_result.csv"))
write.csv(imbal.spe.sd, paste0(direc, "/imbal_spe_sd_result.csv"))
write.csv(imbal.sen.sd, paste0(direc, "/imbal_sen_sd_result.csv"))
write.csv(imbal.acc.sd, paste0(direc, "/imbal_acc_sd_result.csv"))
write.csv(imbal.pre.sd, paste0(direc, "/imbal_pre_sd_result.csv"))
