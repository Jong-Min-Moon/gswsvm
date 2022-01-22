library(WeightSVM) # for svm with instatnce-wise differing C
library(mclust) # for Gaussian mixture model
library(mvtnorm)
library(caret) # for data splitting
library(e1071) # for svm
library(SwarmSVM) # for clusterSVM
library(smotefamily)
library(plotly) #plotly 패키지 로드

library(ggplot2)

source("bayes_rule.R")
source("data_generator.R")
source("bayes_rule.R")
source("zsvm.R")
source("simplifier.R")

army <- read.csv("army.csv")
army <- army[c("PREC", "TEMP", "WIND_SPEED", "RH", "DMG")] #delete CATE, since this variable is not included in the national dataset.
national <- read.csv("national.csv")
national <- national[(national$DMG) == 0, ]
head(army)
head(national)

#Journal of Computational Biology  VOL. 6, NO. 3-4 |  Articles  normal
#Dissimilarity-Based Algorithms for Selecting Structurally Diverse Sets of Compounds
#Peter Willett
start <- army[c("PREC", "TEMP", "WIND_SPEED", "RH")]
samplePool <- national[c("PREC", "TEMP", "WIND_SPEED", "RH")]
newSamp <- maxDissim(start, samplePool, n = 80)

national.selected <- national[newSamp,]

data.full <- rbind(army, national.selected)
n.predictors <- length(names(data.full)) - 1
data.full$DMG <- factor(data.full$DMG, levels = c(0, 1))
levels(data.full$DMG)<- c("neg", "pos")
names(data.full)[5] <- "y"





ggplot(data.gswsvm.train, aes(TEMP, WIND_SPEED))+
  geom_point(aes(col = y, alpha = 0.2))

ggplot(data.test, aes(TEMP, WIND_SPEED))+
  geom_point(aes(col = y, alpha = 0.2))

3d plot
p <- plot_ly(data=data.full,
             x = data.full$RH, y = data.full$TEMP, z = data.full$WIND_SPEED,
             color = data.full$y, colors = c('#BF382A', '#0C4B8E')
) 

htmlwidgets::saveWidget(p, "test.html")


#################################
# Step 1. parameter setting
#################################

# 1.1. simulation parameters
replication <- 1
n.method <- 10
use.method <- list("gswsvm3"= 1, "gswsvm" = 0, "svm" = 1, "svmdc" = 0, "clusterSVM" = 0, "zsvm" = 1, "smotesvm" = 0, "blsmotesvm"=0, "dbsmotesvm" = 0, "smotedc" = 1)
#set.seed(2021)
tuning.ratio <- 3/5

## note: In our paper, positive class = minority and negative class = majority.
## 1.1. misclassification cost ratio
c.neg <- 10
c.pos <- 1
cost.ratio <- c.neg / c.pos
cost.ratio.og.syn <- 19

## 1.2. data generation parameters


imbalance.ratio <- sum(data.full$y == "neg")/sum(data.full$y == "pos")
  
  
pi.pos <- 1 / (1 + imbalance.ratio) # probability of a positive sample being generated
pi.neg <- 1 - pi.pos # probability of a negative sample being generated

### 1.2.2. sampling imbalance ratio(i.e. imbalance ratio after SMOTE)
### since the performance may vary w.r.t to this quantity,
### we treat this as s hyperparameter and
#pi.s.pos <- c(pi.pos*1.25, pi.pos*1.5, pi.pos*1.75, pi.pos*2, pi.pos*2.25, pi.pos*2.5, pi.pos*3, pi.pos*3.5, pi.pos*4)
pi.s.pos <- c(pi.pos * 5 )
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
  set.seed(rep)
  ## 2.1. Prepare a dataset
  
  ### 2.1.1.generate full dataset
  data.range <- list(x1.max = max(data.full[1]), x1.min = min(data.full[1]), x2.max = max(data.full[2]), x2.min = min(data.full[2]))
  
  ### 2.1.2. split the dataset into training set and testing set by 8:2 strafitied sampling
  idx.split.test <- createDataPartition(data.full$y, p = 3/8)
  data.train <- data.full[ -idx.split.test$Resample1, ] # 1 - 1/4
  data.test  <- data.full[  idx.split.test$Resample1, ] # 1/4
  
  #training data scaling and centering
  preProcValues <- preProcess(data.train, method = c("center", "scale"))
  
  data.train <- predict(preProcValues, data.train)
  data.test <- predict(preProcValues, data.test)
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
    gmc.model.pos <- Mclust(data.gswsvm.pos[,1:n.predictors]) 
    
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
          
          y.pred.now <- predict(model.now, data.gswsvm.tune[1:n.predictors]) #fitted value for tuning dataset
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
    gswsvm.pred <- predict(gswsvm.model, data.test[1:n.predictors])
    
    svm.cmat=t(table(gswsvm.pred, data.test$y))
    svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
    svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
    svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
    svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
    svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
    
    par(mfrow=c(2,2))
    plot(gswsvm.model, data.train,  RH ~ WIND_SPEED)
    plot(gswsvm.model, data.train,  RH ~ TEMP)
    plot(gswsvm.model, data.train,  RH ~ PREC)
    
    plot(gswsvm.model, data.train,  WIND_SPEED ~ TEMP)
    plot(gswsvm.model, data.train,  WIND_SPEED ~ PREC)
    
    plot(gswsvm.model, data.train,  TEMP ~ PREC)
    
    
    }# use this method or NOT
  
}#replication