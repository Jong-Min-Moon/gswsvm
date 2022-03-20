library(WeightSVM) # for svm with instatnce-wise differing C
library(mclust) # for Gaussian mixture model
library(mvtnorm)
library(caret) # for data splitting
library(e1071) # for svm
library(SwarmSVM) # for clusterSVM
library(smotefamily)
library(plotly) #plotly 패키지 로드
library(ggplot2)

source("simplifier.R")
replication <- 100

#dataset.type = "original"
dataset.type = "manipulated"

start_time <- Sys.time() 


direc <- "/Users/mac/Documents/GitHub/gswsvm/army_proj_end/results_csv/svmdc/"

setwd("/Users/mac/Documents/GitHub/gswsvm/army_proj_end")

##


#### if we use the dataset without manipulation ####
if (dataset.type == "original"){
  
  army <- read.csv("army.csv")
  predictors <- c("PREC", "TEMP", "WIND_SPEED", "RH")
  army <- army[c(predictors, "DMG")] #delete CATE, since this variable is not included in the national dataset.
  
  national <- read.csv("national.csv")
  national <- national[national$DMG == 0]
  data.full <- rbind(army, national)
  data.full$DMG <- factor(data.full$DMG, levels = c(0, 1))
  levels(data.full$DMG)<- c("neg", "pos")
  names(data.full)[5] <- "y"### 1.2.1. data generation imbalance ratio
  imbalance.ratio = sum(data.full$y == "neg") / sum(data.full$y == "pos")
}


###########################################################################
#### MANIPULATING THE DATASET. THIS PART IS NOT FOR ARTICAL SUBMISSION ####
if (dataset.type == "manipulated"){
imbalance.ratio = 30
imbalance.ratios = c(imbalance.ratio)
# parameters
n.positive <- 40
n.national.positive <- n.positive - 8 #number of national positives to mark as army data

#1. read dataset

#1.1. read army dataset
army <- read.csv("army.csv")
predictors <- c("PREC", "TEMP", "WIND_SPEED", "RH")
army <- army[c(predictors, "DMG")] #delete CATE, since this variable is not included in the national dataset.
army.predictors <-army[predictors]

#1.2. read national dataset
national <- read.csv("national.csv")
national.neg <- national[(national$DMG) == 0, ] #positive dataset
national.neg.predictors <- national.neg[predictors]
national.pos <- national[(national$DMG) == 1, ] #negative dataset
national.pos.predictors <- national.pos[predictors]

#2. manipulate the dataset

#select positive vectors from the national dataset

# standardization of the data. all positive observations from national dataset and army dataset has PREC=0, so we omit PREC when scaling.
preProcValues <-preProcess(rbind(national.pos.predictors, army.predictors)[c("TEMP", "WIND_SPEED", "RH")], method = "range")
national.pos.predictors.for.distance <- national.pos.predictors
national.pos.predictors.for.distance[c("TEMP", "WIND_SPEED", "RH")] <- predict(preProcValues, national.pos.predictors.for.distance[c("TEMP", "WIND_SPEED", "RH")])

army.predictors.for.distance <- army.predictors
army.predictors.for.distance[c("TEMP", "WIND_SPEED", "RH")] <- predict(preProcValues, army.predictors.for.distance[c("TEMP", "WIND_SPEED", "RH")])

l2.distances.pos <- matrix(NA, nrow = nrow(national.pos), ncol = nrow(army.predictors))
rownames(l2.distances.pos) <- rownames(national.pos.predictors)
colnames(l2.distances.pos) <- rownames(army.predictors)

set.seed(2022)
kmeans_4 <- kmeans(army.predictors.for.distance,4)$cluster


for (army.num in rownames(army.predictors.for.distance)){
  army.vector <- unlist(army.predictors.for.distance[army.num, ])
  l2.distances.pos[ , army.num] <- sqrt(
    apply( 
      ( t(as.matrix(national.pos.predictors.for.distance)) - army.vector )^2,
      # in R, matrix - vector is applied column-wise.
      # we want to subtract row-wise, so apply t().
      2, sum)
    # since we applies t(), sum is applied column-wise.
  )
}
l2.distances.pos.kmeans <- l2.distances.pos %*% cbind(kmeans_4 ==1, kmeans_4 ==2, kmeans_4 ==3, kmeans_4 ==4)

l2.distances.pos.group1 <- sort(l2.distances.pos.kmeans[,1])
l2.distances.pos.group2 <- sort(l2.distances.pos.kmeans[,2])
l2.distances.pos.group3 <- sort(l2.distances.pos.kmeans[,3])
l2.distances.pos.group4 <- sort(l2.distances.pos.kmeans[,4])


new.positive.indices.group1 <- names(l2.distances.pos.group1[1:8])
new.positive.indices.group2 <- names(l2.distances.pos.group2[1:8])
new.positive.indices.group3 <- names(l2.distances.pos.group3[1:8])
new.positive.indices.group4 <- names(l2.distances.pos.group4[1:8])

new.positive.instances <- national.pos[c(new.positive.indices.group1, new.positive.indices.group2, new.positive.indices.group3, new.positive.indices.group4), ]

positive.combined <- rbind(new.positive.instances, army)

# calculate l2 distance
# centering and scaling of the data. all positive observations from national dataset and army dataset has PREC=0, so we omit PREC when scaling.
preProcValues <-preProcess(rbind(national.neg.predictors, army.predictors), method = "range")
national.neg.predictors.for.distance <- predict(preProcValues, national.neg.predictors)

army.predictors.for.distance.neg <- predict(preProcValues, army.predictors)

l2.distances.neg <- matrix(NA, nrow = nrow(national.neg), ncol = nrow(army.predictors))
rownames(l2.distances.neg) <- rownames(national.neg.predictors)
colnames(l2.distances.neg) <- rownames(army.predictors)


for (army.num in rownames(army.predictors)){
  army.vector <- unlist(army.predictors.for.distance.neg[army.num, ])
  l2.distances.neg[ , army.num] <- sqrt(
    apply( 
      ( t(as.matrix(national.neg.predictors.for.distance)) - army.vector )^2,
      # in R, matrix - vector is applied column-wise.
      # we want to subtract row-wise, so apply t().
      2, sum)
    # since we applies t(), sum is applied column-wise.
  )
}

l2.distances.neg.mean <- apply(l2.distances.neg, 1, mean)
l2.distances.neg.mean <- sort(l2.distances.neg.mean, decreasing = TRUE)


n.negative <- imbalance.ratio * 40

##IMPORTANT! 
negative.indices.by.distance <- names(l2.distances.neg.mean[ seq(1, n.negative*2, 2) ])

negitive.instances <- national.neg[, ] #Choose the samples that are most different from the army data.


data.full <- rbind(positive.combined, negitive.instances)
data.full$DMG <- factor(data.full$DMG, levels = c(0, 1))
levels(data.full$DMG)<- c("neg", "pos")
names(data.full)[5] <- "y"### 1.2.1. data generation imbalance ratio
}





#################################
# Step 1. parameter setting
############))#####################


# 1.1. simulation parameters
kfoldtimes <- 2
n.method <- 7

tuning.ratio <- 1/2
test.ratio <- 3/8


n.samples <- dim(data.full)[1]
oversample.ratio <- imbalance.ratio - 1


param.set.c = 2^(-10 : 10); 
param.set.gamma = 2^(-10 : 10);


## note: In our paper, positive class = minority and negative class = majority.
## 1.1. misclassification cost ratio




# saving matrices
imbal.gme <- matrix(NA, nrow = length(imbalance.ratios), ncol = n.method)
rownames(imbal.gme) <- imbalance.ratios
colnames(imbal.gme) <- c("gswsvm3", "svm" , "svmdc" , "clusterSVM" , "smotesvm" , "blsmotesvm", "dbsmotesvm")

imbal.sen <- imbal.gme
imbal.spe <- imbal.gme
imbal.acc <- imbal.gme
imbal.pre <- imbal.gme


imbal.gme.sd <- imbal.gme
imbal.spe.sd <- imbal.gme
imbal.sen.sd <- imbal.gme
imbal.acc.sd <- imbal.gme
imbal.pre.sd <- imbal.gme







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
  cat(rep, "th run")
  set.seed(rep)
  ## 2.1. Prepare a dataset
  
  ### 2.1.2. split the dataset into training set and testing set by 8:2 strafitied sampling
  idx.split.test <- createDataPartition(data.full$y, p = test.ratio)$Resample1
  data.train <- data.full[ -1 * idx.split.test, ] # 1 - test.ratio
  data.test  <- data.full[  1 * idx.split.test, ] # test.ratio
  
  
  
  #################################################################################
  # GMC SMOTE + SVMDC
  #################################################################################
  
    n.model <- 1
    set.seed(rep) # for reproducible result
    
    ## 1. copy the training dataset so that the oversampling here doesn't make the dataset for other methods.
    data.gswsvm3 <- data.train 
    
    ## 2. Hyperparamter tuning procedure
    
    ### 2.1. prepare a data.frame for storing the hyperparamter tuning results
    tuning.criterion.values.gswsvm3 <- create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
    
    ### 2.2. Split the original samples into a training set and a tuning set
    
    for (time in 1:kfoldtimes){ #2-times
      idx.split.gswsvm3 <- createDataPartition(data.gswsvm3$y, p = tuning.ratio)$Resample1
      
      for (indicator in c(-1, 1)){ #2-fold cross validation
        
        #data split
        data.gswsvm3.train <- data.gswsvm3[ -1 * indicator * idx.split.gswsvm3, ] 
        data.gswsvm3.tune  <- data.gswsvm3[  1 * indicator * idx.split.gswsvm3, ] 
        
        ### 2.3. data standarization
        preProcValues <-preProcess(data.gswsvm3.train[-5], method = "range")
        data.gswsvm3.train <- predict(preProcValues, data.gswsvm3.train)
        data.gswsvm3.tune <- predict(preProcValues, data.gswsvm3.tune)
        
        ### 2.4. leran GMC model on the positive data
        data.gswsvm3.train.pos <- data.gswsvm3.train[data.gswsvm3.train$y == "pos", ]
        gmc.model.pos <- Mclust(data.gswsvm3.train.pos[ -which(colnames(data.gswsvm3.train.pos) == "y") ], modelNames = c("EII"))#data without y
        data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm3.train.pos, oversample.ratio)
        data.gswsvm3.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm3.train)
        
        ### 2.5. Combine original positive and synthetic positive
        weight.svmdc<- imbalance.ratio * (data.gswsvm3.train$y == 'pos') + 1 * (data.gswsvm3.train$y == 'neg')
        
        ## 2.5. loop over c and gamma and calculate the tuning criterion(sample expected misclassification cost in Lin et al.'s paper)
        for (i in 1:length(param.set.c)){ #loop over c
          for (j in 1:length(param.set.gamma)){ #loop over gamma
            row.idx.now <- (i-1) * length(param.set.gamma) + j #set row index
            
            c.now <- param.set.c[i]
            gamma.now <- param.set.gamma[j]
            
            
            model.now <- wsvm(data = data.gswsvm3.train, y ~ ., weight = weight.svmdc, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE)# fit weighted svm model
            
            y.pred.now <- predict(model.now, data.gswsvm3.tune[ -which(colnames(data.gswsvm3.tune) == "y") ]) #fitted value for tuning dataset
            
            svm.cmat <- table("truth" = data.gswsvm3.tune$y, "pred.gswsvm3" = y.pred.now)
            sen <- svm.cmat[2,2] / sum(svm.cmat[2,]) # sensitivity
            spe <- svm.cmat[1,1] / sum(svm.cmat[1,])
            gme <- sqrt(sen * spe)
            
            
            tuning.criterion.values.gswsvm3[row.idx.now, c("c", "gamma")] <- c(c.now, gamma.now)
            tuning.criterion.values.gswsvm3[row.idx.now, "criterion"] <- tuning.criterion.values.gswsvm3[row.idx.now, "criterion"] + gme
          }} # end of for loops over c and gamma
      } #2-fold 
    } # 5 -times
    
    
    ## 3. fit the model and evalutate its performance
    ### 3.1. get the best parameters
    idx.sorting <- order(-tuning.criterion.values.gswsvm3$criterion, # the larger the g-mean, the better.
                         tuning.criterion.values.gswsvm3$c,
                         tuning.criterion.values.gswsvm3$gamma
                         )
    
    
    tuning.criterion.values.gswsvm3 <- tuning.criterion.values.gswsvm3[idx.sorting, ]
    param.best.gswsvm3 <- tuning.criterion.values.gswsvm3[1,]
    
    ### 3.2. centering and scaling
    preProcValues <-preProcess(data.gswsvm3[-5], method = "range")
    data.gswsvm3 <- predict(preProcValues, data.gswsvm3)
    data.gswsvm3.test <- predict(preProcValues, data.test)
    
    ### 3.3. learn GMC model on the positive data
    data.gswsvm3.train.pos <- data.gswsvm3[data.gswsvm3$y == "pos", ] #on the whole training dataset = data.gswsvm3
    gmc.model.pos <- Mclust(data.gswsvm3.train.pos[ -which(colnames(data.gswsvm3.train.pos) == "y") ], modelNames = c("EII"))#data without y
    data.gmc <- get.gmc.oversample(gmc.model.pos, data.gswsvm3.train.pos, oversample.ratio)
    data.gswsvm3.train <- rbind(data.gmc$"data.gmc.train", data.gswsvm3)
    
    ### 3.4. define weight vector
    weight.svmdc<- imbalance.ratio * (data.gswsvm3.train$y == 'pos') + 1 * (data.gswsvm3.train$y == 'neg')
    
    # 3.4. with the best hyperparameter, fit the gs-wsvm
    svm.model.gswsvm3 <- wsvm(data = data.gswsvm3.train, y ~ .,
                              weight = weight.svmdc,
                              gamma = param.best.gswsvm3$"gamma",
                              cost = param.best.gswsvm3$"c",
                              kernel="radial", scale = FALSE)
    
    pred.gswsvm3 <- predict(svm.model.gswsvm3, data.gswsvm3.test[ -which(colnames(data.gswsvm3.test) == "y") ])
    
    svm.cmat <- table("truth" = data.gswsvm3.test$y, "pred.gswsvm3" = pred.gswsvm3)
    svm.acc[rep,n.model] <- (svm.cmat[1,1] + svm.cmat[2,2]) / sum(svm.cmat) #accuracy
    svm.sen[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[2,]) # sensitivity
    svm.pre[rep,n.model] <- svm.cmat[2,2] / sum(svm.cmat[,2])
    svm.spe[rep,n.model] <- svm.cmat[1,1] / sum(svm.cmat[1,])
    svm.gme[rep,n.model] <- sqrt(svm.sen[rep,n.model] * svm.spe[rep,n.model])
    
    cmat.gswsvm3 <- svm.cmat
    #cmat$"gswsvm3" <- svm.cmat #save confusion matrix for this model, for future analysis
}
  
  



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