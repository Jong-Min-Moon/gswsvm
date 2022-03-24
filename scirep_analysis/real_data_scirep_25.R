library(mvtnorm)

lib.loc.gswsvm = "/Users/mac/Documents/GitHub/gswsvm/packages_for_project"
library("WeightSVM", lib.loc = lib.loc.gswsvm) # for svm with instatnce-wise differing C
library("e1071", lib.loc = lib.loc.gswsvm) # for standard linear and kernel svm
library("mclust", lib.loc = lib.loc.gswsvm) # for Gaussian mixture clustering
library("caret", lib.loc = lib.loc.gswsvm) # 5-fold cv
library("rstudioapi", lib.loc = lib.loc.gswsvm) # 5-fold cv

source("/Users/mac/Documents/GitHub/gswsvm/simplifier.r")

start_time <- Sys.time()


data.full <-
  read.csv(
    "/Users/mac/Documents/GitHub/gswsvm/scirep_analysis/data_manipulated.csv",
    row.names = 1
  )
data.full$y <-
  factor(data.full$y, levels = c("neg", "pos")) # turn the y variable into a factor

#################################
# Step 1. parameter setting
###################################

# 1.1. simulation parameters
n.method <- 3
replication <- 10
param.set.c = 2 ^ (-10:10)

param.set.gamma = 2 ^ (-10:10)


imbalance.ratio.original <- 984 / 40
imbalance.ratio.desired <- imbalance.ratio.original
direc = paste("/Users/mac/Documents/GitHub/gswsvm/scirep_analysis/scirep_result/",
              25,
              sep = "")



#################################
# Step 2. simulation(monte carlo)
#################################

# saving matrices
svm.acc <- matrix(0, replication, n.method)
svm.sen <- matrix(0, replication, n.method)
svm.pre <- matrix(0, replication, n.method)
svm.spe <- matrix(0, replication, n.method)
svm.gme <- matrix(0, replication, n.method)

imbal.gme <- matrix(0, nrow = 1, ncol = n.method)
rownames(imbal.gme) <- imbalance.ratio.desired
colnames(imbal.gme) <- c("gswsvm", "rbfsvm" , "linearsvm")

imbal.sen <- imbal.gme
imbal.spe <- imbal.gme
imbal.acc <- imbal.gme
imbal.pre <- imbal.gme

imbal.gme.sd <- imbal.gme
imbal.spe.sd <- imbal.gme
imbal.sen.sd <- imbal.gme
imbal.acc.sd <- imbal.gme
imbal.pre.sd <- imbal.gme




oversample.ratio <-
  imbalance.ratio.original / imbalance.ratio.desired  - 1
for (rep in 1:replication) {
  cat(rep, "th run")
  
  ### 2.1.2. 5-fold split and use 1 fold as test set
  set.seed(rep) # for reproducible result
  k.fold.test <-
    createFolds(data.full$y,
                k = 5,
                list = FALSE,
                returnTrain = FALSE)
  
  for (foldnum.test in 1:5) {
    set.seed(rep)
    
    # split the dataset into training set and test set
    indices.train <- k.fold.test != foldnum.test
    indices.test  <- k.fold.test == foldnum.test
    data.train <- data.full[indices.train,]
    data.test  <- data.full[indices.test ,]
    
    ## 1. Hyperparamter tuning procedure: for gswsvm and rbfsvm
    
    ### 1.1. prepare a data.frame for storing the hyperparamter tuning results
    tuning.criterion.values.gswsvm <-
      create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
    tuning.criterion.values.rbfsvm <-
      create.tuning.criterion.storage(list("c" = param.set.c, "gamma" = param.set.gamma))
    
    ### 1.2. Split the original samples into a training set and a tuning set
    set.seed(rep) # for reproducible result
    k.fold.tune <-
      createFolds(data.train$y,
                  k = 5,
                  list = FALSE,
                  returnTrain = FALSE)
    for (foldnum.tune in 1:5) {
      indices.train.tune <- k.fold.tune != foldnum.tune
      indices.valid.tune <- k.fold.tune == foldnum.tune
      data.train.tune    <- data.train[indices.train.tune,]
      data.valid.tune    <- data.train[indices.valid.tune,]
      
      ### 1.3. standardize the data
      preProcValues <-
        preProcess(data.train.tune[-5], method = "range") #learn the standardizer on the **training set**
      data.train.tune <-
        predict(preProcValues, data.train.tune) #fit the standardizer on the training set
      data.valid.tune <-
        predict(preProcValues, data.valid.tune) #fit the standardizer on the validation set
      
      ### 1.4.fit and evaluate models for hyperparamter tuning
      
      ### 1.4.1 GMC SMOTE + SVMDC
      ### 1.4.1.1. leran GMC model on the positive data
      data.train.tune.pos <-
        data.train.tune[data.train.tune$y == "pos",]
      
      if (imbalance.ratio.desired != imbalance.ratio.original) {
        gmc.model.pos <-
          Mclust(data.train.tune.pos[-which(colnames(data.train.tune.pos) == "y")], modelNames = c("EII"))#data without y
        n.oversample <-
          round(length(data.train.tune.pos$y) * oversample.ratio)
        print("n.oversample:")
        print(n.oversample)
        data.gmc <-
          get.gmc.oversample(gmc.model.pos, data.train.tune.pos, n.oversample = n.oversample)
        data.gswsvm.train <- rbind(data.gmc, data.train.tune)
      } else{
        print("no oversampling")
        data.gswsvm.train <- data.train.tune
      }
      
      
      ### 1.4.1.2. set weights for gswsvm(two weights)
      weight.svmdc <-
        imbalance.ratio.original * (data.gswsvm.train$y == 'pos') + 1 * (data.gswsvm.train$y == 'neg')
      
      ## 1.4.2. loop over c and gamma and calculate the tuning criterion(sample expected misclassification cost in Lin et al.'s paper)
      for (i in 1:length(param.set.c)) {
        #loop over c
        for (j in 1:length(param.set.gamma)) {
          #loop over gamma
          row.idx.now <-
            (i - 1) * length(param.set.gamma) + j #set row index
          
          c.now <- param.set.c[i]
          gamma.now <- param.set.gamma[j]
          
          ## 1.4.2.1. gs-wsvm
          model.gswsvm.now <-
            wsvm(
              data = data.gswsvm.train,
              y ~ .,
              weight = weight.svmdc,
              gamma = gamma.now,
              cost = c.now,
              kernel = "radial",
              scale = FALSE
            )# fit weighted svm model
          
          y.pred.gswsvm.now <-
            predict(model.gswsvm.now, data.valid.tune[-which(colnames(data.valid.tune) == "y")]) #fitted value for tuning dataset
          
          svm.cmat <-
            table("truth" = data.valid.tune$y, "pred.gswsvm" = y.pred.gswsvm.now)
          sen <- svm.cmat[2, 2] / sum(svm.cmat[2, ]) # sensitivity
          spe <- svm.cmat[1, 1] / sum(svm.cmat[1, ])
          gme <- sqrt(sen * spe)
          
          tuning.criterion.values.gswsvm[row.idx.now, c("c", "gamma")] <-
            c(c.now, gamma.now)
          tuning.criterion.values.gswsvm[row.idx.now, "criterion"] <-
            tuning.criterion.values.gswsvm[row.idx.now, "criterion"] + gme / 5
          
          ## 1.4.2.2. rbfsvm
          model.rbfwsvm.now <- svm(
            data = data.train.tune,
            y ~ .,
            gamma = gamma.now,
            cost = c.now,
            kernel = "radial",
            scale = FALSE
          )# fit weighted svm model
          
          y.pred.rbfwsvm.now <-
            predict(model.rbfwsvm.now, data.valid.tune[-which(colnames(data.valid.tune) == "y")]) #fitted value for tuning dataset
          
          svm.cmat <-
            table("truth" = data.valid.tune$y, "pred.rbfsvm" = y.pred.rbfwsvm.now)
          sen <- svm.cmat[2, 2] / sum(svm.cmat[2, ]) # sensitivity
          spe <- svm.cmat[1, 1] / sum(svm.cmat[1, ])
          gme <- sqrt(sen * spe)
          
          tuning.criterion.values.rbfsvm[row.idx.now, c("c", "gamma")] <-
            c(c.now, gamma.now)
          tuning.criterion.values.rbfsvm[row.idx.now, "criterion"] <-
            tuning.criterion.values.rbfsvm[row.idx.now, "criterion"] + gme / 5
          
        }
      } # end of for loops over c and gamma
    } # end of for loop over tuning 5-fold
    
    ## 2. fit the model and evalutate its performance
    ### 2.1. centering and scaling
    preProcValues <- preProcess(data.train[-5], method = "range")
    data.train <- predict(preProcValues, data.train)
    data.test  <- predict(preProcValues, data.test)
    
    ### 2.2. fit and evalute models
    ### 2.2.1. gswsvm
    n.model <- 1
    ### 2.2.1.1. get the best parameters
    idx.sorting.gswsvm <-
      order(
        -tuning.criterion.values.gswsvm$criterion,
        # the larger the g-mean, the better.
        tuning.criterion.values.gswsvm$c,
        tuning.criterion.values.gswsvm$gamma
      )
    tuning.criterion.values.gswsvm <-
      tuning.criterion.values.gswsvm[idx.sorting.gswsvm,]
    param.best.gswsvm <- tuning.criterion.values.gswsvm[1, ]
    
    ### 2.2.1.2. learn GMC model on the positive data
    data.train.pos <-
      data.train[data.train$y == "pos",] #on the whole training dataset = data.gswsvm
    if (imbalance.ratio.desired != imbalance.ratio.original) {
      gmc.model.pos <-
        Mclust(data.train.pos[-which(colnames(data.train.pos) == "y")], modelNames = c("EII"))#data without y
      n.oversample <-
        round(length(data.train.pos$y) * oversample.ratio)
      print("n.oversample:")
      print(n.oversample)
      
      data.gmc <-
        get.gmc.oversample(gmc.model.pos, data.train.pos, n.oversample = n.oversample)
      data.gswsvm.train <- rbind(data.gmc, data.train)
    } else{
      print("no oversampling")
      data.gswsvm.train <- data.train
    }
    
    ### 2.2.1.3. define weight vector
    weight.svmdc <-
      imbalance.ratio.original * (data.gswsvm.train$y == 'pos') + 1 * (data.gswsvm.train$y == 'neg')
    
    ### 2.3.1.4. with the best hyperparameter, fit the gs-wsvm
    svm.model.gswsvm <- wsvm(
      data = data.gswsvm.train,
      y ~ .,
      weight = weight.svmdc,
      gamma = param.best.gswsvm$"gamma",
      cost = param.best.gswsvm$"c",
      kernel = "radial",
      scale = FALSE
    )
    
    pred.gswsvm <-
      predict(svm.model.gswsvm, data.test[-which(colnames(data.test) == "y")])
    
    svm.cmat <-
      table("truth" = data.test$y, "pred.gswsvm" = pred.gswsvm)
    svm.acc[rep, n.model] <-
      svm.acc[rep, n.model] + ((svm.cmat[1, 1] + svm.cmat[2, 2]) / sum(svm.cmat)) /
      5 #accuracy
    svm.sen[rep, n.model] <-
      svm.sen[rep, n.model] + (svm.cmat[2, 2] / sum(svm.cmat[2, ])) / 5 # sensitivity
    svm.pre[rep, n.model] <-
      svm.pre[rep, n.model] + (svm.cmat[2, 2] / sum(svm.cmat[, 2])) / 5
    svm.spe[rep, n.model] <-
      svm.spe[rep, n.model] + (svm.cmat[1, 1] / sum(svm.cmat[1, ])) / 5
    svm.gme[rep, n.model] <-
      svm.gme[rep, n.model] + (sqrt(svm.sen[rep, n.model] * svm.spe[rep, n.model])) /
      5
    
    
    ### 2.2.2. rbf svm
    n.model <- 2
    ### 2.2.2.1. get the best parameters
    idx.sorting.rbfsvm <-
      order(
        -tuning.criterion.values.rbfsvm$criterion,
        # the larger the g-mean, the better.
        tuning.criterion.values.rbfsvm$c,
        tuning.criterion.values.rbfsvm$gamma
      )
    tuning.criterion.values.rbfsvm <-
      tuning.criterion.values.rbfsvm[idx.sorting.rbfsvm,]
    param.best.rbfsvm <- tuning.criterion.values.rbfsvm[1, ]
    
    ### 2.2.2.2. with the best hyperparameter, fit the gs-wsvm
    svm.model.rbfsvm <- svm(
      data = data.train,
      y ~ .,
      gamma = param.best.gswsvm$"gamma",
      cost = param.best.gswsvm$"c",
      kernel = "radial",
      scale = FALSE
    )
    
    pred.rbfsvm <-
      predict(svm.model.rbfsvm, data.test[-which(colnames(data.test) == "y")])
    
    svm.cmat <-
      table("truth" = data.test$y, "pred.rbfsvm" = pred.rbfsvm)
    svm.acc[rep, n.model] <-
      svm.acc[rep, n.model] + ((svm.cmat[1, 1] + svm.cmat[2, 2]) / sum(svm.cmat)) /
      5 #accuracy
    svm.sen[rep, n.model] <-
      svm.sen[rep, n.model] + (svm.cmat[2, 2] / sum(svm.cmat[2, ])) / 5 # sensitivity
    svm.pre[rep, n.model] <-
      svm.pre[rep, n.model] + (svm.cmat[2, 2] / sum(svm.cmat[, 2])) / 5
    svm.spe[rep, n.model] <-
      svm.spe[rep, n.model] + (svm.cmat[1, 1] / sum(svm.cmat[1, ])) / 5
    svm.gme[rep, n.model] <-
      svm.gme[rep, n.model] + (sqrt(svm.sen[rep, n.model] * svm.spe[rep, n.model])) /
      5
    
    ### 2.2.3. linear svm
    n.model <- 3
    
    ### 2.2.2.2. with the best hyperparameter, fit the gs-wsvm
    svm.model.linearsvm <-
      svm(data = data.train,
          y ~ .,
          kernel = "linear",
          scale = FALSE)
    
    pred.linearsvm <-
      predict(svm.model.linearsvm, data.test[-which(colnames(data.test) == "y")])
    
    svm.cmat <-
      table("truth" = data.test$y, "pred.linearsvm" = pred.linearsvm)
    svm.acc[rep, n.model] <-
      svm.acc[rep, n.model] + ((svm.cmat[1, 1] + svm.cmat[2, 2]) / sum(svm.cmat)) /
      5 #accuracy
    svm.sen[rep, n.model] <-
      svm.sen[rep, n.model] + (svm.cmat[2, 2] / sum(svm.cmat[2, ])) / 5 # sensitivity
    svm.pre[rep, n.model] <-
      svm.pre[rep, n.model] + (svm.cmat[2, 2] / sum(svm.cmat[, 2])) / 5
    svm.spe[rep, n.model] <-
      svm.spe[rep, n.model] + (svm.cmat[1, 1] / sum(svm.cmat[1, ])) / 5
    svm.gme[rep, n.model] <-
      svm.gme[rep, n.model] + (sqrt(svm.sen[rep, n.model] * svm.spe[rep, n.model])) /
      5
  } # end of for loop for train/test 5 fold
}# end of for loop for replications

# save all replications
write.csv(svm.gme,
          paste0(direc, "/gme_result", imbalance.ratio.desired, ".csv"))
write.csv(svm.spe,
          paste0(direc, "/spe_result", imbalance.ratio.desired, ".csv"))
write.csv(svm.sen,
          paste0(direc, "/sen_result", imbalance.ratio.desired, ".csv"))
write.csv(svm.acc,
          paste0(direc, "/acc_result", imbalance.ratio.desired, ".csv"))
write.csv(svm.pre,
          paste0(direc, "/pre_result", imbalance.ratio.desired, ".csv"))


imbal.gme[as.character(imbalance.ratio.desired),] <-
  apply(svm.gme, 2, mean)
imbal.spe[as.character(imbalance.ratio.desired),] <-
  apply(svm.spe, 2, mean)
imbal.sen[as.character(imbalance.ratio.desired),] <-
  apply(svm.sen, 2, mean)
imbal.acc[as.character(imbalance.ratio.desired),] <-
  apply(svm.acc, 2, mean)
imbal.pre[as.character(imbalance.ratio.desired),] <-
  apply(svm.pre, 2, mean)

imbal.gme.sd[as.character(imbalance.ratio.desired),] <-
  apply(svm.gme, 2, sd)
imbal.spe.sd[as.character(imbalance.ratio.desired),] <-
  apply(svm.spe, 2, sd)
imbal.sen.sd[as.character(imbalance.ratio.desired),] <-
  apply(svm.sen, 2, sd)
imbal.acc.sd[as.character(imbalance.ratio.desired),] <-
  apply(svm.acc, 2, sd)
imbal.pre.sd[as.character(imbalance.ratio.desired),] <-
  apply(svm.pre, 2, sd)


sink(file = paste0(direc, "/output.txt"), append = TRUE)
print("---------------------------------")
print("imbalance ratio")
print(imbalance.ratio.desired)
print("gme")
print(apply(svm.gme, 2, mean))
print("gme sd")
print(apply(svm.gme, 2, sd))
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