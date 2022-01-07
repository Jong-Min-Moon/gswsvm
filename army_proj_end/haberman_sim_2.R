
rm(list = ls()); ls()

setwd("D:\\PAPER\\smote\\haberman")

if (!require(mvtnorm)) install.packages("mvtnorm")
if (!require(e1071)) install.packages("e1071")
if (!require(mclust)) install.packages("mclust")
if (!require(smotefamily)) install.packages("smotefamily")
if (!require(mlbench)) install.packages("mlbench")
if (!require(SDEFSR)) install.packages("SDEFSR")

library(mvtnorm)
library(e1071)
library(mclust)
library(smotefamily)
library(mlbench)
library(SDEFSR)

#################################
# parameter setting
###################################
seed1=12997  # data sampling
seed2=234   # kmeans
seed3=4949  # cross validation

seed_a = 999 # duplication
seed_b = 111 # duplication


# PIMA over sampling 200%, ADASYN

#SVM parameter
Cost.set=2^(-5:5);Gamma.set=2^(-5:5);I.kernel="radial"
replication=20;n.method=1

# Dataset parameter
raw.data = read.csv("haberman.csv", header=TRUE);p=ncol(raw.data)
raw.data[,p]=ifelse(raw.data[,p]==2, "pos", "neg")
raw.data=as.data.frame(raw.data); raw.data[,p]=as.factor(raw.data[,p])
rownames(raw.data)=c(1:nrow(raw.data))
colnames(raw.data)=c("x1","x2","x3","y")

#table(pima2$diabetes)
#table(raw.data$y)

#################################################################################
# Simulation
#################################################################################

# setting storing variable
 svm.acc=matrix(, replication,n.method)
 svm.sen=matrix(, replication,n.method)
 svm.pre=matrix(, replication,n.method)
 svm.spe=matrix(, replication,n.method)
 svm.gme=matrix(, replication,n.method)
 
 svm.cost=matrix(, replication,n.method);
 svm.gamma=matrix(, replication,n.method);
 size.ratio=matrix(, replication,n.method);
 


 for(rep in 1:replication){
   
   #################################################################################
   # random partition of data to train and test
   #################################################################################
   set.seed(seed1*rep)
   index = sample(1:nrow(raw.data), round(nrow(raw.data)*(2/3)))
   train.data = raw.data[index,];N=nrow(train.data);p=ncol(train.data);rownames(train.data)=c(1:N)
   p.train=train.data[which(train.data[,p]=="pos"),];N.p=nrow(p.train);rownames(p.train)=c(1:N.p)
   n.train=train.data[-which(train.data[,p]=="pos"),];N.n=N-N.p;rownames(n.train)=c(1:N.n)
   size.diff=N.n-N.p
   
   test.data = raw.data[-index,];rownames(test.data)=c(1:nrow(test.data))
   p.test=test.data[which(test.data[,p]=="pos"),];rownames(p.test)=c(1:nrow(p.test))
   n.test=test.data[-which(test.data[,p]=="pos"),];rownames(n.test)=c(1:nrow(n.test))
   
   #################################################################################
   # SMOTE-SVM
   #################################################################################
   # 0. SMOTE Over sampling
   over.train.data1=SMOTE(X=train.data[,-p], target=train.data[,p],dup_size=3)$data;  
   over.train.data1[,p]=as.factor(over.train.data1[,p]); names(over.train.data1)=c("x1","x2","x3","y")

   # 1. Model selection : k-fold cross validation
   set.seed(seed3*rep)
   n.model=1
   svm.best=tune.svm(y~., data=over.train.data1, kernel="radial", gamma=Gamma.set, cost=Cost.set)
   svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
   svm.cost[rep,n.model]=svm.best$best.parameters$cost
   
   # 2. constructing model   
   svm.model=svm(y~., data=over.train.data1, kernel="radial", gamma=svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
   svm.pred=predict(svm.model, test.data[,-p])   
   
   # 3. testing
   svm.cmat=t(table(svm.pred, test.data[,p]))
   svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
   svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
   svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
   svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
   svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
   
   # 4. size
   size.ratio[rep,n.model]=sum(over.train.data1[,p]=="pos")/ sum(over.train.data1[,p]=="neg")  
   print(rep)
   
 } # End of replication
 
acc_m = round(apply(svm.acc, 2, mean), 4)
acc_s = round(apply(svm.acc, 2, sd), 4)
acc_e = round(acc_s / sqrt(replication), 4)

result_acc = rbind(acc_m, acc_s, acc_e)
rownames(result_acc) = c("mean", "sd", "se")

sen_m = round(apply(svm.sen, 2, mean), 4)
sen_s = round(apply(svm.sen, 2, sd), 4)
sen_e = round(sen_s / sqrt(replication), 4)

result_sen = rbind(sen_m, sen_s, sen_e)
rownames(result_sen) = c("mean", "sd", "se")


pre_m = round(apply(svm.pre, 2, mean), 4)
pre_s = round(apply(svm.pre, 2, sd), 4)
pre_e = round(pre_s / sqrt(replication), 4)

result_pre = rbind(pre_m, pre_s, pre_e)
rownames(result_pre) = c("mean", "sd", "se")


spe_m = round(apply(svm.spe, 2, mean), 4)
spe_s = round(apply(svm.spe, 2, sd), 4)
spe_e = round(spe_s / sqrt(replication), 4)

result_spe = rbind(spe_m, spe_s, spe_e)
rownames(result_spe) = c("mean", "sd", "se")

gme_m = round(apply(svm.gme, 2, mean), 4)
gme_s = round(apply(svm.gme, 2, sd), 4)
gme_e = round(gme_s / sqrt(replication), 4)

result_gme = rbind(gme_m, gme_s, gme_e)
rownames(result_gme) = c("mean", "sd", "se")

write.csv(result_acc,"300_smote_svm_acc.csv")
write.csv(result_sen ,"300_smote_svm_sen.csv")
write.csv(result_pre,"300_smote_svm_pre.csv")
write.csv(result_spe,"300_smote_svm_spe.csv")
write.csv(result_gme,"300_smote_svm_gme.csv")






rm(list = ls()); ls()

setwd("D:\\PAPER\\smote\\haberman")

if (!require(mvtnorm)) install.packages("mvtnorm")
if (!require(e1071)) install.packages("e1071")
if (!require(mclust)) install.packages("mclust")
if (!require(smotefamily)) install.packages("smotefamily")
if (!require(mlbench)) install.packages("mlbench")
if (!require(SDEFSR)) install.packages("SDEFSR")

library(mvtnorm)
library(e1071)
library(mclust)
library(smotefamily)
library(mlbench)
library(SDEFSR)

#################################
# parameter setting
###################################
seed1=12997  # data sampling
seed2=234   # kmeans
seed3=4949  # cross validation

seed_a = 999 # duplication
seed_b = 111 # duplication


# PIMA over sampling 200%, ADASYN

#SVM parameter
Cost.set=2^(-5:5);Gamma.set=2^(-5:5);I.kernel="radial"
replication=20;n.method=1

# Dataset parameter
raw.data = read.csv("haberman.csv", header=TRUE);p=ncol(raw.data)
raw.data[,p]=ifelse(raw.data[,p]==2, "pos", "neg")
raw.data=as.data.frame(raw.data); raw.data[,p]=as.factor(raw.data[,p])
rownames(raw.data)=c(1:nrow(raw.data))
colnames(raw.data)=c("x1","x2","x3","y")


#table(pima2$diabetes)
#table(raw.data$y)

#################################################################################
# Simulation
#################################################################################

# setting storing variable
 svm.acc=matrix(, replication,n.method)
 svm.sen=matrix(, replication,n.method)
 svm.pre=matrix(, replication,n.method)
 svm.spe=matrix(, replication,n.method)
 svm.gme=matrix(, replication,n.method)
 
 svm.cost=matrix(, replication,n.method);
 svm.gamma=matrix(, replication,n.method);
 size.ratio=matrix(, replication,n.method);
 


 for(rep in 1:replication){
   
   #################################################################################
   # random partition of data to train and test
   #################################################################################
   set.seed(seed1*rep)
   index = sample(1:nrow(raw.data), round(nrow(raw.data)*(2/3)))
   train.data = raw.data[index,];N=nrow(train.data);p=ncol(train.data);rownames(train.data)=c(1:N)
   p.train=train.data[which(train.data[,p]=="pos"),];N.p=nrow(p.train);rownames(p.train)=c(1:N.p)
   n.train=train.data[-which(train.data[,p]=="pos"),];N.n=N-N.p;rownames(n.train)=c(1:N.n)
   size.diff=N.n-N.p
   
   test.data = raw.data[-index,];rownames(test.data)=c(1:nrow(test.data))
   p.test=test.data[which(test.data[,p]=="pos"),];rownames(p.test)=c(1:nrow(p.test))
   n.test=test.data[-which(test.data[,p]=="pos"),];rownames(n.test)=c(1:nrow(n.test))
   
   #################################################################################
   # GMC-SVM
   #################################################################################
   # 0. GMC Over sampling
   size.diff=N.p*3
   
   pos.model=Mclust(p.train[,-p])
   G=pos.model$G; d=pos.model$d; prob=pos.model$parameters$pro
   means=pos.model$parameters$mean
   vars=pos.model$parameters$variance$sigma
   
   gmc.index=sample(x=1:G, size=size.diff, replace = T, prob = prob)
   gmc.train.x=matrix(0,size.diff,d)
   for(i in 1:size.diff) gmc.train.x[i,]=rmvnorm(1,mean=means[,gmc.index[i]],sigma=vars[,,gmc.index[i]])
   gmc.train.y=(rep("pos",size.diff))
   gmc.train=data.frame(gmc.train.x,gmc.train.y);names(gmc.train)=c("x1","x2","x3","y")

   over.train.data1=rbind(train.data,gmc.train);

   # 1. Model selection : k-fold cross validation
   set.seed(seed3*rep)
   n.model=1
   svm.best=tune.svm(y~., data=over.train.data1, kernel="radial", gamma=Gamma.set, cost=Cost.set)
   svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
   svm.cost[rep,n.model]=svm.best$best.parameters$cost
   
   # 2. constructing model   
   svm.model=svm(y~., data=over.train.data1, kernel="radial", gamma=svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
   svm.pred=predict(svm.model, test.data[,-p])   
   
   # 3. testing
   svm.cmat=t(table(svm.pred, test.data[,p]))
   svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
   svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
   svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
   svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
   svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
   
   # 4. size
   size.ratio[rep,n.model]=sum(over.train.data1[,p]=="pos")/ sum(over.train.data1[,p]=="neg")  
   print(rep)
   
 } # End of replication
 
acc_m = round(apply(svm.acc, 2, mean), 4)
acc_s = round(apply(svm.acc, 2, sd), 4)
acc_e = round(acc_s / sqrt(replication), 4)

result_acc = rbind(acc_m, acc_s, acc_e)
rownames(result_acc) = c("mean", "sd", "se")

sen_m = round(apply(svm.sen, 2, mean), 4)
sen_s = round(apply(svm.sen, 2, sd), 4)
sen_e = round(sen_s / sqrt(replication), 4)

result_sen = rbind(sen_m, sen_s, sen_e)
rownames(result_sen) = c("mean", "sd", "se")


pre_m = round(apply(svm.pre, 2, mean), 4)
pre_s = round(apply(svm.pre, 2, sd), 4)
pre_e = round(pre_s / sqrt(replication), 4)

result_pre = rbind(pre_m, pre_s, pre_e)
rownames(result_pre) = c("mean", "sd", "se")


spe_m = round(apply(svm.spe, 2, mean), 4)
spe_s = round(apply(svm.spe, 2, sd), 4)
spe_e = round(spe_s / sqrt(replication), 4)

result_spe = rbind(spe_m, spe_s, spe_e)
rownames(result_spe) = c("mean", "sd", "se")

gme_m = round(apply(svm.gme, 2, mean), 4)
gme_s = round(apply(svm.gme, 2, sd), 4)
gme_e = round(gme_s / sqrt(replication), 4)

result_gme = rbind(gme_m, gme_s, gme_e)
rownames(result_gme) = c("mean", "sd", "se")

write.csv(result_acc,"300_gmc_svm_acc.csv")
write.csv(result_sen ,"300_gmc_svm_sen.csv")
write.csv(result_pre,"300_gmc_svm_pre.csv")
write.csv(result_spe,"300_gmc_svm_spe.csv")
write.csv(result_gme,"300_gmc_svm_gme.csv")

