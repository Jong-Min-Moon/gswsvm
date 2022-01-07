setwd("C:/Users/Bang/Desktop/GMC-SVM/Simulation")
source("SVM.BANG(version 7).r")

library(mvtnorm)
library(e1071)
library(mclust)
library(smotefamily)
#################################
# Step 1. parameter setting
###################################
#1.1. seed numnbers
seed1 = 12997  # data sampling
seed2 = 234   # kmeans
seed3 = 4949  # cross validation

#1.2. SVM parameter
Cost.set = 2^(-5 : 5); 
Gamma.set = 2^(-5 : 5);
I.kernel = "radial"
replication = 3; #monte carlo 횟수. 김재오 중령 왈: 일단 10번 정도부터 시작해라.
n.method = 8 # 비교할 방법론 수 (vanilla SVM, SMOTE SVM...)

# GMC parameter


# Parameters for checkerboard dataset generation
 p.mean1=c(5,5);
 p.mean2=c(15,5);
 p.mean3=c(10,10);
 p.mean4=c(20,10);
 p.mean5=c(5,15);
 p.mean6=c(15,15);
 
 p.sigma=matrix(c(2,0,0,2),2,2)
 p.m=rbind(p.mean1,p.mean2,p.mean3,p.mean4, p.mean5, p.mean6)
  
 n.mean1=c(10,5);n.mean2=c(20,5);n.mean3=c(5,10);n.mean4=c(15,10);n.mean5=c(10,15);n.mean6=c(20,15);
 n.sigma=matrix(c(2.5,0,0,2.5),2,2)
 n.m=rbind(n.mean1,n.mean2,n.mean3,n.mean4, n.mean5, n.mean6)

 p.size.train=100;n.size.train=500; size.diff=n.size.train-p.size.train
 p.size.test=10000;n.size.test=10000
 
#################################################################################
#################################################################################
# Part 2. Simulation
#################################################################################
#################################################################################


# setting storing variable
 svm.acc = matrix(, replication,n.method)
 svm.sen = matrix(, replication,n.method)
 svm.pre = matrix(, replication,n.method)
 svm.spe = matrix(, replication,n.method)
 svm.gme = matrix(, replication,n.method)
 
 svm.cost=matrix(, replication,n.method);
 svm.gamma=matrix(, replication,n.method);

#################################################################################
# "replication" 수만큼 모든 동작을 똑같이 반복. 데이터도 매번 새로 만들고, ...
for(rep in 3:replication){ # why start with 3?

 # training data
 set.seed(1365+rep)
 p.index=sample(x=1:6, size=p.size.train, replace = T, prob = NULL)
 n.index=sample(x=1:6, size=n.size.train, replace = T, prob = NULL)

 p.train.x=matrix(0,p.size.train,2);n.train.x=matrix(0,n.size.train,2)
 for(i in 1:p.size.train) p.train.x[i,]=rmvnorm(1,mean=p.m[p.index[i],],sigma=p.sigma)
 for(i in 1:n.size.train) n.train.x[i,]=rmvnorm(1,mean=n.m[n.index[i],],sigma=n.sigma)

 train.x=rbind(p.train.x,n.train.x);p.train.y=rep("pos",p.size.train); n.train.y=rep("neg",n.size.train)
 train.y=c(p.train.y, n.train.y)
 train.data=data.frame(train.x,train.y);colnames(train.data)=c("x1","x2","y")
 
 # test data
 p.index=sample(x=1:6, size=p.size.test, replace = T, prob = NULL)
 n.index=sample(x=1:6, size=n.size.test, replace = T, prob = NULL)
 
 p.test.x=matrix(0,p.size.test,2);n.test.x=matrix(0,n.size.test,2)
 for(i in 1:p.size.test) p.test.x[i,]=rmvnorm(1,mean=p.m[p.index[i],],sigma=p.sigma)
 for(i in 1:n.size.test) n.test.x[i,]=rmvnorm(1,mean=n.m[n.index[i],],sigma=n.sigma)
 
 test.x=rbind(p.test.x,n.test.x);p.test.y=rep("pos",p.size.test); n.test.y=rep("neg",n.size.test)
 test.y=c(p.test.y, n.test.y)
 test.data=data.frame(test.x,test.y);colnames(test.data)=c("x1","x2","y")
 print(test.data)

 #################################################################################
 # Standard SVM
 #################################################################################
 # 1. Model selection : k-fold cross validation
 set.seed(seed3*rep)
 n.model=1
 svm.best=tune.svm(y~., data=train.data, kernel="radial", gamma=Gamma.set, cost=Cost.set)
   svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
   svm.cost[rep,n.model]=svm.best$best.parameters$cost
   
 # 2. constructing model   
 svm.model=svm(y~., data = train.data, kernel="radial", gamma = svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
 svm.pred=predict(svm.model, test.x)   #svm.pred2=predict(svm.best$best.model, test.data[,-3])

 # 3. testing
 svm.cmat=t(table(svm.pred, test.y))
   svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
   svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
   svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
   svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
   svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])

 #################################################################################
 # 2. GMC-SVM
 #################################################################################
 # 0. GMC Over sampling
   pos.model=Mclust(p.train.x)
    G=pos.model$G; d=pos.model$d; prob=pos.model$parameters$pro
    means=pos.model$parameters$mean
    vars=pos.model$parameters$variance$sigma

    gmc.index=sample(x=1:G, size=size.diff, replace = T, prob = prob)
    gmc.train.x=matrix(0,size.diff,d)
    for(i in 1:size.diff) gmc.train.x[i,]=rmvnorm(1,mean=means[,gmc.index[i]],sigma=vars[,,gmc.index[i]])
    
    over.train.x=rbind(train.x,gmc.train.x);gmc.train.y=rep("pos",size.diff)
    over.train.y=c(train.y, gmc.train.y)
    over.train.data=data.frame(over.train.x,over.train.y);colnames(over.train.data)=c("x1","x2","y")
    
 
 # 1. Model selection : k-fold cross validation
   n.model=2
   svm.best=tune.svm(y~., data=over.train.data, kernel="radial", gamma=Gamma.set, cost=Cost.set)
   svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
   svm.cost[rep,n.model]=svm.best$best.parameters$cost
   
   # 2. constructing model   
   svm.model=svm(y~., data=over.train.data, kernel="radial", gamma=svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
   svm.pred=predict(svm.model, test.data[,-3])   #svm.pred2=predict(svm.best$best.model, test.data[,-3])
   
   # 3. testing
   svm.cmat=t(table(svm.pred, test.y))
   svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
   svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
   svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
   svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
   svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
   
   
 #################################################################################
 # 3. SMOTE-SVM
 #################################################################################
 # 0. SMOTE Over sampling
  over.train.data=SMOTE(X=train.data[,-3], target=train.data[,3])$data;  
  over.train.data[,3]=as.factor(over.train.data[,3]); colnames(over.train.data)=c("x1","x2","y") 
     
     # 1. Model selection : k-fold cross validation
     n.model=3
     svm.best=tune.svm(y~., data=over.train.data, kernel="radial", gamma=Gamma.set, cost=Cost.set)
     svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
     svm.cost[rep,n.model]=svm.best$best.parameters$cost
     
     # 2. constructing model   
     svm.model=svm(y~., data=over.train.data, kernel="radial", gamma=svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
     svm.pred=predict(svm.model, test.data[,-3])   #svm.pred2=predict(svm.best$best.model, test.data[,-3])
     
     # 3. testing
     svm.cmat=t(table(svm.pred, test.y))
     svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
     svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
     svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
     svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
     svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
     
 #################################################################################
 # 4. Borderline SMOTE-SVM
 #################################################################################
 # 0. SMOTE Over sampling
  over.train.data=BLSMOTE(X=train.data[,-3], target=train.data[,3])$data;  
  over.train.data[,3]=as.factor(over.train.data[,3]); colnames(over.train.data)=c("x1","x2","y") 
     
     # 1. Model selection : k-fold cross validation
     n.model=4
     svm.best=tune.svm(y~., data=over.train.data, kernel="radial", gamma=Gamma.set, cost=Cost.set)
     svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
     svm.cost[rep,n.model]=svm.best$best.parameters$cost
     
     # 2. constructing model   
     svm.model=svm(y~., data=over.train.data, kernel="radial", gamma=svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
     svm.pred=predict(svm.model, test.data[,-3])   #svm.pred2=predict(svm.best$best.model, test.data[,-3])
     
     # 3. testing
     svm.cmat=t(table(svm.pred, test.y))
     svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
     svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
     svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
     svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
     svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
     
 #################################################################################
 # 5. Safelevel SMOTE-SVM
 #################################################################################
 # 0. Safelevel SMOTE Over sampling
   over.train.data=SLS(X=train.data[,-3], target=train.data[,3])$data;  
   over.train.data[,3]=as.factor(over.train.data[,3]); colnames(over.train.data)=c("x1","x2","y") 
     # 1. Model selection : k-fold cross validation
     n.model=5
     svm.best=tune.svm(y~., data=over.train.data, kernel="radial", gamma=Gamma.set, cost=Cost.set)
     svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
     svm.cost[rep,n.model]=svm.best$best.parameters$cost
     
     # 2. constructing model   
     svm.model=svm(y~., data=over.train.data, kernel="radial", gamma=svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
     svm.pred=predict(svm.model, test.data[,-3])   #svm.pred2=predict(svm.best$best.model, test.data[,-3])
     
     # 3. testing
     svm.cmat=t(table(svm.pred, test.y))
     svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
     svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
     svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
     svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
     svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
     
 ##################################################################################
 # 6. DB SMOTE-SVM
 #################################################################################
 # 0. DBSMOTE Over sampling
  over.train.data=DBSMOTE(X=train.data[,-3], target=train.data[,3])$data;  
  over.train.data[,3]=as.factor(over.train.data[,3]); colnames(over.train.data)=c("x1","x2","y") 
     
     # 1. Model selection : k-fold cross validation
     n.model=6
     svm.best=tune.svm(y~., data=over.train.data, kernel="radial", gamma=Gamma.set, cost=Cost.set)
     svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
     svm.cost[rep,n.model]=svm.best$best.parameters$cost
     
     # 2. constructing model   
     svm.model=svm(y~., data=over.train.data, kernel="radial", gamma=svm.gamma[rep,n.model], cost=svm.cost[rep,n.model])
     svm.pred=predict(svm.model, test.data[,-3])   #svm.pred2=predict(svm.best$best.model, test.data[,-3])
     
     # 3. testing
     svm.cmat=t(table(svm.pred, test.y))
     svm.acc[rep,n.model]=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
     svm.sen[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
     svm.pre[rep,n.model]=svm.cmat[2,2]/sum(svm.cmat[,2])
     svm.spe[rep,n.model]=svm.cmat[1,1]/sum(svm.cmat[1,])
     svm.gme[rep,n.model]=sqrt(svm.sen[rep,n.model]*svm.spe[rep,n.model])
     
     
   print(rep)
     
} # End of replication
 
    colMeans(svm.acc[1:rep,]) 
    colMeans(svm.sen[1:rep,]) 
    colMeans(svm.pre[1:rep,]) 
    colMeans(svm.spe[1:rep,]) 
    colMeans(svm.gme[1:rep,]) 
    
   
   
   
   
 
    
    wkmsvm2.pred=svm.predict(X=compressed.train[,1:(p-1)],Y=compressed.train[,p],new.X=test[,1:(p-1)],new.Y=test[,p],
                          Model=wkmsvm2.model,comp.error.rate=T)  
 wkmsvm2.pred.p=svm.predict(X=compressed.train[,1:(p-1)],Y=compressed.train[,p],new.X=test.p[,1:(p-1)],new.Y=test.p[,p],
                            Model=wkmsvm2.model,comp.error.rate=T)
 wkmsvm2.pred.n=svm.predict(X=compressed.train[,1:(p-1)],Y=compressed.train[,p],new.X=test.n[,1:(p-1)],new.Y=test.n[,p],
                            Model=wkmsvm2.model,comp.error.rate=T)
 
 wkmsvm2.test.err[rep]=wkmsvm2.pred$Error.rate
 wkmsvm2.test.err.p[rep]=wkmsvm2.pred.p$Error.rate
 wkmsvm2.test.err.n[rep]=wkmsvm2.pred.n$Error.rate
 wkmsvm2.n.sv[rep]=wkmsvm2.model$SV$number
 wkmsvm2.best.cost[rep]=wkmsvm2.cost
 wkmsvm2.best.gamma[rep]=wkmsvm2.gamma
 wkmsvm2.best.error[rep]=wkmsvm2.error
 wkmsvm2.time[rep]=Sys.time()-Clustering.start
 
 
 
 
 
Clustering.start=Sys.time() 
  # processing k-means clustering
  k.p=round(N.p/cr.p,0) # number of cluster center for positive class
  k.n=round(N.n/cr.n,0) # number of cluster center for positive class
  
  class.p=0;class.n=0
  set.seed(seed2*rep)
  class.p=kmeans(train.p[,1:(p-1)],k.p, iter.max = iteration, nstart = num.start,algorithm="Lloyd")
  set.seed(seed2*rep)
  class.n=kmeans(train.n[,1:(p-1)],k.n, iter.max = iteration, nstart = num.start,algorithm="Lloyd")
  
  weight.p=class.p$size;weight.n=class.n$size; weight=c(weight.p,weight.n)
  factor.p=N.n/N;factor.n=N.p/N
  weight2=c(factor.p*weight.p,factor.n*weight.n)
  #########################################
  
  y=c(rep(1,k.p),rep(-1,k.n))
  compressed.train=0
  compressed.train=as.matrix(cbind(rbind(class.p$centers,class.n$centers),y)); rownames(compressed.train)=c(1:(k.p+k.n))

#########################################
# WKM-SVM   weight 2
##########################################
  # 1. Model selection : 5-fold cross validation
  set.seed(seed3*rep)
  wkmsvm2.Best=CV.wan(K=10,Weight=weight2,Data=compressed.train,Kernel=I.kernel) # 10-fold cross validation
  wkmsvm2.cost=wkmsvm2.Best$Cost # this is for checking of heuristic strategy in page 61
  wkmsvm2.gamma=wkmsvm2.Best$Gamma  # this is for checking of heuristic strategy in page 61
  wkmsvm2.error=wkmsvm2.Best$Num.error

  # 2. constructing model
  wkmsvm2.model = svm.wan(Weight=weight2,X=compressed.train[,1:(p-1)],Y=compressed.train[,p],Kernel = I.kernel, 
                          Cost=wkmsvm2.cost,Gamma=wkmsvm2.gamma)

  # 3. testing
  wkmsvm2.pred=svm.predict(X=compressed.train[,1:(p-1)],Y=compressed.train[,p],new.X=test[,1:(p-1)],new.Y=test[,p],
                             Model=wkmsvm2.model,comp.error.rate=T)  
    wkmsvm2.pred.p=svm.predict(X=compressed.train[,1:(p-1)],Y=compressed.train[,p],new.X=test.p[,1:(p-1)],new.Y=test.p[,p],
                               Model=wkmsvm2.model,comp.error.rate=T)
    wkmsvm2.pred.n=svm.predict(X=compressed.train[,1:(p-1)],Y=compressed.train[,p],new.X=test.n[,1:(p-1)],new.Y=test.n[,p],
                               Model=wkmsvm2.model,comp.error.rate=T)
  
  wkmsvm2.test.err[rep]=wkmsvm2.pred$Error.rate
    wkmsvm2.test.err.p[rep]=wkmsvm2.pred.p$Error.rate
    wkmsvm2.test.err.n[rep]=wkmsvm2.pred.n$Error.rate
  wkmsvm2.n.sv[rep]=wkmsvm2.model$SV$number
  wkmsvm2.best.cost[rep]=wkmsvm2.cost
  wkmsvm2.best.gamma[rep]=wkmsvm2.gamma
  wkmsvm2.best.error[rep]=wkmsvm2.error
wkmsvm2.time[rep]=Sys.time()-Clustering.start

  # 4. prediction time
start=Sys.time()
  for (i in 1:nrow(test)){
    wkmsvm2.prediction=svm.predict2(X=compressed.train[,1:(p-1)],Y=compressed.train[,p],new.X=test[i,1:(p-1)],new.Y=test[i,p],
                                    Model=wkmsvm2.model,comp.error.rate=T)
   }
wkmsvm2.prediction.time[rep]=Sys.time()-start

print(Sys.time())
print(c("replication=",rep));
print(wkmsvm2.test.err[rep]);print(wkmsvm2.test.err.p[rep]);print(wkmsvm2.test.err.n[rep])
print("mean")
print(mean(wkmsvm2.test.err[1:rep]));print(mean(wkmsvm2.test.err.p[1:rep]));print(mean(wkmsvm2.test.err.n[1:rep]))

} # End of replication

# writing result
directory1= "C:/Documents and Settings/�漺��/���� ȭ��/WKM-SVM Revision#1/Simulation(Revision#1)/Data set/WKM(Wdbc).txt"
wkmsvm2.result=cbind(wkmsvm2.test.err,wkmsvm2.test.err.p,wkmsvm2.test.err.n,wkmsvm2.n.sv,wkmsvm2.best.cost,wkmsvm2.best.gamma,wkmsvm2.time,wkmsvm2.prediction.time)
write.table(wkmsvm2.result,directory1)
