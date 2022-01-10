library(e1071)
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
source("zsvm.R")
#################################
# Step 1. parameter setting
#################################
\

## note: In our paper, positive class = minority and negative class = majority.
## 1.1. misclassification cost ratio
c.neg <- 4
c.pos <- 1
cost.ratio <- c.neg / c.pos

## 1.2. data generation parameters
n.samples = 1000
### 1.2.1. data generation imbalance ratio
imbalance.ratio <- 6
pi.pos <- 1 / (1 + imbalance.ratio) # probability of a positive sample being generated
pi.neg <- 1 - pi.pos # probability of a negative sample being generated






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


  ### 2.1.1.generate full dataset
data.full = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.samples)
data.range <- list(x1.max = max(data.full[1]), x1.min = min(data.full[1]), x2.max = max(data.full[2]), x2.min = min(data.full[2]))
  
  ### 2.1.2. split the dataset into training set and testing set by 8:2 strafitied sampling
  idx.split.test <- createDataPartition(data.full$y, p = 1/4)
  data.train <- data.full[-idx.split.test$Resample1, ]
  data.test <- data.full[idx.split.test$Resample1, ]

model.now <- svm(y ~ ., kernel="linear", scale = FALSE, data = data.train)
model.eval.svm <- model.eval(test.y = data.test$y, pred.y = predict(model.now, data.test[1:2]))

plot.basic <- draw.basic(data.test, col.p = "blue", col.n = "red", alpha.p = 0.3, alpha.n = 0.3)
plot.bayes <-draw.bayes.rule(data.test, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
plot.svm <- draw.svm.rule(data.test, model.now, color = 'green', cutoff = c(-1,0,1))
plot.basic + plot.bayes + plot.svm

iris.small <- iris[1:60, c(1,2,5)] # smaller samples outputs "Model is empty!"
model.now <- svm(Species ~ ., kernel="radial", scale = FALSE, data = iris.small)

#prediction formula test
# linear
model.now <- svm(Species ~ ., kernel="linear", scale = FALSE, data = iris.small)
coef <- model.now$coefs
SV <- as.matrix(model.now$SV)
rho <- model.now$rho
kernel.eval <- SV %*% t(as.matrix(iris[1,1:2]))
sum(kernel.eval * coef) - rho
predict(model.now, iris[1,1:2], decision.values = TRUE) #confirmed!

#rbf
iris.small <- iris[1:60, c(1,2,5)] # smaller samples outputs "Model is empty!"

new.y <- (iris.small$Species == "setosa")
new.y <- factor(new.y, levels = c('TRUE', 'FALSE'))

new.y2 <- factor(new.y, levels = c('FALSE', 'TRUE'))

iris.small[3]<- new.y
model.now <- svm(Species ~ ., kernel="radial", gamma = 1/2, scale = FALSE, data = iris.small)

iris.small2<-iris.small
iris.small2[3]<- new.y2
model.now2 <- svm(Species ~ ., kernel="radial", gamma = 1/2, scale = FALSE, data = iris.small2)






iris.small <- iris[1:60, c(1,2,5)] # smaller samples outputs "Model is empty!"
new.y <- 1*(iris.small$Species == "setosa") + (-1)*(iris.small$Species != "setosa")
iris.small[3]<-new.y
model.now1 <- svm(Species ~ ., kernel="radial", gamma = 1/2, scale = FALSE, data = iris.small, type= "C-classification")
model.now$coefs
model.now$SV
predict(model.now1, iris[1:5,1:2], decision.values = TRUE)

set.seed(1)
iris.small2<-iris.small[sample(1:60, replace=FALSE),]
model.now2 <- svm(Species ~ ., kernel="radial", gamma = 1/2, scale = FALSE, data = iris.small2, type= "C-classification")
model.now2$coefs
model.now2$SV
predict(model.now2, iris[1:5,1:2], decision.values = TRUE)


coef <- model.now$coefs
SV <- as.matrix(model.now$SV)
rho <- model.now$rho
K <- zsvm.rbf.kernel(SV, as.matrix(iris[1:10,1:2]), gamma = 1/2)
t(coef) %*% K - rho
predict(model.now, iris[1:10,1:2], decision.values = TRUE) #confirmed!
zsvm.predict(iris[1:10,1:2], model.now, gamma=1/2, z=1)

for (i in 1:2){
  for (j in 1:3){
    for (k in 1:4){
      index = (i-1) * 2*3 + (j-1) * 3 + k
      print(index)
      
    }
      
  }
}

library(performanceEstimation)
set.seed(2021)
table(data.train$y)
genData = SMOTE(data.train[c("x1", "x2")], data.train["y"], dup_size = 0)
table(genData$syn_data$class)
genData_2 = SMOTE(data_example[,-3],data_example[,3],K=7)
table(genData_2$data$class)

test<-smote.and.split(data.train, genData$syn_data, oversample.ratio[1], tuning.ratio)

table(data.train$y)
table(test$data.train.og.train$y)
table(test$data.train.og.tune$y)

