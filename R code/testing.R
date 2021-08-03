getwd()
setwd("/home/jmmoon/문서/wsvm")
source("wsvm.r")
library(e1071)
library(caret)

data(iris)
standard.scaler <- caret::preProcess(iris[,-5], method = c("center", "scale"))
iris.scaled <- predict(standard.scaler, iris)
dataset <- iris.scaled[iris.scaled$Species != "setosa",]

x <- dataset[,1:4]
label <- dataset[,5]
index.full <- 1:length(label)
type <- 1:length(label)

index.syn <- sample(index.full, 30)
type[label == "virginica"] <- "maj"
type[label == "versicolor"] <- "min"
type[index.syn] <- "syn"

y.values <- rep.int(1, length(label))
y.values[label != "virginica"] <- -1
y.values



#linear svm 
source("wsvm.r")
result.mine <- wsvm.fit(x, y.values, type = type, three.weights = list(maj = 1, min = 1, syn = 1))
result.mine
result.e1071 <- e1071::svm(x, y.values, kernel = "linear", type = "C-classification", cost = 1, scale = F)
result.e1071$coefs
result.mine$alpha.sv

data.frame(
index.mine = result.mine$sv$index,
alpha.mine = result.mine$alpha.sv,
index.e1071 = result.e1071$index,
alpha.e1071 = result.e1071$coefs
)


#rbf svm
result.mine <- wsvm.fit(x, y, wts = list(maj = 1, min = 1, syn = 1), kernel = list(type = "rbf", par = 1/4))
result.mine.df <- data.frame(
  index.mine = result.mine$sv$index,
  alpha.mine = result.mine$alpha.sv)

result.mine$bias
library(WeightSVM)
result.WeightSVM <- WeightSVM::wsvm(x = x, y = y.values,
                weight = rep.int(1,length(y.values)),
                scale = F, kernel = "radial", type = "C-classification",
                gamma = 1/4) 

result.WeightSVM$rho

result.WeightSVM.df <- data.frame(
  index.WeightSVM = result.WeightSVM$index,
  alpha.WeightSVM = result.WeightSVM$coefs)




#rbf svm
result.mine <- wsvm.fit(x, y, wts = list(maj = 0.5, min = 0.7, syn = 0.3), kernel = list(type = "rbf", par = 1/4))
result.mine.df <- data.frame(
  index.mine = result.mine$sv$index,
  alpha.mine = result.mine$alpha.sv)

wsvm.predict(x, result.mine)
wsvm.predict(t(as.matrix(c(1,1,1,1))), result.mine)

result.mine$bias
library(WeightSVM)
result.WeightSVM <- WeightSVM::wsvm(x = x, y = -y.values,
                                    weight = result.mine$weight.vec ,
                                    scale = F, kernel = "radial", type = "C-classification",
                                    gamma = 1/4) 

result.WeightSVM$rho
result.WeightSVM.df <- data.frame(
  index.WeightSVM = result.WeightSVM$index,
  alpha.WeightSVM = result.WeightSVM$coefs)
predict(result.WeightSVM, x, decision.values = T)
predict(result.WeightSVM, t(as.matrix(c(1,1,1,1))), decision.values = T)

