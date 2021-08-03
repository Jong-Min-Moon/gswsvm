source("wsvm.R")
library(caret)

##1. prepare data
set.seed(1)
data(iris)
iris.binary <- iris[iris$Species != "setosa",]#only use two classes

#partition into training and test dataset
idx.training <- createDataPartition(iris.binary $Species, p = .75, list = FALSE)
training <- iris.binary [ idx.training,]
testing  <- iris.binary [-idx.training,]

#make type vector indicating whether samples belong to majority, minority or synthetic sample
label <- training[,ncol(training)]
index.full <- 1:length(label)
type <- 1:length(label)
index.syn <- sample(index.full, 30)
type[label == "virginica"] <- "maj"
type[label == "versicolor"] <- "min"
type[index.syn] <- "syn"

#turn label vector into -1 and 1 for support vector machine fitting
y.values <- -1 * (label == "versicolor") + 1 * (label == "virginica")
y <- as.factor(y.values)

#scale x
x <- training[,-ncol(training)]
training.scaler <- caret::preProcess(x, method = c("center", "scale"))
x.scaled <- predict(training.scaler, x)


##2. fit weighted rbf svm
fit.result <- wsvm.fit(x = x.scaled, y = y, type = type, three.weights = list(maj = 0.5, min = 0.7, syn = 0.3), kernel = list(type = "rbf", par = 1/4))
#"type" vector indicates whether samples belongs to majority, minority or synthetic sample
fit.result$alpha.sv #result: computed Lagrange multipliers
fit.result$bias #result: computed bias term of the hyperplane

##3. evaluate the fitted model with the test set
testing.X <- testing[,-ncol(testing)]
testing.Y <- testing[,ncol(testing)]
testing.Y <- as.factor(-1 * (testing.Y == "versicolor") + 1 * (testing.Y == "virginica"))

testing.X <- predict(training.scaler , testing.X)

pred <-wsvm.predict(testing.X, fit.result)$predicted.Y
confusionMatrix(pred, testing.Y) #confusion matrix
