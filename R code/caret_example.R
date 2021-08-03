source("wsvm.R")
library(caret)
## 1. prepare data
set.seed(1)
data(iris)

iris.binary <- iris[iris$Species != "setosa",]#only use two classes

#partition into training and test dataset
idx.training <- createDataPartition(iris.binary $Species, p = .75, list = FALSE)
training <- iris.binary [ idx.training,]
testing  <- iris.binary [-idx.training,]

#make type vector indicating whether the sample belongs to majority, minority or synthetic sample
label <- training[,ncol(training)]
index.full <- 1:length(label)
type <- 1:length(label)
index.syn <- sample(index.full, 30)
type[label == "virginica"] <- "maj"
type[label == "versicolor"] <- "min"
type[index.syn] <- "syn"

#turn label vector into -1 and 1 for suppor vector machine fitting
y.values <- -1 * (label == "versicolor") + 1 * (label == "virginica")
y <- as.factor(y.values)

#bind type vector to predictors(for k-fold cv)
x <- cbind(training[,-ncol(training)], type)


##2. start hyperparameter tuning
set.seed(2021)
fitControl <- trainControl(method = "repeatedcv",
                           number = 5, #5-fold cv
                           repeats = 10) # repeated 10 times

cv.fit <- train(x, y,  
                   method = weighted.svm, 
                   preProc = c("center", "scale"),
                   tuneLength = 20,
                   trControl = fitControl,
                   three.weights = list(maj = 1, min = 1, syn = 1)
                   )
cv.fit

##3. evaluate the final model with the test data
final.model <- cv.fit$finalModel
final.model.scaler <- cv.fit$preProcess
testing.scaled <- predict(final.model.scaler , testing)

testing.X <- testing.scaled[,-ncol(testing.scaled)]
testing.Y <- testing.scaled[,ncol(testing.scaled)]
testing.Y <- as.factor(-1 * (testing.Y == "versicolor") + 1 * (testing.Y == "virginica"))

pred <-wsvm.predict(testing.X, final.model)$predicted.Y
testing.Y
confusionMatrix(pred, testing.Y) #confusion matrix
