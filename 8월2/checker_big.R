library(WeightSVM)

library(gridExtra)
library(tictoc)
source("data_generator.R")
source("bayes_rule.R")
source("gswsvm.R")

set.seed(2021)
imbalance.ratio = 100
cost.ratio = 50
n.sample = 1e5

#checkerboard data
p.mean1=c(5,5);p.mean2=c(15,5);p.mean3=c(10,10);p.mean4=c(20,10);p.mean5=c(5,15);p.mean6=c(15,15);
p.mus <- rbind(p.mean1, p.mean2, p.mean3, p.mean4, p.mean5, p.mean6)
p.sigma=matrix(c(2,0,0,2),2,2)

n.mean1=c(10,5);n.mean2=c(20,5);n.mean3=c(5,10);n.mean4=c(15,10);n.mean5=c(10,15);n.mean6=c(20,15);
n.mus <- rbind(n.mean1,n.mean2,n.mean3,n.mean4, n.mean5, n.mean6)
n.sigma=matrix(c(2.5,0,0,2.5),2,2)

data.train = generate.mvn.data(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.sample)
data.test = generate.mvn.data(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.sample/4)


#bayes error
test.y = data.test$y
bayes.pred.y = bayes.predict(data.test, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = cost.ratio)
bayes.error <- model.eval(test.y = test.y, pred.y = bayes.pred.y)


weight.vec <- c(rep(4000, sum(data.train$y == "pos")), rep(1, sum(data.train$y == "neg")))
model1 <- wsvm(y ~ ., weight = weight.vec, data = data.train) # same weights
svm.pred.y <- predict(object =model1, newdata = data.test[c("x1", "x2")], decision.values = TRUE)
svm.error <- model.eval(test.y = test.y, pred.y = svm.pred.y)

plot.basic <- draw.basic(data.train, col.p = "blue", col.n = "red", alpha.p = 0.3, alpha.n = 0.3)
plot.bayes <- draw.bayes.rule(data.train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = cost.ratio)
plot.wsvm <- draw.wsvm(model1, data.train)

plot <- plot.basic + plot.bayes + plot.wsvm
#plot1 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 1)
#plot2 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 10)
#plot3 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 20)
#plot4 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 40)
#grid.arrange(plot1, plot2, plot3, plot4, nrow = 1, ncol = 4)


#cost.set <- 2^(-5:5); gamma.set <- 2^(-5:5);
#tic("standard svm")
#svm.best <- tune.svm(y~., dat = data.train, kernel="radial", gamma = gamma.set, cost = cost.set)
#toc()
#svm.gamma[rep,n.model]=svm.best$best.parameters$gamma
#svm.cost[rep,n.model]=svm.best$best.parameters$cost

#svm.unweighted = svm(y~., data=train.data, kernel="radial")

#sum(svm.model$coefs>0) # positive sv
#sum(svm.model$coefs<0) # negative sv