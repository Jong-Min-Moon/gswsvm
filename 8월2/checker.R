library(gridExtra)
library(e1071)
source("data_generator.R")
source("bayes_rule.R")

set.seed(2021)
imbalance.ratio = 10000
n.sample = 1000000

#checkerboard data
p.mu.1 <- c(5,5)
p.mu.2 <- c(10,10)
p.mus <- rbind(p.mu.1, p.mu.2)
p.sigma <- matrix(c(2,0,0,2),2,2)

n.mu.1 <- c(10,5)
n.mu.2 <- c(5,10)
n.mus <- rbind(n.mu.1, n.mu.2)
n.sigma <- matrix(c(2.5,0,0,2.5),2,2)

train.data = generate.mvn.data(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.sample)

#plot1 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 1)
#plot2 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 10)
#plot3 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 20)
#plot4 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 40)
#grid.arrange(plot1, plot2, plot3, plot4, nrow = 1, ncol = 4)

start_time <- Sys.time() 
svm.model=svm(y~., data=train.data, kernel="radial")
end_time <- Sys.time()
end_time - start_time