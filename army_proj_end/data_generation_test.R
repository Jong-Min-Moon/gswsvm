library(WeightSVM)

library(gridExtra)
library(tictoc)
source("data_generator.R")
source("bayes_rule.R")
source("gswsvm.R")


imbalance.ratio = 10
cost.ratio = 50
n.sample = 1e3

#checkerboard data
p.mean1=c(5,5);p.mean2=c(15,5);p.mean3=c(10,10);p.mean4=c(20,10);p.mean5=c(5,15);p.mean6=c(15,15);
p.mus <- rbind(p.mean1, p.mean2, p.mean3, p.mean4, p.mean5, p.mean6)
p.sigma=matrix(c(2,0,0,2),2,2)

n.mean1=c(10,5);n.mean2=c(20,5);n.mean3=c(5,10);n.mean4=c(15,10);n.mean5=c(10,15);n.mean6=c(20,15);
n.mus <- rbind(n.mean1,n.mean2,n.mean3,n.mean4, n.mean5, n.mean6)
n.sigma=matrix(c(2.5,0,0,2.5),2,2)

data.train = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.sample)
data.test = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.sample/4)
head(data.test)

plot.basic <- draw.basic(data.train, col.p = "blue", col.n = "red", alpha.p = 0.3, alpha.n = 0.3)
plot.basic




