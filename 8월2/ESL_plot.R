library(gridExtra)
source("data_generator.R")
source("bayes_rule.R")

set.seed(2021)
imbalance.ratio = 1000
n.sample = 100000
#bisecting data
p.mus <- mvtnorm::rmvnorm(10, mean = c(1,0), sigma = diag(c(1,1)))
p.sigma = diag(c(1/5,1/5))
n.mus <- mvtnorm::rmvnorm(10, mean = c(0,1), sigma = diag(c(1,1)))
n.sigma = diag(c(1/5,1/5))

train = generate.mvn.data(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.sample)


plot1 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 1)
plot2 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 100)
plot3 <- bayes.rule.draw(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, cost.ratio = 1000)
grid.arrange(plot1, plot2, plot3, nrow = 1, ncol = 3)

