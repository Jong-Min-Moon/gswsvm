library(WeightSVM)
source("data_generator.R")
source("bayes_rule.R")

#cost ratio
c.neg = 2
c.pos = 1

#real imbalance ratio
imbalance.ratio = 9
pi.pos = 1/(1+imbalance.ratio)
pi.neg = 1 - pi.pos
#sampling imbalance ratio
pi.s.pos = 0.4
pi.s.neg = 1 - pi.s.pos
imbalance.ratio.s =  pi.s.neg / pi.s.pos

L.pos = c.neg * pi.s.neg * pi.pos
L.neg = c.pos * pi.s.pos * pi.neg


#data generation
imbalance.ratio = 9
cost.ratio = 2
n.sample = 4e3

#checkerboard data
p.mean1 <- c(0,0);
p.mus <- rbind(p.mean1)
p.sigma <- matrix(c(1,0,0,1),2,2)

n.mean1 <- c(2,2)
n.mus <- rbind(n.mean1)
n.sigma <- matrix(c(2,0,0,1),2,2)

#generate dataset
data = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio.s, n.sample = n.sample)

#split the dataset into training set and tuning set
indices.pos = (1:n.sample)[data$y == "pos"]
indices.pos.train = sample(indices.pos, length(indices.pos)/2)
indices.neg = (1:n.sample)[data$y == "neg"]
indices.neg.train = sample(indices.neg, length(indices.neg)/2)

indices.train = sort(c(indices.pos.train,indices.neg.train))
data.train = data[indices.train, ]
data.tune = data[-indices.train, ]


y.tune.real = data.tune$y
L.vector.tune = (y.tune.real == "pos") * L.pos + (y.tune.real == "neg") * L.neg

y.train.real = data.train$y
L.vector.train = (y.train.real == "pos") * L.pos + (y.train.real == "neg") * L.neg

head(data.tune)



#tuning
c.set = 2^(-5 : 5); 
gamma.set = 2^(-5 : 5);
tuning.criterion.values = matrix(nrow = length(gamma.set), ncol = length(c.set))

for (i in 1:length(gamma.set)){
  for (j in 1:length(c.set)){
    gamma.now = gamma.set[i]
    c.now = c.set[j]
    model.now <- wsvm(y ~ ., weight = L.vector.tune, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.tune)
    y.pred.now <- fitted(model.now) #f(x_i) value
    tuning.criterion.values[i,j] <- sum((y.pred.now != y.tune.real) * L.vector.tune)/(n.sample/2)
  }
}

best.hyperparameter.index <- arrayInd(which.min(tuning.criterion.values), dim(tuning.criterion.values))
gamma.best <- gamma.set[best.hyperparameter.index[1]]
c.best <- c.set[best.hyperparameter.index[2]]

best.model <- wsvm(y ~ ., weight = L.vector.train, gamma = gamma.best, cost = c.best, kernel="radial", data = data.train)

plot.basic <- draw.basic(data.tune, col.p = "blue", col.n = "red", alpha.p = 0.3, alpha.n = 0.3)
plot.bayes <-draw.bayes.rule(data.train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio.s, cost.ratio)
plot.wsvm <- draw.svm.rule(data.train, best.model)
plot.basic + plot.bayes + plot.wsvm

plot(best.model, data.train)

