setwd("/Users/mac/Documents/GitHub/army_proj_end")
library(WeightSVM)
library(mclust)
library(mvtnorm)
library(caret)
library(e1071)

source("data_generator.R")
source("bayes_rule.R")

#cost ratio
c.neg = 5
c.pos = 1
cost.ratio = c.neg/c.pos

#real imbalance ratio
imbalance.ratio = 5
pi.pos = 1/(1+imbalance.ratio)
pi.neg = 1 - pi.pos

#sampling imbalance ratio
pi.s.pos <- 0.3
pi.s.neg <- 1 - pi.s.pos
imbalance.ratio.s <-  pi.s.neg / pi.s.pos
oversample.ratio = (pi.s.pos / pi.pos) - 1

L.pos = c.neg * pi.s.neg * pi.pos
L.neg = c.pos * pi.s.pos * pi.neg


#data generation
n.samples = 3000

#checkerboard data
p.mean1 <- c(-7,-5);
p.mean2 <- c(3,-5);
p.mean3 <- c(-2,0);
p.mean4 <- c(8,0);
p.mean5 <- c(-7,5);
p.mean6 <- c(2,5);
p.mus <- rbind(p.mean1, p.mean2, p.mean3, p.mean4, p.mean5, p.mean6)
p.sigma <- matrix(c(3,0,0,3),2,2)

n.mean1 <- c(-2,-5);
n.mean2 <- c(8,-5);
n.mean3 <- c(-7,0);
n.mean4 <- c(3,0);
n.mean5 <- c(-2,5);
n.mean6 <- c(8,5);
n.mus <- rbind(n.mean1,n.mean2,n.mean3,n.mean4, n.mean5, n.mean6)
n.sigma <- matrix(c(6,0,0,6),2,2)

param.set.c = 2^(-5 : 5); 
param.set.gamma = 2^(-5 : 5);

# 1. Prepare a dataset
## 1.1. generate dataset
data.full = generate.mvn.mixture(p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio = imbalance.ratio, n.sample = n.samples)

indices.pos = (1:n.samples)[data.full$y == "pos"]
indices.neg = (1:n.samples)[data.full$y == "neg"]

## 1.2 split the dataset into training set and testing set by 8:2 strafitied sampling
n.samples.pos.train = round(length(indices.pos) * 0.8)
n.samples.neg.train = round(length(indices.neg) * 0.8)

indices.pos.train = sample(indices.pos, n.samples.pos.train)
indices.neg.train = sample(indices.neg, n.samples.neg.train)
indices.train = sort(c(indices.pos.train,indices.neg.train))
data.training = data.full[indices.train, ]
data.testing = data.full[-indices.train, ]




length(indices.neg) / length(indices.pos)

length(data.training$y)
sum((data.training$y) == 'neg')/sum((data.training$y) == 'pos')

length(data.testing$y)
sum((data.testing$y) == 'neg')/sum((data.testing$y) == 'pos')


# 2. GS-WSVM

# 2.1 Apply gmc-smote to the positive class
data.gswsvm <- data.training
data.gswsvm.pos <- data.gswsvm[data.gswsvm$y == 'pos',]
data.gswsvm.neg <- data.gswsvm[data.gswsvm$y == 'neg',]

gmc.model.pos <- Mclust(data.gswsvm.pos[,-3]) # learn Gaussian Mixture Cluster model
G <- gmc.model.pos$G; #learned groups. The number of groups is determined by 
d <- gmc.model.pos$d;
prob <- gmc.model.pos$parameters$pro # learned group membership probabilities
means <- gmc.model.pos$parameters$mean
vars <- gmc.model.pos$parameters$variance$sigma

n.oversample <- round(length(data.gswsvm.pos$y) * oversample.ratio)

gmc.index <- sample(x = 1:G, size = n.oversample, replace = T, prob = prob)
data.gmc.x <- matrix(0, n.oversample, d+1)
for(i in 1 : n.oversample) {
  data.gmc.x[i,1:2] <- rmvnorm(1, mean = means[ , gmc.index[i]],sigma=vars[,,gmc.index[i]])
  data.gmc.x[i,3] <- gmc.index[i]
  }

data.gmc <- data.frame(data.gmc.x, rep("pos", n.oversample))
colnames(data.gmc) <- c("x1", "x2", "group", "y")

# 2.1 split the training data into training and tuning set by 3:1 stratified sampling
# stratified in terms of both synthetic/original and positive/negative

# 2.1.1  split the synthetic positive samples
idx.split.gmc <- createDataPartition(data.gmc$group, p = 3/4)
data.gmc.train <- data.gmc[idx.split.gmc$Resample1, ]
data.gmc.tune <- data.gmc[-idx.split.gmc$Resample1, ]

#confirm stratified
table(data.gmc.train$group)
table(data.gmc.tune$group)

# remove group variable
data.gmc.train = data.gmc.train[-3]
data.gmc.tune = data.gmc.tune[-3]

# 2.1.2 split the original samples
idx.split.gswsvm <- createDataPartition(data.gswsvm$y, p = 3/4)
data.gswsvm.train <- data.gswsvm[idx.split.gswsvm$Resample1, ]
data.gswsvm.tune <- data.gswsvm[-idx.split.gswsvm$Resample1, ]

#confirm stratified
sum(data.gswsvm.tune$y == 'neg') / sum(data.gswsvm.tune$y == 'pos')
sum(data.gswsvm.train$y == 'neg') / sum(data.gswsvm.train$y == 'pos')

#2.1.3. combine original and synthetic
data.gswsvm.train <- rbind(data.gswsvm.train, data.gmc.train)
data.gswsvm.tune <- rbind(data.gswsvm.tune, data.gmc.tune)

L.vector.tune = (data.gswsvm.tune$y == "pos") * L.pos + (data.gswsvm.tune$y == "neg") * L.neg
L.vector.train = (data.gswsvm.train$y == "pos") * L.pos + (data.gswsvm.train$y == "neg") * L.neg

# 2.2 tuning
tuning.criterion.values = matrix(nrow = length(gamma.set), ncol = length(c.set))

for (i in 1:length(gamma.set)){
  for (j in 1:length(c.set)){
    gamma.now = gamma.set[i]
    c.now = c.set[j]
    model.now <- wsvm(y ~ ., weight = L.vector.train, gamma = gamma.now, cost = c.now, kernel="radial", scale = FALSE, data = data.gswsvm.train)
    y.pred.now <- predict(model.now, data.gswsvm.tune[1:2]) #f(x_i) value
    tuning.criterion.values[i,j] <- sum((y.pred.now != data.gswsvm.tune$y) * L.vector.tune)/(n.samples/2)
  }
}

best.hyperparameter.index <- arrayInd(which.min(tuning.criterion.values), dim(tuning.criterion.values))
param.best.gamma <- gamma.set[best.hyperparameter.index[1]]
param.best.c <- c.set[best.hyperparameter.index[2]]

#2.3. fitting and 
gswsvm.model <- wsvm(y ~ ., weight = L.vector.train, gamma = param.best.gamma, cost = param.best.c, kernel="radial", scale = FALSE, data = data.gswsvm.train)
gswsvm.pred <- predict(gswsvm.model, data.testing[1:2])

# 2.4. testing


plot.basic <- draw.basic(data.testing, col.p = "blue", col.n = "red", alpha.p = 0.3, alpha.n = 0.3)
plot.bayes <-draw.bayes.rule(data.testing, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
plot.wsvm <- draw.svm.rule(data.testing, gswsvm.model)
plot.basic + plot.bayes + plot.wsvm

svm.cmat=t(table(gswsvm.pred, data.testing$y))
acc=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
sen=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
pre=svm.cmat[2,2]/sum(svm.cmat[,2])
spe=svm.cmat[1,1]/sum(svm.cmat[1,])
gme=sqrt(sen*spe)

print(cat(acc, sen, pre, spe, gme))

# 3. vanilla svm

svm.best = tune.svm(y~., data = data.gswsvm.tune, kernel="radial", gamma=param.set.gamma, cost = param.set.c, scale = FALSE)
svm.gamma=svm.best$best.parameters$gamma
svm.cost=svm.best$best.parameters$cost

# 2. constructing model   
svm.model=svm(y~., data=data.gswsvm.train, kernel="radial", gamma=svm.gamma, cost=svm.cost, scale = FALSE)
plot.svm.vanilla <- draw.svm.rule(data.gswsvm.train, svm.model)
plot.basic + plot.bayes + plot.svm.vanilla

svm.vanilla.pred <- predict(svm.model, data.testing[1:2])

svm.cmat=t(table(svm.vanilla.pred, data.testing$y))
acc=(svm.cmat[1,1]+svm.cmat[2,2])/sum(svm.cmat)
sen=svm.cmat[2,2]/sum(svm.cmat[2,]) # same as the recall
pre=svm.cmat[2,2]/sum(svm.cmat[,2])
spe=svm.cmat[1,1]/sum(svm.cmat[1,])
gme=sqrt(sen*spe)

print(cat(acc, sen, pre, spe, gme))
