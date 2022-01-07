library(mvtnorm)
library(ggplot2)

#calculate score of bayes rule classifier

bayes.predict <- function(data.text, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio){
  x1 <- data.test$x1
  x2 <- data.test$x2
  score <- bayes.score(x1,x2, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
  label <- -1 * (score<0) + 1 * (score>=0)
  label <- factor(label, levels = c(-1, 1))
  levels(label) <- c("neg", "pos")
  return(label)
}

bayes.score <- function(x1,x2, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio){
  p.prior = 1 / (1+imbalance.ratio)
  n.prior = 1 - p.prior
  p.density.sum = 0
  n.density.sum = 0
  
  #vector input and output
  for (i in 1:nrow(p.mus))
    p.density.sum = p.density.sum + mvtnorm::dmvnorm(x = cbind(x1, x2), mean = p.mus[i, ], sigma = p.sigma)
  for (i in 1:nrow(n.mus))
    n.density.sum = n.density.sum + mvtnorm::dmvnorm(x = cbind(x1, x2), mean = n.mus[i, ], sigma = n.sigma)
  
  return((cost.ratio * p.prior * p.density.sum - n.prior * n.density.sum))
}

draw.basic <- function(dataset, col.p, col.n, alpha.p, alpha.n){
  ggplot.object <- ggplot() +
    geom_point(data = dataset[dataset$y == "neg", ], aes(x = x1, y = x2), shape = 1, col = col.n, alpha = alpha.n) +
    geom_point(data = dataset[dataset$y == "pos", ], aes(x = x1, y = x2), shape = 16, col = col.p, alpha = alpha.p)
  return(ggplot.object)
}
  
draw.bayes.rule <- function(train, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio){
  x1.max = max(train$x1)
  x1.min = min(train$x1)
  x2.max = max(train$x2)
  x2.min = min(train$x2)
  
  #decision boundary
  seq.x1 <- seq(x1.min*0.9, x1.max*1.1, length.out = 1000)
  seq.x2 <- seq(x2.min*0.9, x2.max*1.1, length.out = 1000)
  grid.x <- expand.grid(x1 = seq.x1 , x2 = seq.x2 )
  z <- bayes.score(grid.x$x1, grid.x$x2, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)
  boundary.contour.values <- data.frame(grid.x, z)
  
  #coloring classes
  #seq.x1 <- seq(x1.min*0.9, x1.max*1.1, length.out = 100)
  #seq.x2 <- seq(x2.min*0.9, x2.max*1.1, length.out = 100)
  #grid.x <- expand.grid(x1 = seq.x1 , x2 = seq.x2 )
  #z <- bayes.rule.calculate(grid.x$x1, grid.x$x2, p.mus, n.mus, p.sigma, n.sigma, imbalance.ratio, cost.ratio)>0
  #coloring.values <- data.frame(grid.x, z)
  
  #ggplot.object <- ggplot(train, aes(x = x1, y = x2)) +
  #geom_point(aes(col = y), shape = 1) +
  #geom_point(data = coloring.values, aes(x = x1, y=x2, col = z), size = 0.1) +
  ggplot.object <- geom_contour(data = boundary.contour.values, aes(x = x1, y = x2, z = z), breaks = 0, colour="black")
  
  return(ggplot.object)
}

draw.svm.rule <- function(train, svm.model, color){
  #dataset is only used to set the range of the grid.
  x1.max = max(train$x1)
  x1.min = min(train$x1)
  x2.max = max(train$x2)
  x2.min = min(train$x2)
  
  #decision boundary
  seq.x1 <- seq(x1.min*0.9, x1.max*1.1, length.out = 1000)
  seq.x2 <- seq(x2.min*0.9, x2.max*1.1, length.out = 1000)
  grid.x <- expand.grid(x1 = seq.x1 , x2 = seq.x2 )
  z <- predict(svm.model, grid.x, decision.values = TRUE)
  z <- attr(z, "decision.values")
  boundary.contour.values <- data.frame(grid.x, z)
  
  ggplot.object <- geom_contour(data = boundary.contour.values, aes(x = x1, y = x2, z = z), breaks = c(-0.5, 0, 0.5), colour=color)
  
  return(ggplot.object)
}

model.eval <- function(test.y, pred.y){
  conf.mat <- t(table(pred.y, test.y))
  acc.total <- (conf.mat[1,1] + conf.mat[2,2]) / sum(conf.mat)
  sens <- conf.mat[2,2] / sum(conf.mat[2,]) #TP / real P = recall
  prec <- conf.mat[2,2] / sum(conf.mat[,2]) #TP / predicted P
  spec <- conf.mat[1,1] / sum(conf.mat[1,]) #TN / real N
  gmean <- sqrt(sens * spec)
  
  return(list("conf.mat" = conf.mat, "sens" = sens, "prec" = prec, "spec" = spec, "gmean" = gmean))
}