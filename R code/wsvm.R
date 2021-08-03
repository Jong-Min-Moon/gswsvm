library(quadprog)
library(caret)

#fit svm
wsvm.fit <- function(x, y, type,
                     three.weights = list(maj = 1, min = 1, syn = 1),
                     kernel = list(type = "linear", par = NULL),
                     epsilon = 1e-3){
## type: type of an observation. maj(majority), min(minority) or syn(synthetic sample)
  
  #turn factor into numeric vector with -1 and 1
  y.values <- -1 * (y == -1) + 1 * (y ==1)
  
  #weight vector
  weight.vec <- rep.int(0, length(y.values))
  weight.vec[type == "maj"] <- three.weights$maj
  weight.vec[type == "min"] <- three.weights$min
  weight.vec[type == "syn"] <- three.weights$syn

  #declare preliminary quantities

  epsilon.weighted <- weight.vec * epsilon
  X <- as.matrix(x)
  Y <- as.matrix(y.values)
  n.data <- nrow(Y)
  I.n <- diag(rep(1, n.data))
  
  #compute kernel matrix. Dmat_ij = y_i * y_j * svm.kernel(x_i, x_j).
  K <- svm.kernel(X, X, kernel)
  Dmat <- K * (Y %*% t(Y))
  diag(Dmat) <- diag(Dmat) + 1e-5
  
  #prepare QP
  dvec <- rep(1, n.data)
  Amat <- cbind(Y, I.n, -I.n)
  nonzero <- find.nonzero(Amat) #find.nonzero is a custom function
  Amat = nonzero$Amat.compact
  Aind = nonzero$Aind
  bvec <- c(0, rep(0, n.data), -weight.vec) #wsvm
  
  #find alpha by solving QP
  alpha <- solve.QP.compact(Dmat, dvec, Amat, Aind, bvec, meq = 1)$solution
  alpha.sv <- alpha[alpha > epsilon.weighted]

  #compute the index and the number of support vectors
  index.full <- 1:n.data
  sv.index <- index.full[alpha > epsilon.weighted]#training data points(x_i) whose alpha_i > epsilon are called support vectors
  sv.number <- length(sv.index)
  sv.index.C <- index.full[alpha > epsilon.weighted & alpha < (weight.vec - epsilon.weighted)]
  sv = list(index = sv.index, number = sv.number, index.C = sv.index.C)
  alpha[-sv.index] <- 0 #set alpha_i = 0 if it is too small

  #compute bias term
  if(length(sv.index.C) == 0){
    sv.index.C <- index.full[(alpha > epsilon.weighted) & (alpha <= weight.vec)]
    }
  sv.bias <- Y[sv.index.C] - K[sv.index.C, sv.index] %*% (alpha[sv.index] * Y[sv.index])
  bias <- mean(sv.bias) #average over all support vectors for numerical stability

  #prepare output
  model <- list(alpha = alpha, alpha.sv = alpha.sv,
              bias = bias, sv = sv, kernel = kernel, weight.vec = weight.vec,
              X = X, Y = Y)
  }

svm.kernel <- function(X, U, kernel = list(type = "linear", par = NULL)){
  if (kernel$type == "linear")
    K <- X %*% t(U)
  if (kernel$type == "poly")
    K <- (1 + X %*% t(U)^kernel$par)
  if (kernel$type == "rbf"){
    a <- as.matrix(apply(X^2, 1, sum))
    b <- as.matrix(apply(U^2, 1, sum))
    one.a <- matrix(1, ncol = nrow(b))
    one.b <- matrix(1, ncol = nrow(a))
    K1 <- one.a %x% a
    K2 <- X %*% t(U)
    K3 <- t(one.b %x% b)
    K <- exp(-(K1 - 2*K2 + K3) * kernel$par)
  }
  return(K)}

wsvm.predict <- function(new.X, model){
  X <- model$X
  Y <- model$Y
  new.X <- as.matrix(new.X)
  K <- svm.kernel(X, new.X, model$kernel)
  predicted.values <- model$bias + t(Y*model$alpha) %*% K
  
  predicted.Y <- sign(predicted.values)
  predict <- list(
    predicted.values = predicted.values,
    predicted.Y = as.factor(predicted.Y)
  )
  return(predict)
}

  
find.nonzero <- function(Amat){
  nr <- nrow(Amat)
  nc <- ncol(Amat)
  Amat.compact <- matrix(0, nr, nc)
  Aind <- matrix(0, nr+1, nc)
  for (j in 1:nc){
    index <- (1:nr)[Amat[,j] != 0]
    number <- length(index)
    Amat.compact[1:number, j] <- Amat[index, j]
    Aind[1,j] <- number
    Aind[2:(number+1), j] <- index
  }
  max.number <- max(Aind[1,])
  Amat.compact <- Amat.compact[1:max.number,]
  Aind <- Aind[1:(max.number+1), ]
  compact <- list(Amat.compact = Amat.compact, Aind = Aind)
  
  return(compact)
}


##caret inferface for data preprocessing and hyperparamter tuning
weighted.svm <- list(type = "Classification", library = "kernlab", prob = NULL, loop = NULL) 

#1. paramters element
prm <- data.frame(parameter = c("sigma"), class = c("numeric"), label = c("Sigma"))
weighted.svm$parameters <- prm

#2. grid element
svmGrid <- function(x, y, len = NULL, search = "grid") {
  library(kernlab)
  ## This produces low, middle and high values for sigma 
  ## (i.e. a vector with 3 elements).
  x.values <- as.matrix( x[-dim(x)[2]])
  sigmas <- kernlab::sigest(x.values, na.action = na.omit, scaled = TRUE)  
  ## To use grid search:
  rng <- extendrange(log(sigmas), f = .75)
  out <- data.frame(sigma = exp(seq(from = rng[1], to = rng[2], length.out= len)))
  return(out)
}
weighted.svm$grid <- svmGrid

#3. fit element
svmFit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) { 
  wsvm.fit(x = x[-dim(x)[2]], #separate 'type' column from predictors
           y = y,
           type = x[dim(x)[2]],
           ..., #ellipsis for three.weights argument
           kernel = list(type = "rbf", par = param$sigma)) 
}
weighted.svm$fit <- svmFit

#4. prediction
svmPred <- function(modelFit, newdata, preProc = NULL, submodels = NULL){
  #separate 'type' column from predictors(if exists)
  if (exists("type", newdata)){
    newdata <- subset(newdata, select = - c(type))
  }
  pred <- wsvm.predict(newdata, 
                       modelFit)
  return(pred$predicted.Y)
}
weighted.svm$predict <- svmPred


#6. sort
svmSort <- function(x) x[order(x$sigma),] #the lower the sigma, the smoother the boundary
weighted.svm$sort <- svmSort