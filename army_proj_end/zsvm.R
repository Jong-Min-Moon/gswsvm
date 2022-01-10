

zsvm.rbf.kernel <- function(X, U, gamma){
    a <- as.matrix(apply(X^2, 1, sum))
    b <- as.matrix(apply(U^2, 1, sum))
    one.a <- matrix(1, ncol = nrow(b))
    one.b <- matrix(1, ncol = nrow(a))
    K1 <- one.a %x% a
    K2 <- X %*% t(U)
    K3 <- t(one.b %x% b)
    K <- exp(-(K1 - 2*K2 + K3) * gamma)
  return(K)
    }

zsvm.predict <- function(new.X, svm.model, gamma, z){
  coef <- svm.model$coefs
  coef <- z * coef * (coef>0) + coef * (coef<0)
  SV <- as.matrix(svm.model$SV)
  rho <- svm.model$rho
  K <- zsvm.rbf.kernel(SV, new.X, gamma = gamma)
  score <- t(coef) %*% K - rho
  label <- -1 * (score<0) + 1 * (score>=0)
  label <- factor(label, levels = c(-1, 1))
  levels(label) <- c("neg", "pos")
  return(label)
}

  