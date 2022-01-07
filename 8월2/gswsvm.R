library(WeightSVM)
library(caret)

draw.wsvm <- function(wsvm.model, data.train){
  x1.max = max(data.train$x1)
  x1.min = min(data.train$x1)
  x2.max = max(data.train$x2)
  x2.min = min(data.train$x2)
  
  #decision boundary
  seq.x1 <- seq(x1.min*0.9, x1.max*1.1, length.out = 1000)
  seq.x2 <- seq(x2.min*0.9, x2.max*1.1, length.out = 1000)
  grid.x <- expand.grid(x1 = seq.x1 , x2 = seq.x2 )
  
  svm.pred.y <- predict(object = wsvm.model, newdata = grid.x, decision.values = TRUE)
  z <- attr(svm.pred.y, "decision.values")
  boundary.contour.values <- data.frame(grid.x, z)

  ggplot.object <- geom_contour(data = boundary.contour.values, aes(x = x1, y = x2, z = z), breaks = 0, colour="green")
  
  return(ggplot.object)
}