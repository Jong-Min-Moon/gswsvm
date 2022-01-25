



##### g-mean line plots #####
gme.results <- read.csv("/Users/mac/Documents/GitHub/gswsvm/army_proj_end/outputs/var4_gme_results.csv",
                        row.names = 1)
gme.results <- gme.results[colnames(gme.results)[c(-2,-4, -7)]]


par(mar=c(5, 4, 4, 8), xpd=TRUE)
plot(rownames(gme.results), gme.results$gs.wsvm3,
     ylim = c(0,1), type= 'l',
     xlab = "imbalance.ratio",
     ylab = "g-mean",
     main = "trend of g-mean"
     )
for (i in 2:ncol(gme.results)){
  name <-colnames(gme.results)[i]
  lines(rownames(gme.results),  gme.results[,i], col = i)
}
legend("topright",inset=c(-0.5, 0), legend = colnames(gme.results),
       cex = 0.8, pch = '-', col = 1:dim(gme.results)[1])


##### specificity line plots #####
spe.results <- read.csv("/Users/mac/Documents/GitHub/gswsvm/army_proj_end/outputs/var4_spe_results.csv",
                        row.names = 1)
spe.results <- spe.results[colnames(spe.results)[c(-2,-4, -7)]]


par(mar=c(5, 4, 4, 8), xpd=TRUE)
plot(rownames(spe.results), spe.results$gs.wsvm3,
     ylim = c(0.85,1), type= 'l',
     xlab = "imbalance.ratio",
     ylab = "specificity",
     main = "trend of specificity"
)
for (i in 2:ncol(spe.results)){
  name <-colnames(spe.results)[i]
  lines(rownames(spe.results),  spe.results[,i], col = i)
}
legend("topright",inset=c(-0.5, 0), legend = colnames(spe.results),
       cex = 0.8, pch = '-', col = 1:dim(spe.results)[1])


##### sensitivity line plots #####
sen.results <- read.csv("/Users/mac/Documents/GitHub/gswsvm/army_proj_end/outputs/var4_sen_results.csv",
                        row.names = 1)
sen.results <- sen.results[colnames(sen.results)[c(-2,-4, -7)]]


par(mar=c(5, 4, 4, 8), xpd=TRUE)
plot(rownames(sen.results), sen.results$gs.wsvm3,
     ylim = c(0,1), type= 'l',
     xlab = "imbalance.ratio",
     ylab = "sensitivity",
     main = "trend of sensitivity"
)
for (i in 2:ncol(sen.results)){
  name <-colnames(sen.results)[i]
  lines(rownames(sen.results),  sen.results[,i], col = i)
}
legend("topright",inset=c(-0.5, 0), legend = colnames(sen.results),
       cex = 0.8, pch = '-', col = 1:dim(sen.results)[1])



