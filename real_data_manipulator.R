library(caret) # for data splitting


library(mclust) # for Gaussian mixture model
library(mvtnorm)
library(e1071) # for svm
library(SwarmSVM) # for clusterSVM
library(smotefamily)
library(plotly) #plotly 패키지 로드
library(ggplot2)
library(rstudioapi)  

setwd(dirname(getActiveDocumentContext()$path)) #setwd to the directory of this code file
direc = getwd()

#### MANIPULATING THE DATASET. THIS CODE IS NOT FOR ARTICAL SUBMISSION ####

# set parameters
n.positive <- 40
n.samples <- 984
n.cluster.army <- 4
jump.step = 4
# automatically determined values
n.national.positive <- n.positive - 8 #number of national positives to mark as army data
n.national.positive.per.cluster = n.national.positive/n.cluster.army
n.negative <- n.samples - n.positive


#1. read dataset

#1.1. read army dataset
army <- read.csv("./data/army.csv")
predictors <- c("PREC", "TEMP", "WIND_SPEED", "RH")
army <- army[c(predictors, "DMG")] #delete CATE, since this variable is not included in the national dataset.
army.predictors <-army[predictors]

#1.2. read national dataset
national <- read.csv("./data/national.csv")
national.neg <- national[(national$DMG) == 0, ] #positive dataset
national.predictors.neg <- national.neg[predictors]
national.pos <- national[(national$DMG) == 1, ] #negative dataset
national.predictors.pos <- national.pos[predictors]


####2. Choose positive national instances####

#2.1. Standardize the data, since we will deal with l2 distance, which is sensitive to the scale.
predictors.positive <- rbind(army.predictors, national.predictors.pos)
preProcValues <-preProcess(predictors.positive, method = "range")
predictors.positive.scaled <- predict(preProcValues, predictors.positive)

#2.2. apply k-means on army data
predictors.army.for.distance <- predictors.positive.scaled[1:8, ]
predictors.national.pos.for.distance <- predictors.positive.scaled[-(1:8), ]
set.seed(2022)
kmeans_4 <- kmeans(predictors.army.for.distance,4)$cluster #each element represents the membership

#2.3. calculate distance from national instance to each cluster in #2.2
l2.distances.pos <- matrix(NA, nrow = nrow(national.pos), ncol = nrow(army.predictors))
rownames(l2.distances.pos) <- rownames(national.predictors.pos)
colnames(l2.distances.pos) <- rownames(army.predictors)

#2.3.1 calculate elementwise distances
for (army.num in rownames(predictors.army.for.distance)){
  # loop over 8 army instances.
  # Each iteration produces a column whose [i]th element represents
  # l2 distance between [army.num]th army instance and [i]th national instance.
  army.vector <- unlist(predictors.army.for.distance[army.num, ])
  l2.distances.pos[ , army.num] <- sqrt(
    apply( 
      ( t(as.matrix(predictors.national.pos.for.distance)) - army.vector )^2,
      # in R, matrix - vector is applied column-wise.
      # we want to subtract row-wise, so apply t().
      2, sum)
    # since we applies t(), sum is applied column-wise.
  )
}

#2.3.2 sum over each cluster.
# (i)th column represents sum of distances from (i)th army cluster to all national instances.
l2.distances.pos.kmeans <- l2.distances.pos %*% cbind(kmeans_4 ==1, kmeans_4 ==2, kmeans_4 ==3, kmeans_4 ==4)

l2.distances.pos.group1 <- sort(l2.distances.pos.kmeans[,1])
l2.distances.pos.group2 <- sort(l2.distances.pos.kmeans[,2])
l2.distances.pos.group3 <- sort(l2.distances.pos.kmeans[,3])
l2.distances.pos.group4 <- sort(l2.distances.pos.kmeans[,4])

# 2.3.3. choose the nearest national instances from each army clusters
new.positive.indices.group1 <- names(l2.distances.pos.group1[1:n.national.positive.per.cluster])
new.positive.indices.group2 <- names(l2.distances.pos.group2[1:n.national.positive.per.cluster])
new.positive.indices.group3 <- names(l2.distances.pos.group3[1:n.national.positive.per.cluster])
new.positive.indices.group4 <- names(l2.distances.pos.group4[1:n.national.positive.per.cluster])

new.positive.instances <- national.pos[c(new.positive.indices.group1, new.positive.indices.group2, new.positive.indices.group3, new.positive.indices.group4), ]
positive.combined <- rbind(army, new.positive.instances)


####3. Choose negative national instances####

#3.1. Standardize the data, since we will deal with l2 distance, which is sensitive to the scale.
predictors.negative <- rbind(army.predictors, national.predictors.neg)
preProcValues <-preProcess(predictors.negative, method = "range")
predictors.negative.scaled <- predict(preProcValues, predictors.negative)

#3.2. calculate distance from national instance to each cluster in #2.2
predictors.army.for.distance.neg <- predictors.negative.scaled[1:8, ]
predictors.national.neg.for.distance <- predictors.negative.scaled[-(1:8), ]

l2.distances.neg <- matrix(NA, nrow = nrow(national.neg), ncol = nrow(army.predictors))
rownames(l2.distances.neg) <- rownames(national.predictors.neg)
colnames(l2.distances.neg) <- rownames(army.predictors)

for (army.num in rownames(predictors.army.for.distance.neg)){
  army.vector <- unlist(predictors.army.for.distance.neg[army.num, ])
  l2.distances.neg[ , army.num] <- sqrt(
    apply( 
      ( t(as.matrix(predictors.national.neg.for.distance)) - army.vector )^2,
      # in R, matrix - vector is applied column-wise.
      # we want to subtract row-wise, so apply t().
      2, sum)
    # since we applies t(), sum is applied column-wise.
  )
}

# 3.3. calculate mean distance(mean over all army instances)
l2.distances.neg.mean <- apply(l2.distances.neg, 1, mean)
l2.distances.neg.mean <- sort(l2.distances.neg.mean, decreasing = TRUE)

# 3.4. choose the nearest national instances from each army instances. 
##IMPORTANT! we choose only the negative instances, to prevent making the problem overly easy.

# this makes the problem too easy(try it)
indices.seq <- 1:n.negative 
#negative.indices.by.distance <- names(l2.distances.neg.mean[seq.index])

#take only odd indices 

indices.seq.jump <- seq(1, n.negative*jump.step, jump.step)
negative.indices.by.distance <- names(l2.distances.neg.mean[indices.seq.jump])

negitive.instances <- national.neg[negative.indices.by.distance, ] #Choose the samples that are most different from the army data.


# 4. combine the chosen positive and negative instances

data.full <- rbind(positive.combined, negitive.instances)
data.full$DMG <- factor(data.full$DMG, levels = c(0, 1))
levels(data.full$DMG)<- c("neg", "pos")
names(data.full)[5] <- "y"### 1.2.1. data generation imbalance ratio
write.csv(data.full, file = "data_manipulated.csv")


# 5. plot.
plot(data.full, col = data.full$y)
