# Library -----------------------------------------------------------------
require(caret); require(doParallel); require(glmnet); require(randomForest)

ll <- function(predicted, actual, eps=1e-15) {
  predicted[predicted < eps] <- eps
  predicted[predicted > 1 - eps] <- 1 - eps
  score <- -1/nrow(actual)*(sum(actual*log(predicted)))
  score
}

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Load --------------------------------------------------------------------
setwd("/Users/bikash/repos/kaggle/ProductClassification/")

##########################################################################
########Cleaning up training dataset #####################################
##########################################################################

print("Data Cleaning up process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)


#fit a neural network model
#try different values of the parameters to obtain a better score 
fit<-nnet(target ~ ., train, size = 3, rang = 0.1, decay = 5e-4, maxit = 500) 

#predict on the test data
predicted<-as.data.frame(predict(fit,test[,-1],type="raw"))


# Ground truth ------------------------------------------------------------
pred <- train[ , ncol(train), drop = TRUE]
pred.prob <- model.matrix( ~ pred - 1, data = pred)

# Partition vectors -------------------------------------------------------
cvset <- createDataPartition(pred, p = 0.2, list = FALSE)
isTrainSet <- c(rep(TRUE, nrow(train)), rep(FALSE, nrow(test)))

# Combine -----------------------------------------------------------------
train <- train[-ncol(train)]
all.data <- rbind(train, test)

# IDs ---------------------------------------------------------------------
ids <- all.data[ , 1, drop = TRUE]
all.data <- all.data[-1]
train <- train[-1]







# Pre-process -------------------------------------------------------------
# PCA on all data
pcomps <- predict(preProcess(all.data[isTrainSet, ][-cvset, ], method = c("pca"), thresh = 0.8), all.data)

# distance to k-means cluster centres using log data
km <- kmeans(log1p(all.data[isTrainSet, ][-cvset, ]), 9)
km.dist <- lapply(1:9, function(x) apply(abs(km$centers[x, ] - log1p(all.data[ , ])), 1, sum))
km.dist <- do.call(cbind, km.dist)
dimnames(km.dist)[[2]] <- paste0("km_dist_", 1:9)

# create features using 1st 3 components of each class-pair PCA
pca <- function(cl, pcaComp = 3) {
  pcomps <- predict(preProcess(all.data[isTrainSet, ][as.numeric(pred) %in% cl, ][-cvset, ], method = c("pca"), pcaComp = pcaComp),
                    all.data)
  dimnames(pcomps)[[2]] <- paste(paste(cl, collapse = "_"), dimnames(pcomps)[[2]], sep = "_")
  return(pcomps)
}

pcomps.pairs <- do.call(cbind, apply(combn(9,2), 2, pca))

# normalise all data
all.data <- scale(cbind(all.data, pcomps, km.dist, pcomps.pairs))

rm(pcomps, km.dist, pcomps.pairs, km, pca)

# Train models ------------------------------------------------------------
# RF + NNet
model1 <- randomForest(x = all.data[isTrainSet, ][-cvset, ], y = pred[-cvset], ntree = 1000, mtry = 112, sampsize = 10000, do.trace = TRUE,      method = "rf")

model2 <- train(x = all.data[isTrainSet, ][-cvset, ],
                y = pred[-cvset], 
                method = "nnet", MaxNWts = 50000, maxit = 1000,
                tuneGrid = expand.grid(size = c(8, 9), decay = seq(from = 0.1, to = 0.9, by = 0.1)),
                trControl = trainControl(method = "cv", number = 2, repeats = 1, verboseIter = TRUE, classProbs = TRUE))

# CV ----------------------------------------------------------------------
model1.pred <- predict(model1, newdata = all.data[isTrainSet, ][cvset, ], type = "response")
model2.pred <- predict(model2, newdata = all.data[isTrainSet, ][cvset, ], type = "raw")

model1.prob <- predict(model1, newdata = all.data[isTrainSet, ][cvset, ], type = "prob")
model2.prob <- predict(model2, newdata = all.data[isTrainSet, ][cvset, ], type = "prob")

# confusion matrices + other stats
model1.cm <- confusionMatrix(model1.pred, pred[cvset])
model2.cm <- confusionMatrix(model2.pred, pred[cvset])

model1.cm
model2.cm

# log-loss
ll(model1.prob, pred.prob[cvset, ])
ll(model2.prob, pred.prob[cvset, ])


# Ensemble ----------------------------------------------------------------
model1.prob.test <- predict(model1, newdata = all.data[!isTrainSet, ], type = "prob")
model2.prob.test <- predict(model2, newdata = all.data[!isTrainSet, ], type = "prob")

model1.weight <- model1.cm$overall["Accuracy"]
model2.weight <- model2.cm$overall["Accuracy"]

cv.prob <- ((model1.prob * model1.weight) 
            + (model2.prob * model2.weight)) / (
              model1.weight + model2.weight)

ll(cv.prob, pred.prob[cvset, ])

# Calibrate ---------------------------------------------------------------
model3 <- train(x = cv.prob, 
                y = pred[cvset], 
                method = "gbm",
                tuneLength = 8,
                trControl = trainControl(method = "cv", number = 2, repeats = 1, verboseIter = TRUE))

ll(predict(model3, cv.prob, type = "prob"), pred.prob[cvset, ])

test.prob <- ((model1.prob.test * model1.weight) +
                (model2.prob.test * model2.weight)) /
  (model1.weight + model2.weight)

test.prob <- predict(model3, test.prob, type = "prob")

# Save --------------------------------------------------------------------
write.csv(cbind(data.frame(id = ids[!isTrainSet]), round(test.prob, digits = 6)), file = "./Data/Submission5.csv", row.names = FALSE)
# Local Score ?
# LB Score 0.52669