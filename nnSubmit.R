
#setwd("/Users/bikash/repos/kaggle/ProductClassification/")
setwd("/home/ekstern/haisen/bikash/kaggle/ProductClassification/")
require(caret); require(doParallel); require(randomForest); require(xgboost)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
##########################################################################
########Cleaning up training dataset #####################################
##########################################################################

print("Data Cleaning up process......")
raw_data <- read.csv("data/train.csv", header=TRUE)
testData <- read.csv("data/test.csv", header=TRUE)

##Metric
ll <- function(predicted, actual, eps=1e-15) {
  predicted[predicted < eps] <- eps
  predicted[predicted > 1 - eps] <- 1 - eps
  score <- -1/nrow(actual)*(sum(actual*log(predicted)))
  score
}

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
# create features using 1st 3 components of each class-wise PCA
pca <- function(cl, pcaComp = 3) {
  pcomps <- predict(preProcess(all.data[isTrainSet, ][pred %in% cl, ][-cvset, ], method = c("pca"), pcaComp = pcaComp),
                    all.data)
  dimnames(pcomps)[[2]] <- paste(paste(cl, collapse = "_"), dimnames(pcomps)[[2]], sep = "_")
  return(pcomps)
}

pcomps <- do.call(cbind, lapply(levels(pred), pca))

# normalise all data
all.data <- scale(cbind(log1p(all.data), pcomps))

rm(pcomps, pca)

# Train models ------------------------------------------------------------
# RF + NNet + xgb
model1 <- train(x = all.data[isTrainSet, ][-cvset, ], 
                y = pred[-cvset], 
                method = "rf", ntree = 400,
                tuneLength = 8,
                trControl = trainControl(method = "oob", number = 1, repeats = 1))

model2 <- train(x = all.data[isTrainSet, ][-cvset, ],
                y = pred[-cvset], 
                method = "nnet", MaxNWts = 50000, maxit = 1000,
                tuneGrid = expand.grid(size = c(9, 10, 11), decay = seq(from = 0.1, to = 0.9, by = 0.1)),
                trControl = trainControl(method = "cv", number = 4, repeats = 1))

save(model1, model2, cvset, all.data, file = "model.RData")
load(file = "model.RData")

dtrain <- xgb.DMatrix(data = all.data[isTrainSet, ][-cvset, ], label = as.integer(pred[-cvset]) - 1)

model3.train <- function(){
  param <- list("objective"="multi:softprob",
                "max_depth"=6,
                "eta"=0.1,
                "subsample"=0.8,
                "colsample_bytree"= 0.8,
                "gamma"=2,
                "min_child_weight"=4,
                "eval_metric"="mlogloss",
                "num_class"=9
  )
  param$max_depth <- sample(3:7,1, replace=T)
  param$eta <- runif(1,0.01,0.6)
  param$subsample <- runif(1,0.1,1)
  param$colsample_bytree <- runif(1,0.1,1)
  param$min_child_weight <- sample(1:17,1, replace=T)
  param$gamma <- runif(1,0.1,10)
  param$min_child_weight <- sample(1:15,1, replace=T)
  
  modelcv <- xgb.cv(nfold = 2, nrounds = 1000, earlyStopRound = 5, params = param, data = dtrain)
  nrounds <- which.min(modelcv$test.mlogloss.mean)
  #     model <- xgb.train(nrounds = nrounds, params = param, data = dtrain)
  #     return(model)
  return(data.frame(param,
                    ll=min(modelcv$test.mlogloss.mean), 
                    nrounds, 
                    nrounds2=sum(modelcv$test.mlogloss.mean - min(modelcv$test.mlogloss.mean) > 0.005),
                    stringsAsFactors = FALSE
  ))
}

# xgb.grid <- data.frame(param, ll=NA, nrounds=NA, nrounds2=NA)

for(i in 1:100){    
  xgb.grid <- rbind(xgb.grid, model3.train())
  save(xgb.grid, file = "modelxgb.RData")
  print(i)
}

model3 <- xgb.train(params = as.list(xgb.grid[which.min(xgb.grid$ll), 1:9]), 
                    nrounds =  xgb.grid[which.min(xgb.grid$ll), "nrounds2"], 
                    data = dtrain)


# CV ----------------------------------------------------------------------
model1.prob <- predict(model1, newdata = all.data[isTrainSet, ][cvset, ], type = "prob")
model2.prob <- predict(model2, newdata = all.data[isTrainSet, ][cvset, ], type = "prob")
model3.prob <- t(matrix(predict(model3, newdata = all.data[isTrainSet, ][cvset, ]), nrow = 9))

model1.pred <- predict(model1, newdata = all.data[isTrainSet, ][cvset, ], type = "raw")
model2.pred <- predict(model2, newdata = all.data[isTrainSet, ][cvset, ], type = "raw")
model3.pred <- apply(model3.prob, 1, which.max)
model3.pred <- as.factor(model3.pred)
levels(model3.pred) <- levels(pred)

# confusion matrices + other stats
model1.cm <- confusionMatrix(model1.pred, pred[cvset])
model2.cm <- confusionMatrix(model2.pred, pred[cvset])
model3.cm <- confusionMatrix(model3.pred, pred[cvset])

model1.cm
model2.cm
model3.cm

# log-loss
model1.ll <- ll(model1.prob, pred.prob[cvset, ])
model2.ll <- ll(model2.prob, pred.prob[cvset, ])
model3.ll <- ll(model3.prob, pred.prob[cvset, ])

model1.ll
model2.ll
model3.ll

# Ensemble ----------------------------------------------------------------
weight.grid <- data.frame(expand.grid(w1 = seq(from = 0.01, to = 1, by = 0.05), 
                                      w2 = seq(from = 0.01, to = 1, by = 0.05),
                                      w3 = seq(from = 0.01, to = 1, by = 0.05)), 
                          ll=NA)

for(x in 1:nrow(weight.grid)){
  cv.prob.test <- 
    ((model1.prob * weight.grid$w1[x]) + 
       (model2.prob * weight.grid$w2[x]) +
       (model3.prob * weight.grid$w3[x])) / 
    sum(weight.grid[x, c("w1","w2","w3")])
  weight.grid$ll[x] <- ll(cv.prob.test, pred.prob[cvset, ])
  print(x)
}
weight.best <- which.min(weight.grid$ll)
weight.grid[weight.best, ]

# Build test --------------------------------------------------------------
model1.prob.test <- predict(model1, newdata = all.data[!isTrainSet, ], type = "prob")
model2.prob.test <- predict(model2, newdata = all.data[!isTrainSet, ], type = "prob")
model3.prob.test <- t(matrix(predict(model3, newdata = all.data[!isTrainSet, ]), nrow = 9))

test.prob <- 
  ((model1.prob.test * weight.grid[weight.best, "w1"]) +
     (model2.prob.test * weight.grid[weight.best, "w2"]) +
     (model3.prob.test * weight.grid[weight.best, "w3"])
  ) / sum(
    weight.grid[weight.best, c("w1","w2","w3")]
  )

# Save --------------------------------------------------------------------
write.csv(cbind(data.frame(id = ids[!isTrainSet]), round(test.prob, digits = 6)), file = "output/Submission7.csv", row.names = FALSE)
# Local Score 0.4686704
# LB Score 0.47360