rm(list = ls())

library('nnet')
setwd("/Users/bikash/repos/kaggle/ProductClassification/")



#load the train.csv file
raw_data <- read.csv("data/train.csv")
#X is the data frame with all observations and 93 features
X <- raw_data[, 3:length(raw_data)-1]
#Y is the data frame with all observations but only target column
tr_Y <- raw_data[, length(raw_data)]

num_col <- ncol(X)
num_row <- nrow(X)

# Load test data set
testData <- read.csv('data/test.csv')
testData <- testData[,-1]
Num.test.totalSize <- dim(testData)[1]
Num.test.totalCol <- dim(testData)[2]

set.seed(1)

feature_size <- 30


feature_PCA <- prcomp(X, center = TRUE, scale. = TRUE)

tr_X <- predict(feature_PCA, X)
ts_X <- predict(feature_PCA, testData)

tr_X_sub <- tr_X[, 1:feature_size]
ts_X_sub <- ts_X[, 1:feature_size]
tr_X_sub <- as.data.frame(tr_X_sub)
ts_X_sub <- as.data.frame(ts_X_sub)
tr_Y_matrix <- class.ind(tr_Y)

##nn model
model_nn <- nnet(x = tr_X_sub, y = tr_Y_matrix, size = 9, 
                 rang = 0.5, decay = 0.1, linout = FALSE, 
                 MaxNWts=10000, trace = FALSE)
Y_hat_ts_nn <- predict(model_nn, ts_X_sub) + 10^-15
#result_nn <- max.col(Y_hat_ts_nn)

###knn model
#Y_hat_ts_knn <- knn(tr_X_sub, ts_X_sub, tr_Y, 50)
#result_knn <- class.ind(Y_hat_ts_knn)

# ###tree model
# tr_XY <- cbind(tr_X_sub, tr_Y)
# model_tree <- tree(tr_Y~., tr_XY)
# Y_hat_ts_tree <- predict(model_tree, ts_X_sub)
# result_tree <- max.col(Y_hat_ts_tree)

##combine and vote for class
result <-  Y_hat_ts_nn #+ result_knn + result_tree
final_result <- max.col(result)

result_table <- class.ind(final_result)

pred = data.frame(1:nrow(Y_hat_ts_nn),Y_hat_ts_nn)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred, file = "output/NN_PCA_result.csv", quote=FALSE,row.names=FALSE)