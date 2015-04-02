#################################################################################################
#################################################################################################
## Author: Bikash Agrawal
## Date: 12th March Feb 2015
## Email: er.bikash21@gmail.com
## Description: KNN to classify product in Otto Group Kaggle Product classification competition.
##     http://www.kaggle.com/c/otto-group-product-classification-challenge
##
## References: 
## [1] http://dnene.bitbucket.org/docs/mlclass-notes/lecture16.html
## [2] 
## 
#################################################################################################
#################################################################################################

### setting path of repo folder.
getwd()
setwd("/Users/bikash/repos/kaggle/ProductClassification/")

# load dependencies
library(FNN)

#load data
##########################################################################
########Cleaning up training dataset #####################################
##########################################################################
print("Data Cleaning up process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)
##########################################################################

# make target a factor
train$target = as.factor(train$target)

#
classes <- train[,95]

# remove target
train <- train[,-95]

# remove ID cols
train <- train[,-1]
test <- test[,-1]

results <- data.frame(knn(train, test, classes, k = 9, prob=TRUE))

names(results) <- c("Label")
results$ImageId <- 1:nrow(results)
results <- results[c(2,1)]

write.csv(results, file = "output/knn_submit.csv", quote = FALSE, row.names = FALSE)