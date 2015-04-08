
#################################################################################################
#################################################################################################
## Author: Bikash Agrawal
## Date: 12th March Feb 2015
## Email: er.bikash21@gmail.com
## Description: SVM to classify product in Otto Group Kaggle Product classification competition.
##     http://www.kaggle.com/c/otto-group-product-classification-challenge
##
## References: 
## [1] http://dnene.bitbucket.org/docs/mlclass-notes/lecture16.html
## [2] 
## 
#################################################################################################
#################################################################################################



# set seed
set.seed(1337)

### setting path of repo folder.
getwd()
setwd("/Users/bikash/repos/kaggle/ProductClassification/")

#load data
##########################################################################
########Cleaning up training dataset #####################################
##########################################################################
print("Data Cleaning up process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)
##########################################################################

library(e1071)
train2 <- train[,-1] 
test2 <- test[,-1] 
attach(train2) 
attach(test2) 
model <- svm( train2$target~., train2 ) 
res <- predict( model, test2 ) 
submit <- data.frame(id = test$id, target = res) 
write.csv(submit, file = "output/svmsubmit1.csv", row.names = FALSE)