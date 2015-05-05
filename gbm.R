#################################################################################################
#################################################################################################
## Author: Bikash Agrawal
## Date: 12th March Feb 2015
## Email: er.bikash21@gmail.com
## Description: Gradient boosting machine to classify product in Otto Group Kaggle Product classification competition.
##     http://www.kaggle.com/c/otto-group-product-classification-challenge
##
## References: 
## [1] http://dnene.bitbucket.org/docs/mlclass-notes/lecture16.html
## [2] http://cran.r-project.org/web/packages/gbm/gbm.pdf
## 
#################################################################################################
#################################################################################################

### setting path of repo folder.
getwd()
setwd("/Users/bikash/repos/kaggle/ProductClassification/")

devtools::install_github('dmlc/xgboost',subdir='R-package')
library(gbm)
require(methods)
##########################################################################
########Cleaning up training dataset #####################################
##########################################################################

print("Data Cleaning up process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)

train = train[,-1]
test = test[,-1]



fit <- gbm(target~.,data=train, cv.folds=10, n.trees=5000, distribution="gaussian", interaction.depth=3, bag.fraction=0.5, train.fraction=1.0, shrinkage=0.05, keep.data=TRUE)

pred = predict(fit,test)


  
pred = format(pred, digits=2,scientific=F) # shrink the size of submission


pred = data.frame(test$id,pred)
names(pred) = c('id', paste0('Class_',1:9))

write.csv(pred,file='output/gbm.csv', quote=FALSE,row.names=FALSE)