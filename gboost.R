#################################################################################################
#################################################################################################
## Author: Bikash Agrawal
## Date: 12th March Feb 2015
## Email: er.bikash21@gmail.com
## Description: Random Forest to classify product in Otto Group Kaggle Product classification competition.
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

library(xgboost)
require(methods)