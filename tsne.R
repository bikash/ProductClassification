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

library(Rtsne)
setwd("your/path/to/train")
train <- read.csv("train.csv", stringsAsFactors=FALSE)[, -1]

set.seed(1234)
tsne_out_train <- Rtsne(as.matrix(train[,1:93]), check_duplicates = FALSE, pca = TRUE, 
                        max_iter = 1000, perplexity=30, theta=0.5, dims=2, verbose=TRUE)

my_palette = c("red", "blue", "green", "brown", "magenta", "orange", "cyan", "black", "yellow")
palette(my_palette)

plot(tsne_out_train$Y, xlab="", ylab="", col=as.factor(train$target), pch=".", cex=4, axes=FALSE)

legend("bottomleft", c("1","2", "3", "4", "5", "6", "7", "8", "9"),  
       lty=c(1,1), lwd=c(5,5), col=my_palette, bty="n", cex = 0.7) 

palette("default")