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

devtools::install_github('dmlc/xgboost',subdir='R-package')
library(xgboost)
require(methods)
##########################################################################
########Cleaning up training dataset #####################################
##########################################################################

print("Data Cleaning up process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)

train = train[,-1]
test = test[,-1]
set.seed(1234)

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 16,
              "bst:eta" = .3,
              "bst:max_depth" = 30,
              "lambda" = 1,
              "lambda_bias" = 0,
              "gamma" = 1,
              "alpha" = .8,
              "min_child_weight" = 3,
              "subsample" = .9,
              "colsample_bytree" = .9)

# Run Cross Valication
cv.nround = 91
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, nfold = 10, nrounds=cv.nround)



# Train the model
nround = 91
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='output/xgboost_6.csv', quote=FALSE,row.names=FALSE)


###############################
##### t-SNE Visualization #####
###############################
library(ggplot2)
library(readr)
library(Rtsne)

features <- train[, c(-1, -95)]

tsne <- Rtsne(as.matrix(features), check_duplicates = FALSE, pca = TRUE, perplexity = 30, theta = 0.5, dims = 2)

embedding <- as.data.frame(tsne$Y)
embedding$Class <- as.factor(sub("Class_", "", train[,95]))

pic <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) + geom_point(size=1.25)
pic <- pic + guides(colour = guide_legend(override.aes = list(size=6)))
pic <- pic + xlab("") + ylab("") + ggtitle("t-SNE 2D Embedding of Products Data")
pic <- pic + theme_light(base_size=20) + theme(strip.background = element_blank(),
                                               strip.text.x     = element_blank(),
                                               axis.text.x      = element_blank(),
                                               axis.text.y      = element_blank(),
                                               axis.ticks       = element_blank(),
                                               axis.line        = element_blank(),
                                               panel.border     = element_blank())
pic

ggsave("graph/tsne.png", pic, width=8, height=6, units="in")


### PCA

res <- prcomp(train[, 2:94], center = TRUE, scale = FALSE)
names(res)

explain90 <- min(which(cumsum(res$sdev^2/sum(res$sdev^2)) > 0.9))
print(cumsum(res$sdev^2/sum(res$sdev^2))[explain90])
pic.pca <- ggplot( ) + geom_point(aes(x = 1:93, y = cumsum(res$sdev^2/sum(res$sdev^2))), colour = "#5DA5DA", size = 3) 
pic.pca <- pic.pca + geom_point(aes(x = explain90, y = cumsum(res$sdev^2/sum(res$sdev^2))[explain90]), colour = "#F15854", size = 3)
pic.pca <- pic.pca + annotate("text", x = 43, y = 0.93, label = paste(as.character(round(cumsum(res$sdev^2/sum(res$sdev^2))[explain90], digits = 4) * 100), "%", sep = ""))
pic.pca <- pic.pca + annotate("text", x = 43, y = 0.88, label = "PC43")
pic.pca <- pic.pca + ggtitle("Principal Component Analysis") + xlab("Principal Component") + ylab("Variance Explained")
pic.pca
ggsave("graph/pca.png", pic.pca, width=8, height=6, units="in")


trunc <- res$x[,1 : explain90] %*% t(res$rotation[,1 : explain90])
if(res$scale != FALSE){
  trunc <- scale(trunc, center = FALSE , scale=1/res$scale)
}
if(res$center != FALSE){
  trunc <- scale(trunc, center = -1 * res$center, scale=FALSE)
}

train.pca <- train
train.pca[, 2 : 94] <- trunc


### Distribution of Classes in Training set:
pic1 <- ggplot(data = NULL, aes(x = as.character(unique(train$target)), y = as.numeric(table(train$target)))) + geom_bar(stat="identity", fill = "#5DA5DA")
pic1 <- pic1 + ggtitle("Class Distribution") + xlab("Classes") + ylab("Count")
pic1 <- pic1 + geom_text(aes(label = as.numeric(table(train$target))), vjust = -0.2)
pic
ggsave("graph/dist.png", pic, width=8, height=6, units="in")




