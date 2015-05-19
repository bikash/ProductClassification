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

#devtools::install_github('dmlc/xgboost',subdir='R-package')
library(xgboost)
require(methods)
library(caret)
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
              "min_child_weight" = 4,
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
write.csv(pred,file='output/xgboost_7.csv', quote=FALSE,row.names=FALSE)




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





library(h2o)

setwd("/Users/bikash/repos/kaggle/ProductClassification/")
localH2O <- h2o.init(nthread=16, max_mem_size="40g", min_mem_size="40g")

train.full <- read.csv("data/train.csv", header=TRUE)
test.full <- read.csv("data/test.csv", header=TRUE)


#First I will create a separate predictions frame and CV structure since they will 
#be used for both (and should be able to be used with other models as well)---------
predictions <- train.full[, ncol(train.full)]

#Matrix for storing predicted probablities; needs to be dimensions of train.only
pred.prob <- model.matrix(~ predictions - 1, data = predictions) 
#Using caret to split the data for CV-----------------------------------------------
set.seed(1234)
cv.index <- createDataPartition(predictions, p = 0.25, list = FALSE)

#Removing class label column and IDs; setting IDs aside for use later---------------
train <- train.full[-ncol(train.full)]
ids <- train[, 1]
train <- train[-1]
test.ids <- test.full[, 1]
test <- test.full[, -1]

#Partitioning to training and cv sets-----------------------------------------------
train.only <- train[-cv.index,]
cv <- train[cv.index, ]

#Setting up train/cv formats for xgboost model--------------------------------------
y <- predictions[-cv.index] #Response variable from train.only
y <- gsub("Class_", "", y)  #Just class number
y <- as.integer(y) - 1

x <- rbind(train.only, cv)#Predictors from train.only and cv sets
x <- as.matrix(x)         #Converting to matrix
x <- matrix(as.numeric(x), nrow(x), ncol(x)) #Converting chr to num 

trind <- 1:length(y)      #index to identify training data
cvind <- (length(y)+1):nrow(x)

#Test set as matrix-----------------------------------------------------------------
x.test <- as.matrix(test.full)
x.test <- matrix(as.numeric(x.test), nrow(x.test), ncol(x.test))

###############################Xgboost Modeling#####################################
#Random search function used for tuning parameters----------------------------------
random_search <- function(n_set){
  #param is a list of parameters
  
  # Set necessary parameter
  param <- list("objective" = "multi:softprob",
                "max_depth"=6,
                "eta"=0.1,
                "subsample"=0.7,
                "colsample_bytree"= 1,
                "gamma"=2,
                "min_child_weight"=4,
                "eval_metric" = "mlogloss",
                "silent"=1,
                "num_class" = 9,
                "nthread" = 8)
  
  param_list <- list()
  
  for (i in seq(n_set)){
    
    ## n_par <- length(param)
    param$max_depth <- sample(3:10,1, replace=T)
    param$eta <- runif(1,0.01,0.6)
    param$subsample <- runif(1,0.1,1)
    param$colsample_bytree <- runif(1,0.1,1)
    param$min_child_weight <- sample(1:17,1, replace=T)
    param$gamma <- runif(1,0.1,10)
    param$min_child_weight <- sample(1:15,1, replace=T)
    param_list[[i]] <- param
    
  }
  
  return(param_list)
}

#Set of parameters to test----------------------------------------------------------
xgb.param <- random_search(10)
#Running CV before doing ensemble CV------------------------------------------------
cv.nround = 91
TrainRes <- matrix(, nrow=cv.nround, ncol=length(xgb.param))
TestRes <- matrix(, nrow= cv.nround, ncol=length(xgb.param))

for(i in 1:length(param)){
  print(paste0("CV Round", i))
  bst.cv <- xgb.cv(param = xgb.param[[i]], data = x[trind,], label = y, 
                   nfold = 3, nrounds=cv.nround)
  TrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
  TestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
  
}

#Already found my best parameters in xgboost_script.R, so just inputting here-------
final.xgb.params <- list("objective" = "multi:softprob",
                         "max_depth"=9,
                         "eta"=0.05431259,
                         "subsample"=0.7851139 ,
                         "colsample_bytree"= 0.3923619,
                         "gamma"=0.5,
                         "min_child_weight"=5,
                         "eval_metric" = "mlogloss",
                         "silent"=1,
                         "num_class" = 9,
                         "nthread" = 8) 
#Training solo xgboost model--------------------------------------------------------
nround = 712
xgb.bst <- xgboost(param = final.xgb.params, data = x[trind,], label = y, 
                   nrounds = nround)


##################################CV before Ensembling##############################
xgb.model.prob <- predict(xgb.bst, newdata = x[cvind,])
xgb.model.prob <- t(matrix(xgb.model.prob,9,length(xgb.model.prob)/9))

#Calculating logloss----------------------------------------------------------------
#logloss function; unsure of how accurately this reflects LB calculations-----------
ll <- function(predicted, actual, eps = 1e-15){
  predicted[predicted < eps] <- eps
  predicted[predicted > 1 - eps] <- 1 - eps
  score <- -1/nrow(actual)*(sum(actual*log(predicted)))
  score
}

#Calculating ll for individual models-----------------------------------------------
xgb.model.ll <- ll(xgb.model.prob, pred.prob[cv.index,])

################################Building Test Probabilities#########################
xgb.bst.prob.test <- predict(xgb.bst, newdata = x.test[, -1])




### Neural Net
model_nn <- nnet(x = train.full[,c(2:94)], y = train.full$target, data=train.full[,c(2:95)], size = 10, 
                 rang = 0.5, decay = 0.1, linout = FALSE, 
                 MaxNWts=10000, trace = FALSE)
Y_hat_ts_nn <- predict(model_nn, test.full) + 10^-15
result_nn <- max.col(Y_hat_ts_nn)


#Save-------------------------------------------------------------------------------

# Make prediction
pred = xgb.bst.prob.test
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='output/xgboost_8.csv', quote=FALSE,row.names=FALSE)

##LB 0.45963