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
setwd("/Users/bikash/repos/kaggle/ProductClassification/")


library(xgboost) ## xgboost
require(methods)
require(data.table) ## viewing data
require(magrittr)
##########################################################################
########Cleaning up training dataset #####################################
##########################################################################

print("Data reading process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)


#The evaluation metric function used by caret to score
#a fold. Not optimal - needs vectorization
MCLogLoss <- function(data, lev = NULL, model = NULL)  {
  
  obs <- model.matrix(~data$obs - 1)
  preds <- data[, 3:ncol(data)]
  
  err = 0
  for(ob in 1:nrow(obs))
  {
    for(c in 1:ncol(preds))
    {
      p <- preds[ob, c]
      p <- min(p, 1 - 10e-15)
      p <- max(p, 10e-15)
      err = err + obs[ob, c] * log(p)
    }
  }
  
  out <- err / nrow(obs) * -1
  names(out) <- c("MCLogLoss")
  out
}



# Train dataset dimensions
dim(train) ## [1] 61878    95

# Training content
train[1:6,1:5]

# Test dataset dimensions
dim(test) #[1] 144368     94

## remove ID from test and train set
train = train[,-1]
test = test[,-1]
# Save the name of the last column target
nameLastCol <- names(train)[ncol(train)] ## target
y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

# Display the first 5 levels
y[1:5] ## [1] 0 0 0 0 0

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))

set.seed(1234)

testleng = (nrow(train)+1):nrow(x)

numberOfClasses <- max(y) + 1 ## 9 different classes


## important variable

library(Boruta)
important <- Boruta(target~., data=train)
important$finalDecision

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
cv.nround <- 91
cv.nfold <- 10

bst.cv = xgb.cv(param=param, data = x[1:length(y),], label = y, 
                nfold = cv.nfold, nrounds = cv.nround)
# Train the model
nround = 91
bst = xgboost(param=param, data =  x[1:length(y),], label = y, nrounds=nround)


model <- xgb.dump(bst, with.stats = T)
model[1:10]


# Make prediction
pred = predict(bst,x[testleng,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='output/xgboost_6.csv', quote=FALSE,row.names=FALSE)



## Combine RF+GBM
source("lib/gbm.fit.R")

#TUNE HERE
#How much of the data to use to build the randomForest
TRAIN_SPLIT = 0.7
RF_MTRY = 9
RF_TREES = 125
GBM_IDEPTH = 4
GBM_SHRINKAGE  = 0.1
GBM_TREES = 50
#TUNE HERE
library(caret)
target <- train$target 
train <- train[, -which(names(train)=="target")] 
id <- test$id 
test <- test[, -which(names(test)=="id")] 

#Split training data into two sets(keep class distribution)
set.seed(20739) 
trainIndex <- createDataPartition(target, p = TRAIN_SPLIT, list = TRUE, times = 1) 
allTrain <- train 
allTarget <- target 
train <- allTrain[trainIndex$Resample1, ] 
train2 <- allTrain[-trainIndex$Resample1, ] 
target <- allTarget[trainIndex$Resample1] 
target2 <- allTarget[-trainIndex$Resample1] 

#Build a randomForest using first training set
fc <- trainControl(method = "repeatedCV", 
                   number = 2, 
                   repeats = 1, 
                   verboseIter=FALSE, 
                   returnResamp="all", 
                   classProbs=TRUE) 
tGrid <- expand.grid(mtry = RF_MTRY) 
model <- train(x = train, y = target, method = "rf", 
               trControl = fc, tuneGrid = tGrid, metric = "Accuracy", ntree = RF_TREES) 
#Predict second training set, and test set using the randomForest
train2Preds <- predict(model, train2, type="prob") 
testPreds <- predict(model, test, type="prob")
model$finalModel

#Build a gbm using only the predictions of the
#randomForest on second training set
fc <- trainControl(method = "repeatedCV", 
                   number = 10, 
                   repeats = 1, 
                   #verboseIter=FALSE, 
                   #returnResamp="all", 
                   classProbs=TRUE, 
                   #summaryFunction=MCLogLoss,
                   ) 
tGrid <- expand.grid(interaction.depth = GBM_IDEPTH, shrinkage = GBM_SHRINKAGE, n.trees = GBM_TREES) 
model2 <- train(x = train2Preds, y = target2, method = "gbm", 
                trControl = fc, tuneGrid = tGrid,verbose = TRUE)
model2
hist(model2$resample$MCLogLoss)

#Build submission
submit <- predict(model2, testPreds, type="prob") 
# shrink the size of submission
submit <- format(submit, digits=2, scientific = FALSE)
submit <- cbind(id=1:nrow(testPreds), submit) 
write.csv(submit, "submit.csv", row.names=FALSE)



## Feature importance plot
# Get the feature real names
names <- dimnames(train[,-1])[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])
## tree graph
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)



##
pdf("graph/graph4.pdf")
data <- train
group_data<-split(data[,-94],data$target)
cent<-lapply(group_data,colMeans)
cent<-matrix(unlist(cent),9,93,T)
distance<-dist(cent)
dist<-matrix(0,9,9)
k=1
for(i in 1:8){
  for(j in (i+1):9){
    dist[j,i]<-distance[k]
    dist[i,j]<-dist[j,i]
    k=k+1
  }
}
colnames(dist)<-c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
rownames(dist)<-colnames(dist)
#see the distance of centers of each cluster
dist_heatmap<-heatmap(dist,Rowv=NA,Colv=NA,col=heat.colors(10),scale="column",margins=c(10,10),main="Euclidean distance between each cluster centre")

#ordinary pca
pca<-prcomp(~.,data[,-94])
pca_data<-data.frame(pca$x)
pca_group<-split(pca_data,data$target)
subindex<-sample(1:61878,800)
sub_data<-data[subindex,]
sub_pca_data<-pca_data[subindex,]
sub_pca_group<-split(sub_pca_data,data$target[subindex])
plot(sub_pca_group$Class_1[,c(1,2)],col="red",xlim=c(-30,20),ylim=c(-30,20),main="2-D Projection plot of 800 sample points.")
points(sub_pca_group$Class_2[,c(1,2)],pch=3,col="yellow")
points(sub_pca_group$Class_3[,c(1,2)],pch=2,col="blue")
points(sub_pca_group$Class_4[,c(1,2)],pch=20,col="grey")
points(sub_pca_group$Class_5[,c(1,2)],pch="o",col="black")
points(sub_pca_group$Class_6[,c(1,2)],pch="*",col="brown")
points(sub_pca_group$Class_7[,c(1,2)],pch=11,col="purple")
points(sub_pca_group$Class_8[,c(1,2)],pch=13,col="orange")
points(sub_pca_group$Class_9[,c(1,2)],pch=9,col="lavender")
legend(8,20,legend=colnames(dist),pch=c(1,3,2,20,111,42,11,13,9),col=c("yellow","blue","grey","black","brown","purple","orange","lavender"))



## Important Feature Correlations
library(Boruta)
library(corrplot)
num_rows_sample <- 2500

train_sample <- train[sample(1:nrow(train), size = num_rows_sample),]
features     <- train_sample[,c(-1, -95)]

boruta <- Boruta(features, as.factor(train_sample$target),
                 getImp=function(x, y) getImpRfZ(x, y, ntree=20),
                 doTrace=2, maxRuns=30)

features_imp <- features[,boruta$finalDecision=="Confirmed"]

top20_feature_names <- names(features_imp)[order(-getImpRfZ(features_imp, as.factor(train_sample$target), ntree=100))[1:20]]

features_top20 <- train[,top20_feature_names]
names(features_top20) <- gsub("feat_", "", names(features_top20))

#png(filename = "graph/feature_correlations.png", width=8, height=8, res=200, units="in")
corrplot.mixed(cor(features_top20), lower="ellipse", upper="shade",
               order="hclust", hclust.method="complete")


dev.off()
