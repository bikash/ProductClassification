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

library(dplyr)
library(zoo)
library(randomForest)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(lattice)
library(Amelia) ## Amelia is packages to display missing data using missmap function

library(mclust)
sessionInfo()

##########################################################################
########Cleaning up training dataset #####################################
##########################################################################
print("Data Cleaning up process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)
##########################################################################

head(train) ## column name -> id, feat_1,.......feat_93, target
train1 = train[1:5000,]
test1 = train[5001:10000,]

## boosting
fit <- boosting(target~., data=train1, boos=TRUE, mfinal=10)
pred <- predict.boosting(fit,newdata=test1)
pred$class
pred$error

pdf("graph/missmap.pdf",bg="white")
## Display missing data from training data sets
missmap(train1, main="Training Data - Missings Map", col=c("green", "red"), legend=FALSE)
dev.off()

# remove id column so it doesn't get picked up by the random forest classifier
train2 <- train1[,-1]

## decision treee
model <- rpart(target ~ ., method = "class", data = train2)
plot(model)
text(model)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(model)
prp(model) ## Fast way of plotting
## Random Forest
#which(is.na(train1$id))
#summary(train1$id)

train2 <- train[,-1]
# set a unique seed number so you get the same results everytime you run the below model,
# the number does not matter
set.seed(12)
fit <- randomForest(as.factor(target) ~ ., data=train2, ntree=500, method = "class")

#fit <- randomForest(x= train, y = train$target, data=train, ntree=500, method = "class")

pdf("graph/varimpPlot.pdf",bg="white")
# Look at variable importance
varImpPlot(fit)
dev.off()

# use the random forest model to create a prediction
pred <- predict(fit,test,type="prob")
submit <- data.frame(id = test$id, pred)
write.csv(submit, file = "output/submit_1.csv", row.names = FALSE)


### Condition Inference Random Forest
library(party)
print("Prediction using Condition Inference Random Forest......")
c.fit <- cforest(as.factor(target) ~ .,
               data = train2, controls=cforest_unbiased(ntree=50, mtry=3))

## Tree structure
c.fit = ctree(as.factor(target) ~.,
            data = train2,   
            controls = ctree_control(
              teststat="quad",
              testtype="Univariate",
              mincriterion=.95,
              minsplit=10, 
              minbucket=5,
              maxdepth=0
            ))
prp(c.fit)
fancyRpartPlot(c.fit)

#prediction
Prediction <- predict(c.fit, test, OOB=TRUE, type = "response")
#prop.table(table(test$target, Prediction),1)
## calculate accuracy of model
#accuracy = sum(Prediction==test$target)/length(Prediction)
#print (sprintf("Accuracy = %3.2f %%",accuracy*100)) ### 82.84% accuracy of model
#########################################################################
out <- data.frame(id = test$id, Prediction)
write.csv(out, file = "output/ciRandomForest-predict1.csv", row.names = FALSE)
#########################################################################
#########################################################################

##SVM

