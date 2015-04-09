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

library(ada)

##########################################################################
########Cleaning up training dataset #####################################
##########################################################################
print("Data Cleaning up process......")
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)

### Label data
train$class[train$target == "Class_1"] <- 1
train$class[train$target == "Class_2"] <- 2
train$class[train$target == "Class_3"] <- 3
train$class[train$target == "Class_4"] <- 4
train$class[train$target == "Class_5"] <- 5
train$class[train$target == "Class_6"] <- 6
train$class[train$target == "Class_7"] <- 7
train$class[train$target == "Class_8"] <- 8
train$class[train$target == "Class_9"] <- 9
##########################################################################
train2 = train[,-1]
train2 = train2[,-94]
train1 = train2[1000:61878,]
test1 = train2[1:1000,]




library(rpart)
fit <- rpart(class ~ ., data=train1, method="class")
fancyRpartPlot(fit)
# prediction
pred <- predict(fit, test1, type = "class")
out <- data.frame(id = train$id[1:1000], class = pred)
write.csv(out, file = "output/Decision_tree.csv", row.names = FALSE)
## calculate accuracy of model
accuracy = sum(out$class==test1$class)/length(pred)
print (sprintf("Accuracy = %3.2f %%",accuracy*100)) ### 81.84% accuracy of model using random forest
#########################################################################



## Random Forest
fit <- randomForest(class ~ ., data=train1, ntree=500, method = "class")


default=rpart.control()

m <-glm(target~., data = train1)

gdis<-ada(target~.,data=train1,iter=100,loss="e",type="discrete",control=default)
plot(gdis)

pairs(gdis,train[,-1],maxvar=3)