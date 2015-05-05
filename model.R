library(e1071)
library(lubridate)
library(caret)
library(Metrics)
library(randomForest)
library(nnet)


### setting path of repo folder.
getwd()
setwd("/Users/bikash/repos/kaggle/RestaurantRevenuePrediction/")


train <- read.csv("data/train.csv",header=TRUE)
test <- read.csv("data/test.csv",header=TRUE)

train$day<-as.factor(day(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))
train$month<-as.factor(month(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))
train$year<-as.factor(year(as.POSIXlt(train$Open.Date, format="%m/%d/%Y")))

test$day<-as.factor(day(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))
test$month<-as.factor(month(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))
test$year<-as.factor(year(as.POSIXlt(test$Open.Date, format="%m/%d/%Y")))

train_cols<-train[,c(3:42,44:46)]
labels<-as.matrix(train[,43])
testdata<-test[,3:45]

train_cols <- data.frame(lapply(train_cols,as.numeric))
testdata<-data.frame(lapply(testdata,as.numeric))

train_cols<-train_cols[   which(labels<0.40*max(labels))   ,]
labels<-labels[ which(labels<0.40*max(labels))  ,]

#dimension reduction
trans_train = preProcess(train_cols, method=c("BoxCox", "center", "scale", "pca"))
train_pca = predict(trans_train, train_cols)
test_pca = predict(trans_train, testdata)
#fit SVM model
fit<- svm(x=as.matrix(train_pca),y=labels,cost=10,scale=TRUE,type="eps-regression")
predictions<-as.data.frame(predict(fit,newdata=test_pca))
#fit glm model
fit_glm<- glm(labels~as.matrix(train_pca))
predictions<-as.data.frame(predict(fit_glm,newdata=test_pca))
#fit rf model
fit_rf<- randomForest(labels~.,data=train_cols)
predictions<-as.data.frame(predict(fit_rf,newdata=testdata))
#fit rf model
fit_rf<- randomForest(labels~.,data=train_pca,type="regression",prox=TRUE, ntree=1000)
predictions<-as.data.frame(predict(fit_rf,newdata=test_pca))
#fit neural net model
m.ridge<-lm.ridge(labels ~ ., data = train_cols)
y.pred.ridge = scale(testdata,center = F, scale = m.ridge$scales)%*% m.ridge$coef[,which.min(m.ridge$GCV)] + m.ridge$ym
summary((y.pred.ridge - y.test)^2)
select(m.ridge)


library(lars)
m.lasso <- lars(x=as.matrix(train_cols),y=labels)
plot(m.lasso)
fits <- predict.lars(m.lasso, testdata, type="fit")


library(pls)
m.pls <- plsr(labels ~ .,data=train_cols , validation="CV")
# select number of components (by CV)
ncomp <- which.min(m.pls$validation$adj)
# predict
predictions <- predict(m.pls,testdata , ncomp=ncomp)


submit<-as.data.frame(cbind(test[,1],predictions))
colnames(submit)<-c("Id","Prediction")

write.csv(submit,"output/submission1.csv",row.names=FALSE,quote=FALSE)
