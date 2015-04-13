#install and load the neural network package
install.packages("nnet")
library(nnet)

### setting path of repo folder.
getwd()
setwd("/Users/bikash/repos/kaggle/ProductClassification/")


#set the seed so that the results are reproducible
set.seed(342)

#read the data
train<-read.csv("data/train.csv")
test<-read.csv("data/test.csv")

#fit a neural network model
#try different values of the parameters to obtain a better score 
fit<-nnet(target ~ ., train[,-1], size = 3, rang = 0.1, decay = 5e-4, maxit = 500) 

#predict on the test data
predicted<-as.data.frame(predict(fit,test[,-1],type="raw"))  

#create the submission file
id<-test[,1]
output<-cbind(id,predicted) 
write.csv(output,"output/nnet_submission.csv",row.names=FALSE)