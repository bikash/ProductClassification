## Question 1

library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
# 1. Subset the data to a training set and testing set based on the Case variable in the data set. 
# 2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings. 
# 3. In the final model what would be the final model prediction for cases with the following variable values:
#   a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2 
# b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100 
# c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100 
# d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2 

# 1) root 1009 373 PS (0.63032706 0.36967294)  
# 2) TotalIntenCh2< 45323.5 454  34 PS (0.92511013 0.07488987) *
#   3) TotalIntenCh2>=45323.5 555 216 WS (0.38918919 0.61081081)  
# 6) FiberWidthCh1< 9.673245 154  47 PS (0.69480519 0.30519481) *
#   7) FiberWidthCh1>=9.673245 401 109 WS (0.27182045 0.72817955) *
#   
#   
data <- segmentationOriginal
set.seed(125)
inTrain <- data$Case == "Train"
trainData <- data[inTrain, ]
testData <- data[!inTrain, ]
cartModel <- train(Class ~ ., data=trainData, method="rpart")
cartModel$finalModel
plot(cartModel$finalModel, uniform=T)
text(cartModel$finalModel, cex=0.8)

## Answer
# a. PS 
# b. WS 
# c. PS
# d. Not possible to predict 

# ## Question 2
# If K is small in a K-fold cross validation is the bias in the estimate of out-of-sample 
# (test set) accuracy smaller or bigger? If K is small is the variance in the estimate of 
# out-of-sample (test set) accuracy smaller or bigger. Is K large or small in leave one out cross validation?

### --The bias is larger and the variance is smaller. Under leave one out cross validation K is equal to the sample size.



# ## Question 3
# Load the olive oil data using the commands:
#   
#   library(pgmm)
# data(olive)
# olive = olive[,-1]
# (NOTE: If you have trouble installing the pgmm package, you can download the olive dataset here: 
#olive_data.zip. After unzipping the archive, you can load the file using the load() function in R.)
# These data contain information on 572 different Italian olive oils from multiple regions in Italy. Fit a classification tree where Area is the outcome variable. Then predict the value of area for the following data frame using the tree command with all defaults
# 

library(pgmm)
data(olive)
olive = olive[,-1]
library(randomForest)

#Fit a classification tree where Area is the outcome variable. 
# Then predict the value of area for the following data frame using the tree command with all defaults
#These data contain information on 572 different Italian olive oils from multiple regions in Italy. Fit a classification tree where Area is the outcome variable. Then predict the value of area for the following data frame using the tree command with all defaults
model <- train(Area ~ ., data = olive, method = "rpart2")

newdata = as.data.frame(t(colMeans(olive)))

predict(model, newdata = newdata)

# 2.875. It is strange because Area should be a qualitative variable - but tree 
# is reporting the average value of Area as a numeric variable in the leaf predicted for newdata
#2.875. It is strange because Area should be a qualitative variable - but tree is reporting the average value of Area as a numeric variable in the leaf predicted for newdata




## qquestion 4
###Load the South Africa Heart Disease Data and create training and test sets with the following code:
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

set.seed(13234)

# Then set the seed to 13234 and fit a logistic regression model (method="glm", 
# be sure to specify family="binomial") with Coronary Heart Disease (chd) as the outcome and age at onset,
# current alcohol consumption, obesity levels, cumulative tabacco, type-A behavior, and low density lipoprotein 
# cholesterol as predictors. 
# Calculate the misclassification rate for your model using this function and a prediction on the "response" scale:

model <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, 
               data = trainSA, method = "glm", family = "binomial")

predtrain = predict(model, newdata = trainSA)
predtest = predict(model, newdata = testSA)
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

##missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
##What is the misclassification rate on the training set? What is the misclassification rate on the test set?

## for training set
missClass(trainSA$chd,predtrain) ## 0.2727273

# Test Set Misclassification rate
missClass(testSA$chd, predtest) # 0.3116883

## Answer
# Test Set Misclassification: 0.31 
# Training Set: 0.27

## Question 5
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 


set.seed(33833)

# Fit a random forest predictor relating the factor variable y to the remaining variables.
a <- randomForest(y ~ ., data = vowel.train, importance = FALSE)
b <- varImp(a)
order(b)

##he order of the variables is:
## x.2, x.1, x.5, x.6, x.8, x.4, x.9, x.3, x.7,x.10

