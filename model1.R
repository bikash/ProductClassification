library(lubridate)
library(caret)
library(plyr)
library(AppliedPredictiveModeling)
now <- as.Date(format(Sys.time(), "%Y/%m/%d"))

readData <- function(mf = ""){
  mf <- read.table(mf, sep = ",", header = T, stringsAsFactors = TRUE)
  mf$Open.Date <- as.Date(as.character(mf$Open.Date), "%m/%d/%y")
  mf$Open.Duration <- log(as.numeric(abs(as.duration(interval(ymd(mf$Open.Date),
                                                              ymd(rep(now,nrow(mf))))))))
  
  # log of P
  P.log <- as.data.frame(log(1 + mf[,6:42]))
  names(P.log) <- paste(names(P.log),"log",sep = ".")
  #cbind(mf,P.log)
  
  # sqrt of P
  P.sqrt <- as.data.frame(log(1 + mf[,6:42]))
  names(P.sqrt) <- paste(names(P.sqrt),"sqrt",sep = ".")
  cbind(mf,P.log,P.sqrt)
  
}

### setting path of repo folder.
getwd()
setwd("/Users/bikash/repos/kaggle/RestaurantRevenuePrediction/")


train_set <- read.csv("data/train.csv",header=TRUE)
test_set  <- read.csv("data/test.csv",header=TRUE)

levels(train_set$Type) <- levels(test_set$Type)
levels(train_set$City) <- levels(test_set$City)
levels(train_set$City.Group) <- levels(test_set$City.Group)
#train_set <- train_set[!names(train_set) %in% c("City","City.Group")]

doMC::registerDoMC(2)

x1 <- "log2(revenue) ~ . - Id - Open.Date - City - Type - City.Group" # best
x2 <- "log2(revenue) ~ . - Id - Open.Date - City - Type"
x3 <- "log2(revenue) ~ . - Id - Open.Date - City - City.Group"
x4 <- "log2(revenue) ~ . - Id - Open.Date - City"
x5 <- "log2(revenue) ~ .^2 - Id - Open.Date - City - Type - City.Group"
x6 <- "log2(revenue) ~ .^2 - Id - Open.Date - City - Type"
x7 <- "log2(revenue) ~ .^2 - Id - Open.Date - City - City.Group"
x8 <- "log2(revenue) ~ .^2 - Id - Open.Date - City"


models_loocv <- llply(list(x1,x2,x3,x4), function(form){
  ctrl <- trainControl(method = "LOOCV",
                       allowParallel = T)
  caret::train(as.formula(form), 
               data = train_set, method = "pls", 
               trControl = ctrl, tuneGrid = expand.grid(ncomp = 1:5))
})

models_repeatedcv <- llply(list(x1,x2,x3,x4), function(form){
  ctrl <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 10,
                       allowParallel = T)
  caret::train(as.formula(form), 
               data = train_set, method = "pls", 
               trControl = ctrl, tuneGrid = expand.grid(ncomp = 1:5))
})
# lilo
models_interaction_repeatedcv <- llply(list(x5,x6,x7,x8), function(form){
  ctrl <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 10,
                       allowParallel = F)
  caret::train(as.formula(form), 
               data = train_set, method = "pls", 
               trControl = ctrl, tuneGrid = expand.grid(ncomp = 1:5))
})

# best model
ctrl <- trainControl(method = "LOOCV",
                     allowParallel = T)

rest_fit_log <- caret::train(log2(revenue) ~ . - Id - Open.Date - Type - City - City.Group, 
                             data = train_set, method = "pls", 
                             trControl = ctrl, tuneGrid = expand.grid(ncomp = 2:5))
rest_pred_log <- data.frame(Prediction = 2^predict(rest_fit_log, newdata = test_set))
res_out <- cbind(Id = as.numeric(rownames(rest_fit_log))-1, rest_fit_log)
write.table(res_out, file = "output/pred_log_best.csv", quote = F, sep = ",",row.names = F)


rest_pred <- data.frame(Prediction = 2^predict(rest_fit, newdata = test_set))