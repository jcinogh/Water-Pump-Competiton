setwd("/users/johaidoo/Desktop/Water Pump Project")
outcome<-read.csv("0bf8bc6e-30d0-4c50-956a-603fc693d966.csv")
feature<-read.csv("4910797b-ee55-40a7-8668-10efd5c1b960.csv")
train<-merge(outcome,feature,by="id")
table(train$status_group)
summary(train)
library(Hmisc)
library(tabplot)
describe(train)
nonfeature<-c("id","logitude","latitude","")
### Make a vector of predictor names
predictors <- names(train)[!(names(train) %in% c("status_group","id","date_recorded"))]

tableplot(train[,c("status_group",predictors[1:10])])
tableplot(train[,c("status_group",predictors[11:20])])
tableplot(train[,c("status_group",predictors[21:30])])
tableplot(train[,c("status_group",predictors[31:38])])

#Handle missing data
count_missing<-function(x){
 return(sum(is.na(x)))
}
unlist(lapply(train[predictors],count_missing))


#remove near-zero variance
train<-train[sapply(train, function(x) 
  length(levels(factor(x,exclude=NULL)))>1)]


#partition data
library(caret)
inTrain<-createDataPartition(train$status_group, p = .8, list = FALSE)
train_water<-train[inTrain,]
test_water<-train[-inTrain,]

outcome<-"status_group"
omitt_var<-c("date_recorded","id")
#double  check predictors

predictors<-setdiff(names(train_water),c(outcome,omitt_var))

#Feature Hashing
library(FeatureHashing)





objTrain_hashed<-hashed.model.matrix(~., data=train_water[,predictors],2^20,FALSE) %>% as("dgCMatrix")
objTest_hashed<-hashed.model.matrix(~., data=test_water[,predictors], 2^20,FALSE) %>% as("dgCMatrix")


suppressPackageStartupMessages(library(glmnet))
#simple multinomial model
cv.glm<-cv.glmnet(objTrain_hashed,train_water[,outcome],family="multinomial",
                  type.measure="auc",alpha=0,parallel=TRUE)
glmpred<-predict(cv.glm,objTest_hashed,s="lambda.min",type="class")

plot(varImp(cv.glm,scale=F))
glmnet::auc(test_water[,outcome],glmpred)
library(pROC)
obs<-ifelse(test_water[,outcome]=="functional",1,ifelse(test_water[,outcome]=="functional needs repair",
                                                        2,3))
exp<-ifelse(glmpred=="functional",1,ifelse(glmpred=="functional needs repair",
                                                        2,3))
obs<-factor(obs,levels=c(1,2,3))
exp<-factor(exp,levels=c(1,2,3))
multiclass.roc(obs,exp)

table(test_water[,outcome],glmpred)

#fit gbm using h20
suppressPackageStartupMessages(library(h2o))
localH20=h2o.init()
h2o.checkClient(localH20)

newtrain<-as.h2o(localH20,train_water)
newtest<-as.h2o(localH20,test_water)

fitgbm<-h2o.gbm(y=outcome,x=eng.features,key="mygbm",
                distribution = "multinomial",newtrain, n.trees = 10, 
        interaction.depth = 5, n.minobsinnode = 10, shrinkage = 0.001, 
        n.bins = 20,
        importance = FALSE, nfolds =5,balance.classes = FALSE, 
        max.after.balance.size = 5)
#gbm.VI <-fitgbm@model$varimp[1:20,]
#print(gbm.VI)
#barplot(t(gbm.VI[1]),las=2,main="VI from GBM",cex.names=.8)


# Calculate performance measures at threshold that maximizes precision
waterpump.pred<-as.data.frame(h2o.predict(fitgbm,newtest,type="probs"))

#====================================================================================================
#fit one vrs all approach
#feature Engineering: Create three temporal data set
#====================================================================================================
pump.statuses<-as.character(unique(train_water$status_group))
gen_target_var<-function(x,sublevel){
  x$target<-ifelse(x$status_group==paste(sublevel),1,0)
  x$target<-factor(x$target,levels=c(0,1))
  x<-x[c("target",predictors)]
  return(x)
}
#model function
try.gbm<-function(x,modkey){
  tvar<-"target"
  features<-names(x)[names(x)!=tvar]
  x<-as.h2o(localH20,x)
  fitgbm<-h2o.gbm(y=tvar,x=features,key=modkey, distribution = "bernoulli",
                  x, n.trees = 15, 
                  interaction.depth = 5, n.minobsinnode = 10, shrinkage = 0.001, n.bins = 20,
                  importance = TRUE, nfolds = 5,balance.classes = FALSE, 
                  max.after.balance.size = 5)
  out<-list()
  out$model<-fitgbm
  out$VarImp<-fitgbm@model$varimp[1:20,]
  return(out)
}

#create train data sets
store.train<-list()
for(j in pump.statuses){
  cat(j)
  store.train[[j]]<-gen_target_var(train_water,paste(j))
}

#fit training model
store.gbm<-list()
for(p in names(store.train)){
  cat("Fitting gbm model for status_group:",p,"......................")
  store.gbm[[p]]<-try.gbm(store.train[[p]],gsub("[[:space:]]","",p))
}
#give me the top 30 var that overlap on all three models

allVI<-lapply(store.gbm,function(x) row.names(x[["VarImp"]]))
eng.features<-Reduce(function(x,y)intersect(x,y),allVI)

#feature engineering for test set 
store.test<-list()
for(j in pump.statuses){
  cat(j,"\n")
  store.test[[j]]<-gen_target_var(test_water,paste(j))
}

#get the probabilities for each class
prob.status<-list()
for(status in pump.statuses){
  cat("Predicting status group",status,"\n")
probs<-as.data.frame(h2o.predict(store.gbm[[status]]$model,as.h2o(localH20,
  store.test[[status]]),type="response"))
prob.status[[status]]<-probs$predict
}
tmp<-as.data.frame(do.call("cbind",prob.status))
#find max prob for each test sample and show class
test.pump.prediction<-NULL
for(c in 1:dim(test_water)[1]){
  pump.class<-names(which(sapply(prob.status,function(x)x[c])==max(sapply(prob.status,
  function(x)x[c]))))
  test.pump.prediction<-c(test.pump.prediction,pump.class)
}
outcome<-data.frame(pred=test.pump.prediction,actual=test_water$status_group)
dcast(outcome,pred~actual,"value.var")
#=======================================================================================
#Fit model using xgboost
#=======================================================================================
nlev<-length(levels(train_water[,outcome]))
response<-train_water[,outcome]
levels(response)<-1:nlev
y<-as.matrix(as.integer(response)-1)
#=========================================================================
#Perform a cross-fold validation and find the best accuracy
#==========================================================================
cv <- 5
cvDivider <- floor(nrow(train_water)/(cv+1))
#smallestError <- 100
for (depth in seq(2,20,1)) { 
  for (rounds in seq(50,250,50)) {
    str_accuracy <- c()
    indexCount <- 1
    for (cv in seq(1:cv)) {
      # assign chunk to data test
      dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
      dataTest <- train_water[dataTestIndex,]
      # everything else to train
  dataTrain<-train_water[-dataTestIndex,]
  dataTrain_dg<-hashed.model.matrix(~., data=dataTrain[,predictors],2^20,FALSE) %>% as("dgCMatrix")
  dataTest_dg<-hashed.model.matrix(~., data=dataTest[,predictors], 2^20,FALSE) %>% as("dgCMatrix")      
  resp<-train_water[-dataTestIndex,outcome]
  levels(resp)<-1:nlev
  resp<-as.matrix(as.integer(resp)-1)
      bst <- xgboost(data = dataTrain_dg,
                     label = resp,
                     max.depth=depth, nround=rounds,
                     objective = "multi:softprob", verbose=0,
                     num_class=nlev)
      gc()
      pred<-predict(bst, dataTest_dg) 
      
      pred <- matrix(pred, nrow=nlev, ncol=length(pred)/nlev)
      pred <-t(pred)
      pred <- max.col(pred, "last")
      pred.char<-toupper(levels(dataTest[,outcome])[pred])
      # confusion matrix
      xprob<-confusionMatrix(dataTest[,outcome],factor(tolower(pred.char)))
      accuracy<-xprob$overall[1]
      str_accuracy <- c(str_accuracy, accuracy)
    }
    if (mean(str_accuracy) > 0) {
      avg.accuracy = mean(str_accuracy)
      print(paste(depth,rounds,avg.accuracy))
    }  
  }
} 
#=========================================================================
#Fit xgboost based on tunning paramters
#==========================================================================


test.xgbm<-xgboost(data=objTrain_hashed,label=y,
                   max.depth = 11,
                   eta = 0.1, nthread = 8, nround = 150, objective = "multi:softprob",
                   num_class=nlev)
pred<-predict(test.xgbm, objTest_hashed) 

pred <- matrix(pred, nrow=nlev, ncol=length(pred)/nlev)
pred <-t(pred)
pred <- max.col(pred, "last")
pred.char<-toupper(levels(test_water[,outcome])[pred])
# confusion matrix
xprob<-confusionMatrix(test_water[,outcome],factor(tolower(pred.char)))
xprob$overall[1]
#=======================================================
#Variable of Importance
#=======================================================
model = xgb.dump(test.xgbm, with.stats=TRUE)
# get the feature real names
names <- dimnames(objTrain_hashed)[[2]]
# compute feature importance matrix
importance_matrix <- xgb.importance(names, model=bst)

# plot
gp = xgb.plot.importance(importance_matrix)
print(gp)
