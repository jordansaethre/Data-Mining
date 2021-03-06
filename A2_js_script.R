### Assignment 2 Bank Decision Tree and Na�ve Bayes Classification

# 1. A.

# load packages
# C50, caret, e1071, knitr, matrixStats rmarkdown, and rminer

library(C50)
library(caret)
library(e1071)
library(knitr)
library(matrixStats)
library(rmarkdown)
library(rminer)

# import data (factors as character strings first)

setwd("C:/Users/jordan.saethre/Documents/IS 6482")

inputfile <- "C:/Users/jordan.saethre/Documents/IS 6482/bank_full.csv"

bank_full <- read.csv(file = inputfile, stringsAsFactors = FALSE)

# transform string variables to factors

bank_full <- read.csv(file = inputfile, stringsAsFactors = TRUE)

# Structure of Bank Data

str(bank_full)

# Summary of Bank Data

summary(bank_full)

# 1. B. 

# Partition the data set for simple hold-out evaluation - 
# 50% for training and the remaining 50% for testing.

# set seed to a value for createDataPartition(). With the same value and input, 
# the partitions output will be consistent each time the following commands are executed.

set.seed(100)

inTrain <- createDataPartition(bank_full$deposit, p=0.5, list=FALSE)

# inTrain is a list of indices to the rows in the bank_full data frame

# Assign the rows in bank_full indexed by inTrain to create a train set
# Assign all other rows indexed by -inTrain to create a test set

bank_fullTrain <- bank_full[inTrain,]
bank_fullTest <- bank_full[-inTrain,]

# 1. C. 

# Show the overall structure and summary of train and test sets.

str(bank_fullTrain$deposit)
summary(bank_fullTrain$deposit)

str(bank_fullTest$deposit)
summary(bank_fullTest$deposit)

# Show the distributions of deposit in the entire set, the train set and the test set.

# Distribution of deposit in entire set:

prop.table(table(bank_full$deposit))

# Distribution of deposit in train set:

prop.table(table(bank_fullTrain$deposit))

# Distribution of deposit in test set:

prop.table(table(bank_fullTest$deposit))

# 2. A. 

# Train a C5.0 model using the default setting. 
# Show information about this model and the summary of the model. 

bank_full.c50 <- C5.0(bank_fullTrain$deposit~.,bank_fullTrain)
bank_full.c50

summary(bank_full.c50)

# Generate and compare this model's confusion matrices and classification 
# evaluation metrics in test and train sets.

predicted_deposit_test1 <- predict(bank_full.c50, bank_fullTest)

mmetric(bank_fullTest$deposit, predicted_deposit_test1, metric="CONF")

mmetric(bank_fullTest$deposit, predicted_deposit_test1, metric=c("ACC","TPR","PRECISION","F1"))

predicted_deposit_train1 <- predict(bank_full.c50, bank_fullTrain)

mmetric(bank_fullTrain$deposit, predicted_deposit_train1, metric="CONF")

mmetric(bank_fullTrain$deposit, predicted_deposit_train1, metric=c("ACC","TPR","PRECISION","F1"))

# 2. B. 

# Explore increasing the tree complexity by choosing a relatively high CF levels 
# (e.g. 0.7 or higher). In the code, select a CF level of your choice to train and test 
# another C5.0 model.  Show information about this model and the summary of the model. 

bank_full.c50.2 <- C5.0(bank_fullTrain$deposit~., bank_fullTrain, control = C5.0Control(CF = 0.8))
bank_full.c50.2
summary(bank_full.c50.2)

# Generate and compare this model's confusion matrices and classification evaluation metrics 
# in test and train sets.

predicted_deposit_test2 <- predict(bank_full.c50.2, bank_fullTest)

mmetric(bank_fullTest$deposit, predicted_deposit_test2, metric="CONF")

mmetric(bank_fullTest$deposit, predicted_deposit_test2, metric=c("ACC","TPR","PRECISION","F1"))

predicted_deposit_train2 <- predict(bank_full.c50.2, bank_fullTrain)

mmetric(bank_fullTrain$deposit, predicted_deposit_train2, metric="CONF")

mmetric(bank_fullTrain$deposit, predicted_deposit_train2, metric=c("ACC","TPR","PRECISION","F1"))

# 2. C.	

# Observe the difference between the output in task 2A and task 2B. 
# Write a few sentences to 1) describe whether the prediction accuracy between 
# the train set and the test set in task 2B becomes larger or not, in contrast 
# to the comparison in task 2A; and 2) provide a possible explanation for such 
# difference (hint: think about the concept we have learned in Week 3).


# 3. A.	

# Train a naiveBayes model. Show information about this model. 

bank_full.nb <- naiveBayes(bank_fullTrain$deposit~.,bank_fullTrain)
bank_full.nb

summary(bank_full.nb)

# Generate and compare this model's confusion matrices and classification 
# evaluation metrics in test and train sets.

predicted_deposit_nb_test1 <- predict(bank_full.nb, bank_fullTest)

mmetric(bank_fullTest$deposit, predicted_deposit_nb_test1, metric="CONF")

mmetric(bank_fullTest$deposit, predicted_deposit_nb_test1, metric=c("ACC","TPR","PRECISION","F1"))

predicted_deposit_nb_train1 <- predict(bank_full.nb, bank_fullTrain)

mmetric(bank_fullTrain$deposit, predicted_deposit_nb_train1, metric="CONF")

mmetric(bank_fullTrain$deposit, predicted_deposit_nb_train1, metric=c("ACC","TPR","PRECISION","F1"))

# 4.

# Create named cross validation function - cv_function

# A.

# This function uses arguments for data frame, classification algorithm, seed value, 
# number of folds, and a set of classification metrics (DO NOT include confusion matrix output).

# B.

# The function should generate the mean values and standard deviations of each 
# performance metric over all folds as well.

# C.

# Use kable() to show the performance metrics by fold and their mean values and standard deviations.

cv_function <- function(df, target, nFolds, seedVal, classification, metrics_list)
{
  
  set.seed(seedVal)
  folds = createFolds(df[,target],nFolds)
  # folds
  
  cv_results <- lapply(folds, function(x)
  { 
    train <- df[-x,-target]
    test  <- df[x,-target]
    
    train_target <- df[-x,target]
    test_target <- df[x,target]
    
    classification_model <- classification(train,train_target) 
    
    pred<- predict(classification_model,test)
    
    return(mmetric(test_target,pred,metrics_list))
  })
  
  cv_results_m <- as.matrix(as.data.frame(cv_results))
  
  cv_mean<- as.matrix(rowMeans(cv_results_m))
  
  colnames(cv_mean) <- "Mean"
  
  cv_sd <- as.matrix(rowSds(cv_results_m))
  
  colnames(cv_sd) <- "Sd"
  
  cv_all <- cbind(cv_results_m, cv_mean, cv_sd)
  
  kable(cv_all,digits=2)
}

# 5. A. 

# Use cv_function to generate and compare 5-fold and 10-fold C5.0 and naiveBayes evaluation performance.
# Use the original bank_full data set to evaluate C5.0 and naiveBayes models by 5-fold as well as 10-fold
# cross-validation evaluations.

df <- bank_full
target <- 5
nFolds <- 5
seedVal <- 500
assign("classification", naiveBayes)
metrics_list <- c("ACC","PRECISION","TPR","F1")

cv_function(df, target, nFolds, seedVal, classification, metrics_list)

# Different nFolds

nFolds <- 10

cv_function(df, target, nFolds, seedVal, classification, metrics_list)

# Different classification algorithm

assign("classification", C5.0)

cv_function(df, target, nFolds, seedVal, classification, metrics_list)

# Different nFolds

nFolds <- 5

cv_function(df, target, nFolds, seedVal, classification, metrics_list)
