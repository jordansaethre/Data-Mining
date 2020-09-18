# Assignment 3 Bike Rental Prediction

# 1. A. 

# Load packages and import data. Show the overall structure and summary of the input data.  
# All non-numeric fields should be factor variables.

library(caret)
library(knitr)
library(matrixStats)
library(rmarkdown)
library(rminer)
library(rpart)
library(psych)
library(RWeka)

setwd("C:/Users/jordan.saethre/Documents/IS 6482")

inputfile <- "C:/Users/jordan.saethre/Documents/IS 6482/bike_share.csv"

bike <- read.csv(file = inputfile, stringsAsFactors = TRUE)

str(bike)

summary(bike)

# 1. B. 

# Include commands to explore the distributions of the numeric variables 
# (i.e. hour, temp, humidity, and windspeed) and their correlations.

hist(bike$hour)

hist(bike$temp)

hist(bike$humidity)

hist(bike$windspeed)

# correlations

# exploring relationships among features: correlation matrix

cor(bike[c("hour", "temp", "humidity", "windspeed")])

# visualizing correlations

pairs.panels(bike[c("hour", "temp", "humidity", "windspeed")])

# 1. C. 

# Include commands to explore the relationship between factor variables and the target variable 
# (i.e. season, year2012, month, holiday, weekday, and weathersit). Although some of them are 
# treated as binary variables, they have the same meaning to categorical variables in regression 
# models. (Hint, use boxplot function). Do NOT take the first two variables (i.e. instance and date) 
# into account.

boxplot(ttlrent~season, data = bike)

boxplot(ttlrent~year2012, data = bike)

boxplot(ttlrent~month, data = bike)

boxplot(ttlrent~holiday, data = bike)

boxplot(ttlrent~weekday, data = bike)

boxplot(ttlrent~weathersit, data = bike)

# 1. D.	

# Think about how 1.B and 1.C help you assess the potential strong predictors and potentially 
# helpful data transformations, and anticipate the potential model fit performance and the necessary 
# improvements.

# 1. E.	

# Use the whole data set WITHOUT instance and date, build a linear regression model. Show the summary 
# of the model to understand the significance and coefficients of the predictors in the model and the 
# overall model fit.

bike_base_model <- lm(bike[,13]~., data = bike[,3:13])

bike_base_model

summary(bike_base_model)

# 1. F.	

# Think about the evidences for why your thoughts for task 1.D are confirmed or disconfirmed.

# 1. G.	

# Partition the data set for simple hold-out evaluation - 70% for training and the remaining 30% for testing.

set.seed(500)
inTrain <- createDataPartition(y=bike$ttlrent, p = 0.70, list=FALSE)
train_target <- bike[inTrain, 13] 
test_target <- bike[-inTrain, 13]
train_input <- bike[inTrain,3:12]
test_input <- bike[-inTrain,3:12]

# 1. H.

# Show the overall summary of train sets and test sets.

summary(train_target)
summary(test_target)
summary(train_input)
summary(test_input)

# 2. A.	

# Train three models using lm, rpart and M5P. Use the default settings of these methods throughout this 
# assignment.

# lm

bike_base_train_model <- lm(train_target~ ., data = train_input)

bike_base_train_model

predictions_base_test <- predict(bike_base_train_model, test_input)

summary(predictions_base_test)

summary(test_target)

predictions_base_train <- predict(bike_base_train_model, train_input)

summary(predictions_base_train)

summary(train_target)

#rpart

bike_rpart_model <- rpart(train_target ~ ., data = train_input)

bike_rpart_model

predictions_rpart_test <- predict(bike_rpart_model, test_input)

summary(predictions_rpart_test)

summary(test_target)

predictions_rpart_train <- predict(bike_rpart_model, train_input)

summary(predictions_rpart_train)

summary(train_target)

#m5p

bike_m5p_model <- M5P(train_target ~ ., data = train_input)

bike_m5p_model

predictions_m5p_test <- predict(bike_m5p_model, test_input)

summary(predictions_m5p_test)

summary(test_target)

predictions_m5p_train <- predict(bike_m5p_model, train_input)

summary(predictions_m5p_train)

summary(train_target)

# 2. B.	

# Generate and this model's explanatory evaluation metrics and predictive error metrics (as in Week 4 
# tutorials) in both the test sets and train sets.

#lm

mmetric(test_target,predictions_base_test,c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2")) #why Null?
mmetric(train_target,predictions_base_train,c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2"))

#rpart

mmetric(test_target,predictions_rpart_test,c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2"))
mmetric(train_target,predictions_rpart_train,c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2"))

#m5p

mmetric(test_target,predictions_m5p_test,c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2"))
mmetric(train_target,predictions_m5p_train,c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2"))

# 2. C.	

# In the sets of performance metrics from 2.B, does higher R2 guarantee lower MAE? Does lower MAE guarantee
# lower RMSE? What is the meaning of MAPE and a potential reason for a high MAPE? What's the meaning of RAE 
# and a potential reason for high RAE? Which model is more generalizable in terms of model fit? Which is 
# more generalizable based on predictive errors?

# 3. A.	

# Define a named function for cross-validation of numeric prediction models that generates a table of the 
# model fit and error metrics used in Week 4 tutorials for each fold along with the means and standard 
# deviations of the metrics over all folds.

cv_function <- function(df, target, nFolds, seedVal, prediction_method, metrics_list)
{
  # create folds
  set.seed(seedVal)
  folds = createFolds(df[,target],nFolds) 
  # perform cross validation
  cv_results <- lapply(folds, function(x)
  { 
    test_target <- df[x,target]
    test_input  <- df[x,-target]
    
    train_target <- df[-x,target]
    train_input <- df[-x,-target]
    
    prediction_model <- prediction_method(train_target~.,train_input) 
    pred<- predict(prediction_model,test_input)
    return(mmetric(test_target,pred,metrics_list))
  })
  # generate means and sds and show cv results, means and sds using kable
  cv_results_m <- as.matrix(as.data.frame(cv_results))
  cv_mean<- as.matrix(rowMeans(cv_results_m))
  cv_sd <- as.matrix(rowSds(cv_results_m))
  colnames(cv_mean) <- "Mean"
  colnames(cv_sd) <- "Sd"
  cv_all <- cbind(cv_results_m, cv_mean, cv_sd)
  kable(t(cv_all),digits=2)
}

# 3. B.	

# Call the function in 3.A to generate 10-fold cross validation results of the simple lm, 
# rpart and M5P models.

df <- bike[,3:13]
target <- 11
nFolds <- 10
seedVal <- 500
assign("prediction_method", lm)
metrics_list <- c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2")

cv_function(df, target, nFolds, seedVal, prediction_method, metrics_list)

assign("prediction_method", rpart)

cv_function(df, target, nFolds, seedVal, prediction_method, metrics_list)

assign("prediction_method", M5P)

cv_function(df, target, nFolds, seedVal, prediction_method, metrics_list)

# 3. C.	

# Are the performance results from 2.B and 3.B similar or not? What do the comparisons suggest? 
# Are you concerned with applying some or all of the models to predict the target variable when 
# new data, that has similar patterns, is available? Why or why not?

# 4. A.	

# Create and add the quadratic term of hour, e.g., hour_squared, to the predictors for the target variable.

bike$hour_squared <- bike$hour^2

# 4. B.	

# Build a lm model with hour_squared included. Show the summary of this lm model. Write a line to explain 
# the meaning of the relationship between hour and ttlrent.

bike_base_model2 <- lm(bike[,13]~., data = bike[,3:14])

bike_base_model2

summary(bike_base_model2)

# 4. C.	

# Has the model fit improved over that from 1.D? What are the changes in the coefficients and significance 
# of predictors compared to those of the model in 1.D? Are these changes meaningful and actionable? Would 
# you suggest some other quadratic terms, interaction terms or predictor transformations that might further 
# improve the model fit?

# 4. D.	

# Call the cross-validation function defined for 3.A, to generate 10-fold cross validation results of the 
# simple lm, rpart and M5P models with hour_squared included. (Note: Be careful when you defined the target 
# variable. Do not include instance and date in the model.)

df <- bike[,3:14]
target <- 11
nFolds <- 10
seedVal <- 500
assign("prediction_method", lm)
metrics_list <- c("MAE","RMSE","MAPE","RMSPE","RAE", "RRSE", "COR", "R2")

cv_function(df, target, nFolds, seedVal, prediction_method, metrics_list)

assign("prediction_method", rpart)

cv_function(df, target, nFolds, seedVal, prediction_method, metrics_list)

assign("prediction_method", M5P)

cv_function(df, target, nFolds, seedVal, prediction_method, metrics_list)

# 4. E.	

# Are the performance results from 4.B and 4.D similar or not? What do the comparisons suggest?  Are you 
# concerned with applying some or all of the models in 4.D to predict the target variable when new data, 
# that has similar patterns, is available?  Why or why not?


