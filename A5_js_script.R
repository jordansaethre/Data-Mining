#Assignment 5 - KNN, Clustering and Association Rule Mining

#Packages required:  Install C50, psych, rweka, caret, rminer, matrixStats, knitr and arules packages.

# Understand retail_visits.csv using correlation analysis (pairs.panels from psych), decision trees (C5.0), 
# and clustering (SimpleKmeans in RWeka)

# 1. A.	

# Load packages and import retail_visits.csv. Show the overall structure of the input file. Transform factor 
# variables if necessary, and show the summary of the input data file. Use pairs.panels to exam variable 
# distributions and correlations. Do NOT include invoiceID and customerID in the analysis.

library(C50)
library(psych)
library(RWeka)
library(caret)
library(rminer)
library(matrixStats)
library(knitr)
library(arules)

setwd("C:/Users/jordan.saethre/Documents/IS 6482")

inputfile <- "C:/Users/jordan.saethre/Documents/IS 6482/retail_visits.csv"

r.visits <- read.csv(file = inputfile, stringsAsFactors = TRUE)

rvisits <- r.visits[,3:12]

summary(rvisits)

str(rvisits)

pairs.panels(rvisits[1:7])
pairs.panels(rvisits[2:8])
pairs.panels(rvisits[3:9])
pairs.panels(rvisits[4:10])

# 1. B.	

# Build a C5.0 decision tree using the default setting and the whole data set (retention is the target variable).
# Show summary of the model to see the tree and the in-sample confusion matrix.

rvisits.c50 <- C5.0(rvisits$retention~.,rvisits)

summary(rvisits.c50)

# Build and show clusters to better understand visits in clusters of similar visits according to the following 
# requirements.

# 1. C. i. 	

# Use SimpleKMeans for all tasks. Remove retention (i.e. the target variable) from input for building clusters. 
# Show the standard deviations in addition to the centroids of the clusters.

# 1. C. ii.	

# Generate and show 6 clusters using the default (i.e. random) initial cluster assignment and the default distance 
# function (Euclidean).

k <- 6

rvisits_clustering_6 <- SimpleKMeans(rvisits[-1], Weka_control(N=k, V=TRUE))

rvisits_clustering_6

# 1. C. iii.	

# Keep the number of clusters to be 6 and the distance function to be Euclidean. Change the initial cluster assignment
# method to the Kmeans++ method. Regenerate and show the results.

rvisits_clustering_6_kmeansmethod <- SimpleKMeans(rvisits[-1], Weka_control(N=k, init = 1, V=TRUE))

rvisits_clustering_6_kmeansmethod

# 1. C. iv.	

# Keep the number of clusters to be 6 and the initial cluster assignment method to be the Kmeans++ method. Change the 
# distance function to "weka.core.ManhattanDistance". Regenerate and show the results.

rvisits_clustering_6_kmeansmethod_manhattan <- SimpleKMeans(rvisits[-1], Weka_control(N=k, 
                                        A="weka.core.ManhattanDistance", init = 1, V=TRUE))

rvisits_clustering_6_kmeansmethod_manhattan

# 1. C. v.	

# Use the same distance function and initial assignment method selected for task C.iv of this chunk. Change the number 
# of clusters to 3.  Regenerate and show the results.

k <- 3

rvisits_clustering_3_kmeansmethod_manhattan <- SimpleKMeans(rvisits[-1], Weka_control(N=k, 
                                         A="weka.core.ManhattanDistance", init = 1, V=TRUE))

rvisits_clustering_3_kmeansmethod_manhattan

# KNN-based retention classification using IBk of RWeka

# 2. A.	

# Define two cross-validation functions - one with fixed K and I, the other with automatic K and I - so that you can 
# tune different IBk's parameters.

# Fixed K and I

cv_IBk <- function(df, target, nFolds, seedVal, metrics_list, k, i)
{
  # create folds using the assigned values
  
  set.seed(seedVal)
  folds = createFolds(df[,target],nFolds)
  
  # The lapply loop
  
  cv_results <- lapply(folds, function(x)
  { 
    # data preparation:
    
    test_target <- df[x,target]
    test_input <- df[x,-target]
    
    train_target <- df[-x,target]
    train_input <- df[-x,-target]
    pred_model <- IBk(train_target ~ .,data = train_input,control = Weka_control(K=k,I=i))  
    pred <- predict(pred_model, test_input)
    return(mmetric(test_target,pred,metrics_list))
  })
  
  cv_results_m <- as.matrix(as.data.frame(cv_results))
  cv_mean<- as.matrix(rowMeans(cv_results_m))
  cv_sd <- as.matrix(rowSds(cv_results_m))
  colnames(cv_mean) <- "Mean"
  colnames(cv_sd) <- "Sd"
  kable(t(cbind(cv_mean,cv_sd)),digits=3)
}

# Automatic K and I

cv_IBkX <- function(df, target, nFolds, seedVal, metrics_list, k, i)
{
  # create folds using the assigned values
  
  set.seed(seedVal)
  folds = createFolds(df[,target],nFolds)
  
  # The lapply loop
  
  cv_results <- lapply(folds, function(x)
  { 
    # data preparation:
    
    test_target <- df[x,target]
    test_input <- df[x,-target]
    
    train_target <- df[-x,target]
    train_input <- df[-x,-target]
    pred_model <- IBk(train_target ~ .,data = train_input,control = Weka_control(K=k,I=i,X=TRUE))  
    pred <- predict(pred_model, test_input)
    return(mmetric(test_target,pred,metrics_list))
  })
  
  cv_results_m <- as.matrix(as.data.frame(cv_results))
  cv_mean<- as.matrix(rowMeans(cv_results_m))
  cv_sd <- as.matrix(rowSds(cv_results_m))
  colnames(cv_mean) <- "Mean"
  colnames(cv_sd) <- "Sd"
  kable(t(cbind(cv_mean,cv_sd)),digits=3)
}


# 2. B.	

# Call one of the functions defined in A of this chunk with the default parameter setting of IBk to set a base line 
# out-of-sample performance of KNN-based retention classification. Set the number of folds to 5 or more.

df <- rvisits
target <- 1
seedVal <- 500
metrics_list <- c("ACC","TPR","PRECISION","F1")
nFolds <- 5

cv_IBk(df, target, nFolds, seedVal, metrics_list, 1, FALSE)

# Performance improvement based on the following requirements and suggestions:
  
# 2. C. i.-v.	

# Change IBk's parameters in the calls of functions defined in task A of this chunk to improve this classifier's 
# overall accuracy in cross validation.

# You can also selectively remove some predictors in this chunk in order to improve the effectiveness of selecting 
# nearest neighbors.

# Use the same number of folds as what you set for task B of this chunk.

# While your goal is to improve performance, you may also want to learn from the process the non-linear relationships 
# between parameter values, predictors and classification performance. Experiment with the different parameter values, 
# predictor selection as different combinations may give you similar performance improvements.

# Your final code for task C of this chunk should include only two calls to the function defined in A with two 
# different combinations of IBk parameter values and/or predictor selections that have given you higher classification 
# accuracies than that in B.

cv_IBk(df, target, nFolds, seedVal, metrics_list, 5, FALSE)

cv_IBk(df[-9], target, nFolds, seedVal, metrics_list, 5, FALSE)

# Read and mine retail_baskets.csv in the long file format.

# 3. A.	

# Import retail_baskets.csv file using the following read.transactions() function with the "single" format (for long 
# format) and save it in a sparse matrix called, e.g., Retail_baskets.

Retail_baskets = read.transactions ("retail_baskets.csv", format="single", sep = ",", cols = c("invoiceID","item"))

# 3. B.	

# Inspect the baskets in the first 5 transactions.

inspect(Retail_baskets[1:5])

# Use the itemFrequencyPlot command to perform the following tasks.

# 3. C. i.	

# View the frequency (in percentage, i.e., the relative format) of all of the item sets with support = 0.12 or higher.

itemFrequencyPlot(Retail_baskets, type="relative", support = 0.12)

# 3. C. ii.	

# Plot the most frequent 8 items in the descending order of transaction frequency in percentage.

itemFrequencyPlot(Retail_baskets, type="relative", topN = 8)

# 3. D.	

# Use the apriori command to generate about 50 to 100 association rules from the input data. Set your own minimum 
# support and confidence threshold levels. Remember if the thresholds are too low, you will get too many rules, or 
# if you set them too high, you may not get any or enough rules. Show the rules in the descending order of their lift 
# values.


retail_rules <- apriori(Retail_baskets, parameter = list(support = 0.05, confidence = 0.85, minlen = 2))

inspect(sort(retail_rules, by = "lift"))
