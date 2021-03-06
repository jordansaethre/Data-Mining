library(psych)
library(scatterplot3d)
library(rmarkdown)

# Import csv and xlsx files

setwd("C:/Users/jordan.saethre/Documents/IS 6482")

inputfile1 <- "C:/Users/jordan.saethre/Documents/IS 6482/bank_full.csv"

bank_full <- read.csv(file = inputfile1, stringsAsFactors = FALSE)

# Structure of Bank Data

str(bank_full)

# Summary of Bank Data

summary(bank_full)

#Transform String Variables into Factor Variables

bank_full <- read.csv(file = inputfile, stringsAsFactors = TRUE)

# Structure of Bank Data

str(bank_full)

# Summary of Bank Data

summary(bank_full)

# Number of Rows in bank_full

row <- nrow(bank_full)
row

# Number of Columns in bank_full

col <- ncol(bank_full)
col

# Show the first 10 records of bank_full

head(bank_full, n=10)

# Show the last 10 records of bank_full

tail(bank_full, n = 10)

# Histograms and Boxplots of Numeric Variables: age, duration, campaign, and pdays

# age

hist(bank_full$age, main="Histogram of Age", ylab="Age")

boxplot(bank_full$age, main="Boxplot of Age", ylab="Age")

# duration

hist(bank_full$duration, main="Histogram of Duration", ylab="Duration")

boxplot(bank_full$duration, main="Boxplot of Duration", ylab="Duration")

# Campaign 

hist(bank_full$campaign, main="Histogram of Campaign", ylab="Campaign")

boxplot(bank_full$campaign, main="Boxplot of Campaign", ylab="Campaign")

# pdays

hist(bank_full$pdays, main="Histogram of Paydays", ylab="Paydays")

boxplot(bank_full$pdays, main="Boxplot of Paydays", ylab="Paydays")

# Statistics of Interest

## duration

# mean

mean(bank_full$duration)

# variance

var(bank_full$duration)

# standard deviation

sd(bank_full$duration)

# quantiles

quantile(bank_full$duration)

# dectiles

quantile(bank_full$duration, seq(from = 0, to = 1, by = 0.10))

## campaign

# mean

mean(bank_full$campaign)

# variance

var(bank_full$campaign)

# standard deviation

sd(bank_full$campaign)

# quantiles

quantile(bank_full$campaign)

# dectiles

quantile(bank_full$campaign, seq(from = 0, to = 1, by = 0.10))

## pdays

# mean

mean(bank_full$pdays)

# variance

var(bank_full$pdays)

# standard deviation

sd(bank_full$pdays)

# quantiles

quantile(bank_full$pdays)

# dectiles

quantile(bank_full$pdays, seq(from = 0, to = 1, by = 0.10))

### Min-Max Normalization

# calculate the difference of the range in duration

max.duration <- max(bank_full$duration)
min.duration <- min(bank_full$duration)

max.campaign <- max(bank_full$campaign)
min.campaign <- min(bank_full$campaign)

max.pdays <- max(bank_full$pdays)
min.pdays <- min(bank_full$pdays)

range.diff.duration <- max.duration - min.duration
range.diff.campaign <- max.campaign - min.campaign
range.diff.pdays <- max.pdays - min.pdays

# min-max normalization of first observation's duration to a value between zero zna 1

bank_full.duration.n <- (bank_full$duration[] - min.duration)/range.diff.duration
bank_full.campaign.n <- (bank_full$campaign[] - min.campaign)/range.diff.campaign
bank_full.pdays.n <- (bank_full$pdays[] - min.pdays)/range.diff.pdays

## Statistics of Interest after Normalization

## duration

# mean

mean(bank_full.duration.n)

# variance

var(bank_full.duration.n)

# standard deviation

sd(bank_full.duration.n)

# quantiles

quantile(bank_full.duration.n)

# dectiles

quantile(bank_full.duration.n, seq(from = 0, to = 1, by = 0.10))

## campaign

# mean

mean(bank_full.campaign.n)

# variance

var(bank_full.campaign.n)

# standard deviation

sd(bank_full.campaign.n)

# quantiles

quantile(bank_full.campaign.n)

# dectiles

quantile(bank_full.campaign.n, seq(from = 0, to = 1, by = 0.10))

## pdays

# mean

mean(bank_full.pdays.n)

# variance

var(bank_full.pdays.n)

# standard deviation

sd(bank_full.pdays.n)

# quantiles

quantile(bank_full.pdays.n)

# dectiles

quantile(bank_full.pdays.n, seq(from = 0, to = 1, by = 0.10))


### Exploration of Factor Variables


# Check to see if these are factor variables

is.factor(bank_full$job)
is.factor(bank_full$education)
is.factor(bank_full$contact)
is.factor(bank_full$poutcome)

# Show count of each factor variable

summary(bank_full$job)
summary(bank_full$education)
summary(bank_full$contact)
summary(bank_full$poutcome)

# Show percentage value of each 

job.table <- prop.table(table(bank_full$job))
education.table <- prop.table(table(bank_full$education))
contact.table <- prop.table(table(bank_full$contact))
poutcome.table <- prop.table(table(bank_full$poutcome))

job.table
education.table
contact.table
poutcome.table

# Barplots of job and education

barplot(sort(table(bank_full$job), decreasing = TRUE), main = "Counts of Job types in Bank Data Set", xlab = "Jobs")

barplot(sort(table(bank_full$education), decreasing = TRUE), main = "Counts of Education Types in Bank Data Set", xlab = "Education Types")

# Number of Levels in contact and poutcome

contact_levels <-nlevels(bank_full$contact)
contact_levels

poutcome_levels <- nlevels(bank_full$poutcome)
poutcome_levels

# Demonstation of relationships between multiple variables

cor(bank_full[c("age", "duration", "campaign", "pdays")])

pairs.panels(bank_full[c("age", "duration", "campaign", "pdays")])

# Boxplots of age by deposit

boxplot(age~deposit, data = bank_full)

# Boxplots of age by housing

boxplot(age~housing, data = bank_full)

# Aggregate age by deposit

aggregate(age~deposit, summary, data = bank_full)

# Aggregate age by housing

aggregate(age~housing, summary, data = bank_full)

# Aggregate campaign by deposit

aggregate(campaign~deposit, summary, data = bank_full)

# Aggregate campaign by housing

aggregate(campaign~housing, summary, data = bank_full)

# 3D Scatterplot of deposit values
# Circle for "no" and triangle for "yes"

scatterplot3d(bank_full$age, bank_full$campaign, bank_full$duration, pch = as.numeric(bank_full$deposit), 
              xlab = "Age", ylab = "Campaign", zlab = "Duration", main = "3D scatter plot of Bank Data")

legend('topright', legend = levels(bank_full$deposit),  cex = 0.8, pch = 1:2)
