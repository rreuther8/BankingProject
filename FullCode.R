
# Creating training at test dataset
# http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

### Load Packages ----
if (!require(dplyr)) install.packages('dplyr')
library(dplyr)
if (!require(tidyr)) install.packages('tidyr')
library(tidyr)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(randomForest)) install.packages('randomForest')
library("randomForest")
if (!require(rpart)) install.packages('rpart')
library(rpart)
if (!require(rpart.plot)) install.packages('rpart.plot')
library(rpart.plot)
if (!require(e1071)) install.packages('e1071')
library(e1071)
if (!require(Amelia)) install.packages('Amelia')
library(Amelia)
if (!require(ggplot2)) install.packages('ggplot2')
library(ggplot2)
if (!require(GGally)) install.packages('GGally')
library(GGally)
if (!require(polycor)) install.packages('polycor')
library(polycor)
if (!require(corrplot)) install.packages('corrplot')
library(corrplot)


### Load Data ----
# Automatically downloads data from url
file_name <- "bank-full.csv"
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
# Information on dataset : http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
temp_file <-  tempfile()
download.file(url, temp_file)
raw_data <- read.csv2(unz(temp_file, file_name))

### Data Exploration - Part 1 --------------------------------------------------------------

# Review of the raw dataset.
str(raw_data)

# Tons of "unknown" values. Let's Change these to NA. 
# create dataset where "unknown" values are flipped to "unknown"
raw_data2 <- read.csv2(unz(temp_file, file_name), na = c("unknown"))
unlink(temp_file)

# ONLY FACTORS HOLD NA's
a <- raw_data2[!complete.cases(raw_data2),]

# Proportion of data was missing one or more records:
count(a) / count(raw_data2)

# Most of the missing data (NA values) are in the poutcome column
missmap(raw_data2)

# Upon exploring the column we're predicting, y, I noticed most y records were set to 0/no.
# This would indicate class imbalance
# prevalence of y==1 or y=="yes"
prop <- data.frame(Method = "Original", Probability = prop.table(table(raw_data2$y))[2])
prop
# There is a major class imbalance in this dataset. We will need to improve this.
# We have two options for cleaning up the data/fixing the class imbalance
# 1 - Imputing random categorical values based on each categorical predictor's probability (probability-based random assignment) 
# 2 - Removing all NA values.

### Data Cleaning --------------------------------------------------------------

# Probability-based Imputation
# 1) Get prevalence of each element for each predictor
# 2) Use sample() to apply name back to 

# Find columns that have NA's
colnames(raw_data)[colSums(is.na(raw_data2)) > 0]

# Use data without NA's to gather Medians/Modes for those columns
cleaned <- raw_data2[complete.cases(raw_data2),]
N <- count(cleaned)$n

#Gather the prevalence of each unique element for job, education, contact, and poutcome
# JOB
pctjobs <- prop.table(table(cleaned$job))
jobnames <- sort(as.character(unique(cleaned$job)))

# EDUCATION
pcteducation <- prop.table(table(cleaned$education))
educationnames <- sort(as.character(unique(cleaned$education)))

# CONTACT
pctcontact <- prop.table(table(cleaned$contact))
contactnames <- sort(as.character(unique(cleaned$contact)))

#POUTCOME
pctpoutcome <- prop.table(table(cleaned$poutcome))
poutcomenames <- sort(as.character(unique(cleaned$poutcome)))

cleaned_data <- raw_data2 %>% mutate(
  job = factor(ifelse(is.na(job), sample(jobnames, 1, replace = TRUE, prob = pctjobs), as.character(job))),
  education = factor(ifelse(is.na(education), sample(educationnames, 1, replace = TRUE, prob = pcteducation), as.character(education))),
  contact = factor(ifelse(is.na(contact), sample(contactnames, 1, replace = TRUE, prob = pctcontact), as.character(contact))),
  poutcome = factor(ifelse(is.na(poutcome), sample(poutcomenames, 1, replace = TRUE, prob = pctpoutcome), as.character(poutcome)))
)

# imputation method... cleaned_data => train and test
imp_cleaned <- cleaned_data %>% 
  mutate(y = factor(ifelse(y=="no",0,1)), poutcome = factor(ifelse(poutcome=="success",1,0)),campaign = as.integer(campaign))

# Add the prevalence of y == yes/1 to the prevalence table
prop <- bind_rows(prop,
                  data_frame(Method = "Imputation Method",
                             Probability = prop.table(table(imp_cleaned$y))[2]))
prop
## Removal 

# Create a dataset with all records not containing NA's
rem_cleaned <- raw_data2[complete.cases(raw_data2),]

prop <- bind_rows(prop,
                  data_frame(Method = "Removal Method",
                             Probability = prop.table(table(rem_cleaned$y))[2])
)

print(prop)

# Proportions for the raw dataset and the imputation method are the same.
#  This is because we we didn't remove any records or manipulat y values.
# The removal method has a higher proportion of y = 'yes' records. Since removing NA records increased
#  the 'yes' probability, we know that users that pay for the service are more likely to put all their information in.
#  This will be explored further in the data exploration section.


## Pre-Emptive Algo testing 
# IMPUTATION Algo Testing
imp_test_index <- createDataPartition(y = imp_cleaned$y , times = 1, p = 0.1, list = FALSE)
imp_train <- imp_cleaned[-imp_test_index,]  
imp_test <- imp_cleaned[imp_test_index,]  
imp_fit <- train(y~., data = imp_train, method = "glm")
imp_y_hat <- predict(imp_fit, imp_test)
imp_con <- confusionMatrix(imp_y_hat,imp_test$y, positive = "1")
comp <- data.frame(Type = "Imputation", 
                   Accuracy = imp_con$overall["Accuracy"], 
                   Sensitivity = imp_con$byClass["Sensitivity"], 
                   Specificity= imp_con$byClass["Specificity"]
)

# REMOVAL Algo Testing
rem_cleaned <- raw_data2[complete.cases(raw_data2),]
rem_test_index <- createDataPartition(y = rem_cleaned$y , times = 1, p = 0.1, list = FALSE)
rem_train <- rem_cleaned[-rem_test_index,]  
rem_test <- rem_cleaned[rem_test_index,]  
rem_fit <- train(y~., data = rem_train, method = "glm")
rem_y_hat <- predict(rem_fit, rem_test)
rem_con <- confusionMatrix(rem_y_hat, rem_test$y, positive = "1")
comp <- bind_rows(comp,
                  data.frame(Type = "Removal", 
                             Accuracy = rem_con$overall["Accuracy"], 
                             Sensitivity = rem_con$byClass["Sensitivity"], 
                             Specificity= rem_con$byClass["Specificity"])
)

print(comp)
# The removal method improved Specificity. 
# Given the class imbalance, working to improve Specificity is more important than a high accuracy and sensitivity. 
rem_con

# Final Data Cleaning 
# Using Removal method
set.seed(123)
data <- raw_data2[complete.cases(raw_data2),] %>% mutate(y= factor(ifelse(y == "yes",1,0)), poutcome = factor(poutcome)) 
test_index <- createDataPartition(y = data$y , times = 1, p = 0.1, list = FALSE)
algotraining <- data[-test_index,] 
validation <- data[test_index,]  

set.seed(123)
test_index <- createDataPartition(y = algotraining$y , times = 1, p = 0.1, list = FALSE)
train <- algotraining[-test_index,] 
test <- algotraining[test_index,]  


### Data Exploration - Part 2 ----

# All Correlations with y are as follows:
h <- hetcor(data)
col <- colnames(h$correlations[])
ycorr <- data.frame("y-Correlation" = h$correlations[,"y"])
ycorr[order(ycorr$y.Correlation), , drop = FALSE]


# Train the algo using RPART to develop a decision tree
# Helps to visualize which users traditionally sign on as subscribers
rpart_fit1 <- rpart(y~., data = train, method = "class")
rpart_y_hat1 <- predict(rpart_fit1, test, type = "class")
rpart.plot(rpart_fit1,  main="RPART Decision Tree (cp = 0.01)", extra=100)

# Review the importance of the different predictors, as this helps interpret which predictors 
varImp(rpart_fit1)

#Housing
data %>% mutate(housing = ifelse(housing=="yes",1,0)) %>%group_by(y) %>% 
  summarize(h = mean(housing), sd = sd(housing), n = n()) %>% 
  ggplot(aes(y, h, fill=y,color = y)) + geom_bar(stat = "identity") + 
  ggtitle("Housing status vs subscription status")


# last month that user was campaigned and percentage of subscribed user
data %>% group_by(month) %>% summarize(pct = mean(y==1)) %>% mutate(month = reorder(month,desc(pct))) %>%
  ggplot(aes(month,pct)) + 
  geom_bar(position = "dodge2", stat = "identity") + 
  ggtitle("Last month of campaign vs percentage of subscribed users")


# how often, and at what cadence, are users contacted?
# subscribers are contacted every 100 and 200 days
# non-subscribers are contacted often at the beginning, then as it hits the 100 mark the amoun goes to
# indicates that users are usually contacted in cycles
data %>% group_by(pdays,y) %>% summarize(cnt = n()) %>% 
  ggplot(aes(x = pdays, y = cnt, group = y, color = y)) +
  geom_line() +  scale_x_continuous(limits = c(0, 500)) + scale_y_continuous(limits = c(0,100))



### Training ----

## RPART ##
rpart_fit <- rpart(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing,  data = train, method = "class")
rpart_y_hat <- predict(rpart_fit, test, type = "class")
rpart_con <- confusionMatrix(rpart_y_hat, test$y, positive = "1")

results <- data.frame(Type = "RPART", 
                      Accuracy = rpart_con$overall["Accuracy"], 
                      Sensitivity = rpart_con$byClass["Sensitivity"], 
                      Specificity= rpart_con$byClass["Specificity"]
)

## Binary Logistic Regression  ##
# The variable to predict is a is a 0 or 1, so it is binary.
# We need to specify type = "prob" for conditional probabilities when using "predict" with glm objects
glm_fit <- train(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing,  data = train, method = "glm", family="binomial")
glm_p_hat <- predict(glm_fit, test, type = "prob")
glm_y_hat <- factor(ifelse(glm_p_hat[1] > glm_p_hat[2], 0,1))
glm_con <- confusionMatrix(glm_y_hat, test$y, positive = "1")

results <- bind_rows(results,
                     data.frame(Type = "Logistic Regression", 
                                Accuracy = glm_con$overall["Accuracy"], 
                                Sensitivity = glm_con$byClass["Sensitivity"], 
                                Specificity= glm_con$byClass["Specificity"]
                     )
)

## Naive Bayes ##
# The Naive Bayes model assumes we can estimate conditional distributions of the predictors. 
# Because there are 16 features, so we will not be using Naive Bayes model in the final algorithm.
nb_fit <- naiveBayes(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing, data = train)
nb_y_hat <- predict(nb_fit, test)
nb_con <- confusionMatrix(nb_y_hat, test$y, positive = "1")


## Quantitative Discriminant Analysis ##
# Form of Naive Bayes that assumes distribution of variable we're predicting is bivariate normal.
# Since we are training on 16 predictors, the assumption of normality will most likely not hold for 
# all predictors. With more predictors, we are susceptible to overfitting.

qda_fit <- train(yy~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing, data = train, method = "qda")
qda_y_hat <- predict(qda_fit, test)
qda_con <- confusionMatrix(qda_y_hat,test$y, positive = "1")
results <- bind_rows(results,
                     data.frame(Type = "Quantitative Discriminant Analysis", 
                                Accuracy = qda_con$overall["Accuracy"], 
                                Sensitivity = qda_con$byClass["Sensitivity"], 
                                Specificity= qda_con$byClass["Specificity"]
                     )
)

## Linear Discriminative Analysis ##
# To account for several predictors, we can use Linead Discriminant analysis, which assumes
# the correlation structure is the same for all classes, which reduces the number of parameters 
# we need to estimate

lda_fit <- train(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing, data = train, method = "lda")
lda_y_hat <- predict(lda_fit, test)
lda_con <- confusionMatrix(lda_y_hat,test$y, positive = "1")
results <- bind_rows(results,
                     data.frame(Type = "Linear Discriminant Analysis", 
                                Accuracy = lda_con$overall["Accuracy"], 
                                Sensitivity = lda_con$byClass["Sensitivity"], 
                                Specificity= lda_con$byClass["Specificity"]
                     )
)

## KNN ##
knn_fit <- train(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing, data = train, method = "knn")
# final KNN algo uses 9 nearest-neighbors
knn_y_hat <- predict(knn_fit, test)
knn_con <- confusionMatrix(knn_y_hat,test$y, positive = "1")
results <- bind_rows(results,
                     data.frame(Type = "K-Nearest Neighbors", 
                                Accuracy = knn_con$overall["Accuracy"], 
                                Sensitivity = knn_con$byClass["Sensitivity"], 
                                Specificity= knn_con$byClass["Specificity"]
                     )
)

## Random Forest ##
# How many Trees is optimal?
rf_fit <- randomForest(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing,  data = train, proximity = TRUE)
rf_fit.df <- data.frame(
  Tree = rep(1:nrow(rf_fit$err.rate), times = 3),
  Type = rep(c("OOB","0/NO", "1/YES"),each=nrow(rf_fit$err.rate)),
  Error = c(rf_fit$err.rate[,"OOB"],
            rf_fit$err.rate[,"0"],
            rf_fit$err.rate[,"1"]
  )
)
rf_fit.df %>% ggplot(aes(Tree,Error)) + geom_line(aes(color = Type))
# The error rates level off around 100 trees, so this will be the number of trees created in the model.

# How many nodes are optimal?
oob <- vector(length = 10)
for(i in 1:10){
  rf <- randomForest(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing, data = train, proximity = TRUE, ntree = 100, mtry = i)
  oob[i] <- rf$err.rate[nrow(rf$err.rate),1]
}
oob[which.min(oob)]
node.min <-which.min(oob)

rf_fit <- randomForest(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing, data = train, proximity = TRUE, ntree = 100, mtry = node.min)
rf_y_hat <- predict(rf_fit, test)
rf_con <- confusionMatrix(rf_y_hat,test$y, positive = "1")

results <- bind_rows(results,
                     data.frame(Type = "Random Forest", 
                                Accuracy = rf_con$overall["Accuracy"], 
                                Sensitivity = rf_con$byClass["Sensitivity"], 
                                Specificity= rf_con$byClass["Specificity"]
                     )
)

## Ensemble ##
df <- data.frame(
  glm = ifelse(as.numeric(glm_y_hat)==1,0,1),
  knn = ifelse(as.numeric(knn_y_hat)==1,0,1),
  lda = ifelse(as.numeric(lda_y_hat)==1,0,1), 
  qda = ifelse(as.numeric(qda_y_hat)==1,0,1), 
  rPart = ifelse(as.numeric(rpart_y_hat)==1,0,1),
  rF = ifelse(as.numeric(rf_y_hat)==1,0,1)
)

p <- mean(test$y == 1)
votes <- rowSums(df)


ensemble_y_hat <- ifelse(votes >= 3,1,0)
ens_con1 <- confusionMatrix(factor(ensemble_y_hat), test$y, positive = "1")
results <- bind_rows(results,
                     data.frame(Type = "Ensemble v1", 
                                Accuracy = ens_con1$overall["Accuracy"], 
                                Sensitivity = ens_con1$byClass["Sensitivity"], 
                                Specificity= ens_con1$byClass["Specificity"]
                     )
)

results %>% arrange(Specificity)

ensemble_loop <- data.frame(MissingModel = character(),  
                            Accuracy = numeric(),
                            Sensitivity = numeric(),
                            Specificity= numeric()
)

for (i in 1:length(df)){
  loop.df <- df[,-i]
  loop.votes <- rowSums(loop.df)
  loop.y <- ifelse(loop.votes >= 3,1,0)
  loop.conf <- confusionMatrix(factor(loop.y),test$y, positive = "1")
  ensemble_loop <- bind_rows(ensemble_loop,
                             data.frame(MissingModel = colnames(df[i]),  
                                        Accuracy = loop.conf$overall["Accuracy"], 
                                        Sensitivity = loop.conf$byClass["Sensitivity"], 
                                        Specificity= loop.conf$byClass["Specificity"]
                             ))
}
ensemble_loop

max_ensemble <- ensemble_loop[which.max(ensemble_loop$Specificity),]

results <- bind_rows(results,
                     data.frame(Type = "Ensemble v2 (remove worst performer)", 
                                Accuracy = max_ensemble$Accuracy, 
                                Sensitivity = max_ensemble$Sensitivity,
                                Specificity= max_ensemble$Specificity
                     )
)

results

# one final one... top three from DF...
# Top three are as LDA (3), QDA (4), RandomForest (6)
df3 <- df[c(3,4,6)]
votes3 <- rowSums(df3)
ensemble_y_hat3 <- ifelse(votes3 >= 2,1,0)
ens_con3 <- confusionMatrix(factor(ensemble_y_hat3), test$y, positive = "1")


results <- bind_rows(results,
                     data.frame(Type = "Ensemble v3 (Top 3 Performers)", 
                                Accuracy = ens_con3$overall["Accuracy"], 
                                Sensitivity = ens_con3$byClass["Sensitivity"], 
                                Specificity= ens_con3$byClass["Specificity"]
                     )
)
results

# After reviewing the logic, it appears QDA provided the best specificity, however the lowest accuracy, 
# so this might not be the best option. Additionally, the 1st and 3rd version of the ensemble method and the 
# Random forest method were very good. Considering the ensemble moethods requre 5 or more methods to train, 
# this will take a good deal of time to complete.

# In order to baalnce computer performance and algorithm performance, the Random Forest algorithm alone 
# would be best in efficiently predicting potential subscribers.
#Final Model

fit <- randomForest(y~poutcome+month+job+pdays+balance+day+education+age+marital+campaign+housing, data = train, proximity = TRUE, ntree = 100, mtry = node.min)
final_y_hat <- predict(fit, validation)
confusionMatrix(final_y_hat, validation$y, positive = "1")

