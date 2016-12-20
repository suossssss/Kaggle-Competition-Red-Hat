
library(data.table)
library(FeatureHashing)
library(Matrix)
library(xgboost)


### 1) Basic data preparation via data.table

##Before merging the people and train tables we'll convert the T/F people features to 1/0 so that they can be treated as a dummy variable directly later on. The 'set' function is the easy way to do this by reference in data.table. The outcome values are separated from the training features for convenience.


people <- fread("people.csv", showProgress = F)

##create a variable-temp to help clean the original data
temp<-c("char_10","char_11","char_12","char_13","char_14","char_15",
        "char_16","char_17","char_18","char_19","char_20","char_21",
        "char_22","char_23","char_24","char_25","char_26","char_27",
        "char_28","char_29","char_30","char_31","char_32","char_33",
        "char_34","char_35","char_36","char_37")

##change "TRUE"&"FALSE" values into 1&0 ,respectively 
for (char in temp)
  people[,char:=as.numeric(as.logical(get(char))),with=FALSE]


train  <- fread("act_train.csv", showProgress = F)
test<-fread('act_test.csv', showProgress = F)

#reducing group_1 dimension
people$group_1[people$group_1 %in% names(which(table(people$group_1)==1))]='group unique'

d1 <- merge(train, people, by = "people_id", all.x = T)

Y <- d1$outcome
d1[ , outcome := NULL]



### 2) Process categorical features via FeatureHashing

##The FeatureHashing package is a nice quick way to encode the categorical features into a sparse matrix. Numeric features will automatically be included in the matrix as they are. For now, we'll exclude the date features (along with the id features). The performance of the model doesn't appear to improve much above a hash size of 2 ^ 22.


b <- 2 ^ 22
f <- ~ . - people_id - activity_id - date.x - date.y - 1

X_train <- hashed.model.matrix(f, d1, hash.size = b)


##We can easily check how many columns of the sparse matrix are occupied by at least one row of the training data (this one line is the only reason to load the Matrix library).


sum(colSums(X_train) > 0)



### 3) Validate xgboost model

##The linear mode of xgboost provides a nice baseline before we try anything more time consuming like boosted trees. The validation set is chosen by people_id. Perhaps surprisingly, adding regularisation via alpha or lambda doesn't seem to help.

set.seed(75786)
unique_p <- unique(d1$people_id)
valid_p  <- unique_p[sample(1:length(unique_p), 30000)]

valid <- which(d1$people_id %in% valid_p)
model <- (1:length(d1$people_id))[-valid]

param <- list(objective = "binary:logistic", 
              eval_metric = "auc",
              booster = "gblinear", 
              eta = 0.03)

dmodel  <- xgb.DMatrix(X_train[model, ], label = Y[model])
dvalid  <- xgb.DMatrix(X_train[valid, ], label = Y[valid])

m1 <- xgb.train(data = dmodel, param, nrounds = 100,
                watchlist = list(model = dmodel, valid = dvalid),
                print_every_n = 10)




### 4) Retrain on all data and predict for test set

#We'll watch the training error just to check nothing has gone awry. Then another advantage of hashing: we can process the test data independent of the training data.


dtrain  <- xgb.DMatrix(X_train, label = Y)

m2 <- xgb.train(data = dtrain, param, nrounds = 100,
                watchlist = list(train = dtrain),
                print_every_n = 10)


d2   <- merge(test, people, by = "people_id", all.x = T)

X_test <- hashed.model.matrix(f, d2, hash.size = b)
dtest  <- xgb.DMatrix(X_test)

out <- predict(m2, dtest)
sub <- data.frame(activity_id = d2$activity_id, outcome = out)
write.csv(sub, file = "sub.csv", row.names = F)
summary(sub$outcome)




