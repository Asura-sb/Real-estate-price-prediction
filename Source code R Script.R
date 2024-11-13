Install.packages("tidyverse")
Install.packages("caret")
Install.packages("randomForest")
Install.packages("xgboost")
Install.packages("glmnet")
Install.packages("ggplot2")

library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(ggplot2)

#loding DataSets
dataset1 <- read_csv("C:\\Users\\Owner\\Desktop\\dpa Project\\Datasets\\Real estate.csv")
dataset2 <- read_csv("C:\\Users\\Owner\\Desktop\\dpa Project\\Datasets\\data.csv")

#renaming columns
colnames(dataset1) <- c("No", "transaction_date", "house_age", "distance_to_mrt", "num_convenience_stores", "latitude", "longitude", "price")
colnames(dataset2) <- c("date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "street", "city", "statezip", "country")


#data Preproccessing
dataset1 <- dataset1 %>% 
  na.omit() %>% 
  select(-No)

dataset2 <- dataset2 %>% 
  na.omit() %>% 
  select(-c(date,street, city, statezip, country))
  


#training and spilting data

set.seed(42)
splitIndex1 <- createDataPartition(dataset1$price, p = 0.8, list = FALSE)
train1 <- dataset1[splitIndex1, ]
test1 <- dataset1[-splitIndex1, ]

set.seed(42)
splitIndex2 <- createDataPartition(dataset2$price, p = 0.8, list = FALSE)
train2 <- dataset2[splitIndex2, ]
test2 <- dataset2[-splitIndex2, ]

# Create data matrix and response vector for Lasso Regression
x_train2 <- as.matrix(train2[, c("bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated")])
y_train2 <- train2$price

# Perform cross-validation to find optimal lambda value
set.seed(123)
cv_results <- cv.glmnet(x_train2, y_train2, alpha = 1, nfolds = 5)
lambda_opt <- cv_results$lambda.min

# Train models on dataset1
model_rf1 <- randomForest(price ~ ., data = train1)
model_xgb1 <- xgboost(data = as.matrix(train1[, -ncol(train1)]), label = train1$price, nrounds = 100, objective = "reg:squarederror")
model_lasso1 <- cv.glmnet(x = as.matrix(train1[, -ncol(train1)]), y = train1$price, alpha = 1)

# Train models on dataset2
model_rf2 <- randomForest(price ~ ., data = train2)
model_xgb2 <- xgboost(data = as.matrix(train2[, -ncol(train2)]), label = train2$price, nrounds = 100, objective = "reg:squarederror")
model_lasso2 <- glmnet(x_train2, y_train2, alpha = 1, lambda = lambda_opt)


# Feature importance for dataset1
importance_rf1 <- importance(model_rf1)
importance_xgb1 <- xgb.importance(colnames(train1[, -ncol(train1)]), model = model_xgb1)

# Feature importance for Lasso model on dataset1
lasso_coef1 <- as.matrix(coef(model_lasso1, s = model_lasso1$lambda.min))
importance_lasso1 <- data.frame(Feature = rownames(lasso_coef1), Importance = abs(lasso_coef1))

# Feature importance for Random Forest and XGBoost on dataset2
importance_rf2 <- importance(model_rf2)
importance_xgb2 <- xgb.importance(colnames(train2[, -ncol(train2)]), model = model_xgb2) # Exclude 'price'


# Feature importance for Lasso model on dataset2
lasso_coef2 <- as.matrix(coef(model_lasso2, s = model_lasso2$lambda.min))
importance_lasso2 <- data.frame(Feature = rownames(lasso_coef2), Importance = abs(lasso_coef2))


# Predictions for dataset1
pred_rf1 <- predict(model_rf1, test1)
pred_xgb1 <- predict(model_xgb1, as.matrix(test1[, -ncol(test1)]))
pred_lasso1 <- predict(model_lasso1, newx = as.matrix(test1[, -ncol(test1)]), s = model_lasso1$lambda.min)

# Predictions for dataset2
pred_rf2 <- predict(model_rf2, test2)
pred_xgb2 <- predict(model_xgb2, as.matrix(test2[, -ncol(test2)]))
pred_lasso2 <- predict(model_lasso2, newx = as.matrix(test2[, -ncol(test2)]), s = model_lasso2$lambda.min)

# Compute performance metrics
performance1 <- data.frame(
  Model = c("Random Forest", "XGBoost", "Lasso Regression"),
  RMSE1 = c(sqrt(mean((test1$price - pred_rf1)^2)), sqrt(mean((test1$price - pred_xgb1)^2)), sqrt(mean((test1$price - pred_lasso1)^2))),
  MAE1 = c(mean(abs(test1$price - pred_rf1)), mean(abs(test1$price - pred_xgb1)), mean(abs(test1$price - pred_lasso1))),
  R21 = c(cor(test1$price, pred_rf1)^2, cor(test1$price, pred_xgb1)^2, cor(test1$price, pred_lasso1)^2)
)

performance2 <- data.frame(
  Model = c("Random Forest", "XGBoost", "Lasso Regression"),
  RMSE2 = c(sqrt(mean((test2$price - pred_rf2)^2)), sqrt(mean((test2$price - pred_xgb2)^2)), sqrt(mean((test2$price - pred_lasso2)^2))),
  MAE2 = c(mean(abs(test2$price - pred_rf2)), mean(abs(test2$price - pred_xgb2)), mean(abs(test2$price - pred_lasso2))),
  R22 = c(cor(test2$price, pred_rf2)^2, cor(test2$price, pred_xgb2)^2, cor(test2$price, pred_lasso2)^2)
)

# Display model performance
cat("Model Performance on Dataset 1:\n")
print(performance1)

cat("\nModel Performance on Dataset 2:\n")
print(performance2)

# Display feature importances
cat("\nFeature Importance for Dataset 1 (Random Forest):\n")
print(importance_rf1)

cat("\nFeature Importance for Dataset 1 (XGBoost):\n")
print(importance_xgb1)

cat("\nFeature Importance for Dataset 1 (Lasso Regression):\n")
print(importance_lasso1)

cat("\nFeature Importance for Dataset 2 (Random Forest):\n")
print(importance_rf2)

cat("\nFeature Importance for Dataset 2 (XGBoost):\n")
print(importance_xgb2)

cat("\nFeature Importance for Dataset 2 (Lasso Regression):\n")
print(importance_lasso2)


#dataset 1
library(ggplot2)

features1 <- names(dataset1[, -ncol(dataset1)])
n_features1 <- length(features1)

plot_list1 <- list()

for (i in 1:n_features1) {
  plot_list1[[i]] <- ggplot(dataset1, aes_string(features1[i], "price")) +
    geom_point(alpha = 0.5) +
    theme_minimal() +
    labs(x = features1[i], y = "Price", title = paste(features1[i], "vs Price"))
}

# Print plots
gridExtra::grid.arrange(grobs = plot_list1, ncol = 3)

#for dataset 2

# Select relevant columns for plotting
features2 <- c("bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated")
n_features2 <- length(features2)

plot_list2 <- list()

for (i in 1:n_features2) {
  plot_list2[[i]] <- ggplot(dataset2, aes_string(features2[i], "price")) +
    geom_point(alpha = 0.5) +
    theme_minimal() +
    labs(x = features2[i], y = "Price", title = paste(features2[i], "vs Price"))
}

# Print plots
gridExtra::grid.arrange(grobs = plot_list2, ncol = 4)




