# David Harar

# Machine Learning Online Class - Exercise 2: Logistic Regression
rm(list = ls())
gc()

# home wd
setwd("C:/Users/Daivid Harar/Desktop/ML/course 1 - standford/ex2/")

# work wd
#setwd("C:/Users/davidh/Desktop/ml/ex2/")

# Q1 - admition
# Your task is to build a classifcation model that estimates an applicant's
# probability of admission based the scores from those two exams

# 0 - Setup -----
# Libraries ----
library(dplyr)
library(ggplot2)
library(tidyverse)
# Functions ----


# load data ----
ex2data1 <- read.csv("./ex2data1.csv", header = F)
names(ex2data1) <- c("exam1", "exam2", "y")
X <- cbind(rep(1,nrow(ex2data1)),ex2data1[,c("exam1", "exam2")]) %>% as.matrix()
y <- ex2data1[,"y"] %>% as.matrix()

# Part 1: Plotting ----
ex2data1 %>% ggplot(aes(x = exam1, y = exam2, color = y)) + 
  geom_point() +
  theme(legend.position = "none")

# 2 - sigmoid
sigmoid <- function(z){
  if (class(z) == "matrix"){
    g <- matrix(0,nrow = nrow(z), ncol = ncol(z))
    for(j in 1:ncol(z)){
      for(i in 1:nrow(z)){
        g[i,j] <- 1/(1+exp(-1*(z[i,j])))  
      }
    }
  }
  if (class(z) == "numeric") g <- 1/(1+exp((-1)*z))
  return(g)
}

# Part 2: Compute Cost and Gradient -----
'Since we have Xs that vary over a large range, we first need to normalize them. Otherwise,
sigmoid function will yield mostly ones and zeros. Im here using a basic function. Note:
Later replace it with the function you have built in PS1'

normalize <- function(X){
  X_norm <- as.matrix(X)
  mu    <- data.frame(matrix(nrow = 1, ncol = 0))
  sigma <- data.frame(matrix(nrow = 1, ncol = 0))
  for(j in 1:ncol(X)){
    if (length(unique(X[,j]))>2){
      mu_j  <- mean(X[,j])
      sigma_j <- sd(X[,j])
      X_norm[,j] <- (X_norm[,j]-rep(mu_j,nrow(X_norm)))/sigma_j
      
      mu_temp <- data.frame(matrix(mu_j, nrow = 1, ncol = 1))       # keep mu and sigma for
      names(mu_temp) <- colnames(X)[j]                              # analysis stage
      mu <- cbind(mu, mu_temp)
      sigma_temp <- data.frame(matrix(sigma_j, nrow = 1, ncol = 1))
      names(sigma_temp) <- colnames(X)[j]
      sigma <- cbind(sigma, sigma_temp)
    }
    else {
      mu_temp <- data.frame(matrix(0, nrow = 1, ncol = 1))        # if a dummy variable 
      names(mu_temp) <- colnames(X)[j]                            # then i dont need to
      mu <- cbind(mu, mu_temp)                                    # normalized
      sigma_temp <- data.frame(matrix(1, nrow = 1, ncol = 1))
      names(sigma_temp) <- colnames(X)[j]
      sigma <- cbind(sigma, sigma_temp)
    }
  }
  return(list(X_norm, mu, sigma))
}

# assume initial theta
theta <- matrix(0, nrow = ncol(X), ncol = 1)

costFunction <- function(X,y,theta) {
  m <- length(y)
  ones <- rep(1,m)
  X_norm <- normalize(X)[[1]]
  
  #J <- (1/m)*((-1)*t(y)%*%log(sigmoid(X_norm%*%t(theta)))-t(ones-y)%*%log(ones-sigmoid(X_norm%*%t(theta))))
  J <- (1/m)*((-1)*t(y)%*%log(sigmoid(X_norm%*%theta))-t(ones-y)%*%log(ones-sigmoid(X_norm%*%theta)))
  
  grad <- (1/m)*(t(X_norm)%*%sigmoid(X_norm%*%theta - y))
  #return(list(J, grad))
  return(J)
}

# Part 3: Optimizing using fminunc -------
# Note: learn how to use optim!

theta_0 <- theta
gradfunction <- function(X,y,theta) {
  m <- length(y)
  ones <- rep(1,m)
  X_norm <- normalize(X)[1]
  grad <- (1/m)*(t(X_norm)%*%sigmoid(X_norm%*%theta - y))
  return(grad)
}

predict_optim <- function(train, test, optim_resoults) {
  
  mu <- normalize(train)[[2]] %>% as.matrix() %>% t()
  sigma <- normalize(train)[[3]] 
  sigma <- 1/sigma %>% as.matrix() %>% t()
  
  test_mat <- test %>% as.matrix()
  coef <- optim_resoults$par %>% as.matrix()
  z <- t(coef)%*%((test_mat-mu)*sigma)
  
  r <- ifelse((exp(z)/(1+exp(z)))>0.5, 1, 0)
  return(print(paste("value: ",
                     (exp(z)/(1+exp(z))),
                     ", Prediction: ", r, sep = "")))                   # Not sure which one is the correct
  #return(print(1 - 1/(1+exp(z))))
  
}

optim_resoults <- (result <- optim(par = c(0,0,0), fn = costFunction,
                                   method = 'L-BFGS-B',
                                   X = X, y = y))

"(result <- optim(par = c(0,0,0), 
fn = costFunction, 
gr = gradfunction,
method = 'L-BFGS-B',
X = X, y = y))"

# test <- c(1, 45, 85) %>% as.matrix()

X_norm <- normalize(X)[[1]]
dat <- cbind(X_norm, y) %>% as.data.frame()
colnames(dat) <- c("X1", "X2", "X3", "y")
glm(y~ X1 + X2 + X3, data = dat, family = binomial(link = "logit")) %>% summary()
# same as optim_resoults. 


# un-normalized
dat <- cbind(X, y) %>% as.data.frame()
colnames(dat) <- c("X1", "X2", "X3", "y")
glm(y~ X1 + X2 + X3, data = dat, family = binomial(link = "logit")) %>% summary()

-25.16133 + 45*0.20623 + 85*0.20147
1/(1+exp((-1)*(-25.16133 + 45*0.20623 + 85*0.20147))) # = 0.776
' The right probability is given by the formula above. The problem is that the
coefficients of the normalized Xs are different. Nevertheless, the algorithm 
does work since it yield the same coefficients in the normalized case.'

' Since the training set was normalized, we need to normalized the test set 
(the one sample it contains) to fit the model. For that, I kept mu and 
sigma.'

# Part 4: Predict and Accuracies ----

predict_optim(train = X,
              test = c(1,45,85),
              optim_resoults = optim_resoults) # Yessss

confusion_matrix <- "TBA"


# Exercise 2: Logistic Regression -----
# setup----
rm(list = setdiff(ls(), lsf.str()))
gc()
# load data ----
ex2data2 <- read.csv("./ex2data2.csv", header = F)
names(ex2data2) <- c("test1", "test2", "y")
X <- ex2data2[,c("test1", "test2")] %>% as.matrix()
y <- ex2data2[,"y"] %>% as.matrix()

# Plotting ----
ex2data2 %>% ggplot(aes(x = test1, y = test2, color = y)) + 
  geom_point() +
  theme(legend.position = "none")

# Part 1: Regularized Logistic Regression ----
mapFeature <- function(X,degree) {
  X <- cbind(rep(1, nrow(X)),X)
  frmla <- "X[,2]+X[,3]+(X[,2]*X[,3])"
  for(d in 2:degree){
    tmp <- paste("+X[,2]^",d,"+X[,3]^",d,"+(X[,2]*X[,3])^",d, sep = "")
    frmla <- paste(frmla, tmp, sep = "")
  }
  #as.formula(frmla)
  return(frmla)
}
#lm(as.formula(paste("y ~ ",frmla,sep = "")), data = ex2data2) # It does work in a formula. speak tomarow with evyatar
#frmlaa <- mapFeature(X= X, 6)

costFunctionReg <- function(X,y,theta, lambda){
  m <- length(y)
  ones <- rep(1,m)
  X_norm <- normalize(X)[[1]]
  theta_1 <- theta[-1] %>% as.matrix()
  
  #J <- (1/m)*((-1)*t(y)%*%log(sigmoid(X_norm%*%t(theta)))-t(ones-y)%*%log(ones-sigmoid(X_norm%*%t(theta))))
  J <- (1/m)*((-1)*t(y)%*%log(sigmoid(X_norm%*%theta))-t(ones-y)%*%log(ones-sigmoid(X_norm%*%theta)))+
    colSums((lambda/(2*m))*(theta_1^2))
  
  grad <- (1/m)*(t(X_norm)%*%sigmoid(X_norm%*%theta - y))
  #return(list(J, grad))
  return(J)
}

# case 1: initial theta is zero vector and lambda = 1  
optim_resoults <- (result <- optim(par = c(0,0,0), fn = costFunctionReg,
                                   method = 'L-BFGS-B',
                                   X = X, y = y, lambda = 1))
' I have got value of 0.6903344 (rather than .693) for function J.'

# case 2: initial theta is one vector and lambda = 10  
optim_resoults <- (result <- optim(par = c(1,1,1), fn = costFunctionReg,
                                   method = 'L-BFGS-B',
                                   X = X, y = y, lambda = 1000))
' Increasing lambda does make my coefficient turn close to 0, but the cost func
still happens to be of the same approximate size, .69'
optim_resoults <- (result <- optim(par = c(1,1,1), fn = costFunctionReg,
                                   gr = gradfunctionReg, # define gradfunctionReg and see if it solves the situation.
                                   method = 'L-BFGS-B',
                                   X = X, y = y, lambda = 10))
