# David Harar

# Programming Exercise 5: Regularized Linear Regression and Bias v.s. Variance

# Libraries----
rm(list = ls())
gc()
options(scipen = 999)
library(dplyr)
library(ggpubr)
library(ggplot2)

# work directory
setwd("C:/Users/davidh/Desktop/ml/ex5/")
# home directory
#setwd("C:/Users/Daivid Harar/Desktop/ML/course 1 - standford/ex4/")

# Part 1: Regularized Linear Regression --------
  ex5data <- R.matlab::readMat("./ex5data1.mat")
  X <- ex5data$X
  y <- ex5data$y
  Xval <- ex5data$Xval
  yval <- ex5data$yval
  Xtest <- ex5data$Xtest
  ytest <- ex5data$ytest

## 1.1 Visualizing the dataset ----
  temp <- cbind(X,y) %>% as.data.frame()
  names(temp) <- c("X","y")
  ggplot(temp, aes(x=X, y = y)) + geom_point()

## 1.2 Regularized linear regression cost function ----
  linearRegCostFunction <- function(X, y, theta, lambda) {
    m <- nrow(y)
    theta_1 <- theta[-1] %>% as.matrix()
    X <- cbind(rep(1, nrow(X)),X)
    J <- 1/(2*m)*rowSums(t(X%*%theta-y)%*%(X%*%theta-y))+
      colSums((lambda/(2*m))*(theta_1^2))
    
    grad <- (1/m)*(t(X)%*%(X%*%theta - y)) + (lambda/m)*theta
    grad[1] <- (1/m)*(t(X[,1])%*%(X%*%theta - y))
    return(list(J,grad))
  }
# test
# theta <- matrix(c(1,1), nrow = 2, ncol = 1)
# linearRegCostFunction(X = X, y = y, theta = theta, lambda = 1) # 3.39932 as in the pdf

## 1.3 Regularized linear regression gradient ----
# theta <- matrix(c(1,1), nrow = 2, ncol = 1)
# linearRegCostFunction(X = X, y = y, theta = theta, lambda = 1)[[2]] # as in the pdf

## 1.4 Fitting linear regression ----
  costFunctionReg <- function(X, y, theta, lambda) {
    linearRegCostFunction(X = X, 
                          y = y,
                          theta = theta,
                          lambda =lambda)[[1]]
  } 
  gradfunctionReg <- function(X, y, theta, lambda) {
    linearRegCostFunction(X = X, 
                          y = y,
                          theta = theta,
                          lambda =lambda)[[2]]
  } 
  
  initial_theta <- matrix(c(1,1), nrow = 2, ncol = 1)
  optim_resoults <- (result <- optim(initial_theta, 
                                     fn = costFunctionReg,
                                     gr = gradfunctionReg,
                                     method = 'L-BFGS-B',
                                     X = X, 
                                     y = y, 
                                     lambda = 0))
  optim_resoults$par # inspect parameters
  
  ggplot(temp, aes(x=X, y = y)) + 
    geom_point() + 
    geom_abline(slope = 0.3677792, intercept = 13.0879012) + # those are the values from optim_resoults
    xlim(-50, 40) + ylim(-5, 40)

# Part 2: Bias-variance --------
## 2.1 Learning curves ----
  trainLinearReg <- function(X, y, lambda){
    initial_theta <- matrix(0, nrow = (ncol(X)+1), ncol = 1)
    optim_resoults <- (result <- optim(initial_theta, 
                                       fn = costFunctionReg,
                                       gr = gradfunctionReg,
                                       method = 'L-BFGS-B',
                                       X = X, 
                                       y = y, 
                                       lambda = lambda))
    theta <- optim_resoults$par
    return(theta)
  }
  
  learningCurve <- function(X, y, Xval, yval, lambda) {
    learningCurve_graph <- matrix(nrow = 0, ncol = 3)
    learningCurve_graph <- as.data.frame(learningCurve_graph)
    for(i in 1:nrow(X)){
      # sampeling
      X_temp <- matrix(X[1:i,], ncol=ncol(X), nrow = i)
      y_temp <- matrix(y[1:i,], ncol = ncol(y)) 
      # train theta
      theta_temp <- trainLinearReg(X = X_temp, y = y_temp, lambda = lambda)
      J_train_temp <- costFunctionReg(X = X_temp, y = y_temp, theta = theta_temp, lambda = lambda)
      # test computed theta on cross validation set
      J_cv_temp <- costFunctionReg(X = Xval, y = yval, theta = theta_temp, lambda = lambda)
      # graphing resoults
      learningCurve_graph_temp <- matrix(c(i, J_train_temp, J_cv_temp), nrow = 1, ncol = 3)
      names(learningCurve_graph_temp) <- names(learningCurve_graph)
      learningCurve_graph <- rbind(learningCurve_graph, learningCurve_graph_temp)
    }
    names(learningCurve_graph) <- c("m", "J_train", "J_cv")
    return(learningCurve_graph)
    }
  
  learning_curve_df <- learningCurve(X = X, y = y, Xval = Xval, yval = yval, lambda = 0)
  
  ggplot(learning_curve_df, aes(x=m)) + 
    geom_line(data = learning_curve_df, aes(y = J_train, colour = "Train")) + 
    geom_line(data = learning_curve_df, aes(y = J_cv, colour = "Cross Validation")) + 
    theme(legend.title = element_blank(), legend.position = "top") +
    labs(x = "Error", y = "Number of training examples", caption = "Learning Curve for Linear Regression") +
    theme(panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "grey"))
    
# both the train error and cross validation error are high when the number of training examples is increased. This
# reflects a high bias problem in the model - the linear regression model is too simple and is unable to fit our dataset well.


# Part 3: Polynomial regression --------
  polyFeatures <- function(X, p) {
    # this function assumes that X is an m by 1 matrix. 
    # It can be easily generalized using for loop over js, the columns of X.
    X_poly <- X
    for(l in 2:p){
      X_poly <- cbind(X_poly, (X[,1]^l))
    }
    return(X_poly)
  }
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

## 3.1 Learning Polynomial Regression ----
  # transform X_train
  X_poly <- polyFeatures(X = X, p = 8)
  X_poly_norm_list <- normalize(X = X_poly)
  X_poly_norm <- X_poly_norm_list[[1]]
  # transform X_cv
  Xval_poly <- polyFeatures(X = Xval, p = 8)
  Xval_poly_norm_list <- normalize(X = Xval_poly)
  Xval_poly_norm <- Xval_poly_norm_list[[1]]
  
  # Figure 4 - Polynomial Regression Fit 
  temp <- cbind(X,y) %>% as.data.frame()
  names(temp) <- c("X","y")
  
  ggplot(temp, aes(x=X, y = y)) + 
    geom_point() + 
    stat_smooth(aes(y = y),method = "lm", formula = y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) + I(x^8), size = 1) + 
    xlim(-50, 40) + ylim(-5, 40)
  
  temp_wide <- cbind(X_poly_norm,y) %>% as.data.frame()
  initial_theta <- matrix(1, nrow = 9, ncol = 1)
  
  optim_resoults <- (result <- optim(initial_theta, 
                                     fn = costFunctionReg,
                                     gr = gradfunctionReg,
                                     method = 'L-BFGS-B',
                                     X = X_poly_norm, 
                                     y = y, 
                                     lambda = 0))
  theta_poly_norm <- optim_resoults$par  # Note: I'm not sure how to implement a p degree abline (such as in part one) 
  
  # Figure 5 - Polynomial Regression Learning Curve
  learning_curve_df <- learningCurve(X = X_poly_norm, y = y, Xval = Xval_poly_norm, yval = yval, lambda = 0)
  
  ggplot(learning_curve_df, aes(x=m)) + 
    geom_line(data = learning_curve_df, aes(y = J_train, colour = "Train")) + 
    geom_line(data = learning_curve_df, aes(y = J_cv, colour = "Cross Validation")) + 
    theme(legend.title = element_blank(), legend.position = "top") +
    labs(x = "Error", y = "Number of training examples", caption = "Polynomial Regression Learning Curve (lamda = 0.0000)") +
    theme(panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "grey")) 

## 3.2 Adjusting the regularization parameter ----
# One way to combat the overfitting (high-variance) problem is to add regularization to the model.
# In this part, we would different Lambdas to see how regularization can lead to a better model.

  # Lambda = 1 (Figure 7):
  learning_curve_df <- learningCurve(X = X_poly_norm, y = y, Xval = Xval_poly_norm, yval = yval, lambda = 1)
  
  ggplot(learning_curve_df, aes(x=m)) + 
    geom_line(data = learning_curve_df, aes(y = J_train, colour = "Train")) + 
    geom_line(data = learning_curve_df, aes(y = J_cv, colour = "Cross Validation")) + 
    theme(legend.title = element_blank(), legend.position = "top") +
    labs(x = "Error", y = "Number of training examples", caption = "Polynomial Regression Learning Curve (lamda = 0.0000)") +
    theme(panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "grey")) 
  
  learning_curve_df <- learningCurve(X = X_poly_norm, y = y, Xval = Xval_poly_norm, yval = yval, lambda = 100)
  
  ggplot(learning_curve_df, aes(x=m)) + 
    geom_line(data = learning_curve_df, aes(y = J_train, colour = "Train")) + 
    geom_line(data = learning_curve_df, aes(y = J_cv, colour = "Cross Validation")) + 
    theme(legend.title = element_blank(), legend.position = "top") +
    labs(x = "Error", y = "Number of training examples", caption = "Polynomial Regression Learning Curve (lamda = 0.0000)") +
    theme(panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "grey")) 
