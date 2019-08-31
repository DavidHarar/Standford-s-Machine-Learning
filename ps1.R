
# Machine Learning Online Class - Exercise 1: Linear Regression
# Setup ----
library("dplyr")
library("ggplot2")
rm(list = ls())
gc()
# Q2 - gradient decendence ----
  # functions----
    computeCost <- function(X, y, theta){
      # COMPUTECOST Computes cost for linear regression
      #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
      #   parameter for linear regression to fit the data points in X and y
      
      # Initialize some useful values
      m = length(y) # number of training examples
      
      # You need to return the following variables correctly 
      J <- 1/(2*m)*rowSums(t(X%*%theta-y)%*%(X%*%theta-y))
      
      # Instructions: Compute the cost of a particular choice of theta
      #               You should set J to the cost.
      
      return(J)
      
    }
    gradientDescent <- function(X, y, theta, alpha, num_iters) {
      # GRADIENTDESCENT Performs gradient descent to learn theta
      #   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
      #   taking num_iters gradient steps with learning rate alpha
      
      # Initialize some useful values
      m = length(y) # number of training examples
      J_history = rep(0, num_iters) 
      
      for (p in 1:num_iters){
        #alpha_temp <- alpha*(0.333^(p-1))
        
        # temporary values under linear regression:
        for(j in 1:ncol(X)){
          assign(paste("temp",(j-1), sep = ""), theta[j,1]-alpha*((1/m)*(t(X%*%theta-y)%*%X[,j])))
        }
        # one var example: temp1 <- theta[2,1]-alpha*(1/m*rowSums((t(theta)%*%t(X)-y)%*%X[,2]))
        
        # update theta values:
        for (j in 0:(nrow(theta)-1)){
          temp <- get(paste("temp", j, sep = ""))
          l <- j+1
          theta[l,1] <- temp
          #theta[2,1] <- temp1  
        }
        
        # Now we have updated thetas 
        # return:
        # Save the cost J in every iteration :   
        J_history[p] <- computeCost(X, y, theta)
      }
      return(list(theta, J_history))
      # return(theta)
      # return(J_history)
    }
    featureNormalize <- function(X) {
      X_norm <- X
      mu     <- matrix(0, nrow = 1, ncol = ncol(X))
      sigma  <- matrix(0, nrow = 1, ncol = ncol(X))
      # Subtract mean value
      for (j in 1:ncol(X)){
        if (length(unique(X[,j]))>2 &
            class(X[,j]) != "factor"){ # do not compute for
          # binary variables and
          # factors
          mu[1,j]    <- mean(X[,j])
          sigma[1,j] <- sd(X[,j])
          X_norm[,j] <- (X[,j]-mu[1,j])/sigma[1,j]
        }
      }
      return(X_norm)
    }
    normalEqn <- function(X,y) {
      theta_closed <- solve((t(X)%*%X))%*%t(X)%*%y
      return(theta_closed)
    }
#-----

  # load data----
setwd("C:/Users/Daivid Harar/Desktop/ML/course 1 - standford/ex1/") # set your working directory here
A <- read.csv("./ex1data1.csv")
names(A) <- c("pop", "profit_per_ft")

  # ploting
  A %>% ggplot(., aes(x = pop, y = profit_per_ft)) + 
    geom_point() + 
    geom_smooth(method = "lm")
  
  # built in method for linear model
  lm_model <- A %>% lm(profit_per_ft ~ pop, data = .)
  lm_model %>% summary()
  
  B <- cbind(rep(1, nrow(A)),A)
  names(B) <- c("ones", names(A))
  
  X <- B[,1:2] %>% as.matrix()
  y <- B[,3]
  m <- length(y)
  theta_lm <- solve((t(X)%*%X))%*%t(X)%*%y # linear algebra method for OLS
#-----

  # GD guessing ----
  theta <- matrix(0, ncol = 1, nrow = 2)
  iterations = 15000
  alpha = 0.01
  theta_GDalg <- gradientDescent(X = X, 
                                 y = y,
                                 theta = theta,
                                 alpha = alpha,
                                 num_iters = iterations)[[1]]
  J_GDalg <- gradientDescent(X = X, 
                             y = y,
                             theta = theta,
                             alpha = alpha,
                             num_iters = iterations)[[2]]
#-----
# Q3 - housing prices (multiple variables) ----
  rm(list = setdiff(ls(), lsf.str())) # keep only the functions
  gc()
  A <- read.csv("./ex1data2.csv")
  names(A) <- c("size_sqft", "no_bdrms", "price")
  X <- cbind(rep(1, nrow(A)),A[,c("size_sqft", "no_bdrms")]) %>% as.matrix() # predictors with a constant
  y <- A[,"price"]   %>% as.matrix()
  X_norm <- featureNormalize(X)  
  theta_LnrAlg <- solve((t(X)%*%X))%*%t(X)%*%y # linear algebra method for OLS
  
  A %>% lm(data = ., price ~ no_bdrms + size_sqft) %>% summary()
  
  theta <- matrix(0, ncol = 1, nrow = ncol(X))
  
  iterations = 150000
  alpha = 0.001
  theta_GDalg <- gradientDescent(X = X_norm, 
                                 y = y,
                                 theta = theta,
                                 alpha = alpha,
                                 num_iters = iterations)[[1]]
  
  z <- gradientDescent(X = X_norm, 
                       y = y,
                       theta = theta,
                       alpha = alpha,
                       num_iters = iterations)[[2]]
  
  
  J_graph <- cbind(c(1:length(z)), z) %>% 
       as.data.frame()
     J_graph %>% ggplot(data = ., aes(x = V1, 
                                      y = z)) +
       geom_line()
  # the graph have been converged but the values did not.
  
  