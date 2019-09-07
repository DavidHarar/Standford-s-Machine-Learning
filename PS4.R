# Machine Learning Online Class - Exercise 4 Neural Network Learning -------

# Libraries----
  rm(list = ls())
  gc()
  options(scipen = 999)
  library(dplyr)
  library(ggpubr)
  library(ggplot2)
  
  # work directory
  #setwd("C:/Users/davidh/Desktop/ml/ex4/")
  # home directory
   setwd("C:/Users/Daivid Harar/Desktop/ML/course 1 - standford/ex4/")
  
# Functions -------
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
  nncostfunction <- function(X, y, Theta_vector, 
                                L1_in, L1_out,
                                L2_in, L2_out,
                                lambda){
    # think of how to generalized this using ... and lists for starts list and ends list
    k <-unique(y)
    m <- nrow(y)
    # turn y into matrix
    y_mat <- matrix(0, nrow = length(y), ncol = k)
    colnames(y_mat) <- paste("y_", 1:10, sep = "")
    for(i in 1:m) {
      y_mat[i,y[i]] <- 1
    }
    ones_mat_y <- matrix(1, ncol = ncol(y_mat), nrow = nrow(y_mat))
    
    L1_in1 <- L1_in + 1
    L2_in1 <- L2_in + 1
    
    THETA1 <- matrix(Theta_vector[1:(L1_in1*L1_out)], nrow = L1_out, ncol = L1_in1)
    THETA2 <- matrix(Theta_vector[((L1_in1*L1_out)+1):((L1_in1*L1_out)+(L2_in1*L2_out))], nrow = L2_out, ncol = L2_in1)
    
    # forward propagation (feedforward)
    ones <- matrix(1, ncol = 1, nrow = nrow(y_mat))
    a1 <- cbind(ones, X)
    z2 <- a1%*%t(THETA1)
    a2 <- sigmoid(z2)
    a2 <- cbind(ones,a2)
    z3 <- a2%*%t(THETA2)
    a3 <- sigmoid(z3)
    h_theta <- a3
    
    # Backpropagation
    delta3 <- a3-y_mat
    delta2 <- (delta3%*%THETA2[,2:ncol(THETA2)])*(sigmoidGradient(a2[,2:ncol(a2)])) 
      # since THETA2 and a2 include bias units, we need to ignore them when we calculate delta2.
    DELTA1 <- t(delta2)%*%a1
    DELTA2 <- t(delta3)%*%a2
    
    THETA1_grad <- (1/m)*(DELTA1+(lambda*THETA1))
    THETA2_grad <- (1/m)*(DELTA2+(lambda*THETA2))
    THETA1_grad[,1] <- (1/m)*DELTA1[,1]
    THETA2_grad[,1] <- (1/m)*DELTA2[,1]
      
    
    grad <- mget(ls(pattern = "_grad"))
    
    # Unregularized cost function
    #J <- (-1/m)*(y_mat*log(h_theta)+(ones_mat_y-y_mat)*log(ones_mat_y-h_theta))
    
    # Regularized cost function
    THETA1_1 <- THETA1[,2:ncol(THETA1)]
    THETA2_1 <- THETA2[,2:ncol(THETA2)]
    J <- sum((-1/m)*(y_mat*log(h_theta)+(ones_mat_y-y_mat)*log(ones_mat_y-h_theta)))+
      (lambda/(2*m))*(sum(THETA1_1^2) + sum(THETA2_1^2))
    sum(J) 
    cost <- sum(J)
    
    resoults <- list(cost, unlist(grad))
    return(resoults)
  }
  rotate <- function(x) t(apply(x, 2, rev))
  display_digits <- function(number, X) {
    graph_rows <- round(sqrt(number))
    graph_cols <- round(sqrt(number))
    digits_sample <- sample(1:nrow(X), number)
    sampleX <- X[digits_sample,]
    
    # First I seperate between single "photos". 20 would be the hight.
    # Create digit matrecies from the gross rows
    for (j in 1:number){
      tmp <- rbind(matrix(1, ncol=20, nrow = 1),                           # seperate horizontaly
                   rotate(rotate(rotate(t(matrix(sampleX[j,],20,20))))),
                   matrix(1, ncol=20, nrow = 1))
      tmp <- cbind(matrix(1, ncol=1, nrow = 22),                           # separete vertically
                   tmp,
                   matrix(1, ncol=1, nrow = 22))
      assign(paste("mat_",j, sep = ""),tmp)
    }
    
    # creating rows from columns indecies
    for (m in 1:sqrt(number)){
      assign(paste("graph_heatmap_row_",m,sep=""),matrix(ncol = 0, nrow = 22))
    }
    
    m <- 1
    for (j in 1:number) {
      xx <- get(paste("mat_",j, sep = ""))
      yy <- get(paste("graph_heatmap_row_",m,sep=""))
      
      if (ncol(yy)<(22*sqrt(number)-1)) {
        assign(paste("graph_heatmap_row_",m,sep=""),
               cbind(yy, xx))
      } else {
        m <- m+1
        yy <- get(paste("graph_heatmap_row_",m,sep=""))
        assign(paste("graph_heatmap_row_",m,sep=""),
               cbind(yy, xx))
      }
    }
    rows_list <- mget(ls(pattern = "graph_heatmap_row_"))
    
    mat <- do.call(rbind, rows_list)
    
    heatmap(mat,Rowv=NA,Colv=NA,col=paste("gray",1:100,sep=""))
    return(heatmap(mat,Rowv=NA,Colv=NA,col=paste("gray",1:100,sep=""))) 
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
  sigmoidGradient <- function(z){
    if (class(z) == "matrix"){
      dgdz <- matrix(0,nrow = nrow(z), ncol = ncol(z))
      for(j in 1:ncol(z)){
        for(i in 1:nrow(z)){
          dgdz[i,j] <- sigmoid(z[i,j])*(1-sigmoid(z[i,j]))
        }
      }
    }
    if (class(z) == "numeric") dgdz <- sigmoid(z)*(1-sigmoid(z))
    return(dgdz)
  }
  RANDINITIALIZEWEIGHTS <- function(L_in, L_out, epsilon) {
    W <- matrix(0, nrow = L_out, ncol = (L_in+1))
    for (i in 1:nrow(W)) {
      for (j in 1:ncol(W)) {
        W[i,j] <- runif(1, -epsilon, epsilon)
      }
    }
    return(W)
  } 
  
# Part 1 - Neural Networks ----
  # 1.1 Loading and Visualizing Data ----
  ex4data1 <- R.matlab::readMat("./ex4data1.MAT")
  X <- ex4data1$X
  Y <- ex4data1$y
  
  display_digits(number = 100, X= X)
  
  # 1.2 Model representation ----
  weights <- R.matlab::readMat("./ex4weights.MAT")
  THETA1 <- weights[[1]]
  THETA2 <- weights[[2]]
  unrolled_THETA1 <- matrix(THETA1, ncol = 1)
  unrolled_THETA2 <- matrix(THETA2, ncol = 1)
  unrolled_THETA <- rbind(unrolled_THETA1, unrolled_THETA2)
  unrolled_THETA <- as.vector(unrolled_THETA)
  
  # 1.3 Feedforward and cost function
      # Unregularized cost function
  y <- Y # for some reason I have to do this.
   nncostfunction(X = X,
                      y = y,
                      Theta_vector = unrolled_THETA,
                      L1_in = 400,
                      L1_out = 25,
                      L2_in = 25,
                      L2_out = length(unique(y)),
                      lambda = 0)[[1]] # 0.287629 
      
  # 1.4: Implement Regularization
    nncostfunction(X = X,
                   y = y,
                   Theta_vector = unrolled_THETA,
                   L1_in = 400,
                   L1_out = 25,
                   L2_in = 25,
                   L2_out = length(unique(y)),
                   lambda = 1)[[1]] # 0.383770
      
# 2. Backpropagation
  # 2.1. Sigmoid gradient function
      sigmoidGradient(0) # # test == .25
  # 2.2. Random initialization
      # When training neural networks, it is important to randomly initialize the parameters 
      # for symmetry breaking. One effective strategy for random initialization is to randomly 
      # select values for THETA(l) uniformly in the range [-epsilon, epsilon].
      initial_Theta1 = RANDINITIALIZEWEIGHTS(L_in = 400, L_out = 25, epsilon = 0.12)
      initial_Theta2 = RANDINITIALIZEWEIGHTS(L_in = 25, unique(y), epsilon = .12)
      
      unrolled_initial_THETA1 <- matrix(initial_Theta1, ncol = 1)
      unrolled_initial_THETA2 <- matrix(initial_Theta2, ncol = 1)
      unrolled_initial_THETA <- rbind(unrolled_initial_THETA1, unrolled_initial_THETA2)
      unrolled_initial_THETA <- as.vector(unrolled_initial_THETA)
      
      nncostfunction(X = X,
                     y = y,
                     Theta_vector = unrolled_initial_THETA,
                     L1_in = 400,
                     L1_out = 25,
                     L2_in = 25,
                     L2_out = length(unique(y)),
                     lambda = 0)[[1]] # 6.77
      
  # 2.3. Backpropagation
      nncostfunction(X = X,
                     y = y,
                     Theta_vector = unrolled_THETA,
                     L1_in = 400,
                     L1_out = 25,
                     L2_in = 25,
                     L2_out = length(unique(y)),
                     lambda = 0)[[2]]

  # 2.4. Gradient Checking
      # TBA
  
  # 2.5. Regularized Neural Networks
      nncostfunction(X = X,
                     y = y,
                     Theta_vector = unrolled_THETA,
                     L1_in = 400,
                     L1_out = 25,
                     L2_in = 25,
                     L2_out = length(unique(y)),
                     lambda = 3)[[2]]
  
  # 2.6. Learning parameters using fmincg
    # Prof' Ng uses fmincg which he wrote. The main difference between "fminunc" and "fmincg" is the memory usage.
    # See the discussion in here:
    # https://stackoverflow.com/questions/12115087/octave-logistic-regression-difference-between-fmincg-and-fminunc
    # and especially gregS's comment.
      costFunctionReg <- function(X,y, Theta_vector, L1_in, L1_out, L2_in, L2_out, lambda){
        cost <-  nncostfunction(X = X,
                                y = y,
                                Theta_vector = unrolled_THETA,
                                L1_in = L1_in,
                                L1_out = L1_out,
                                L2_in = L2_in,
                                L2_out = L2_out,
                                lambda = lambda)[[1]]
        return(cost)
      }
      gradfunctionReg <- function(X,y, Theta_vector, L1_in, L1_out, L2_in, L2_out, lambda){
        grad <-  nncostfunction(X = X,
                                y = y,
                                Theta_vector = unrolled_THETA,
                                L1_in = L1_in,
                                L1_out = L1_out,
                                L2_in = L2_in,
                                L2_out = L2_out,
                                lambda = 3)[[2]]
        grad <- as.vector(grad)
        return(grad)
      }
      
        # Optim takes two functions, the cost and the gradient. In order to impement nncostfunction with 
        # optim, one first need to seperate it into two different functions.
      optim_resoults <- (result <- optim(unrolled_initial_THETA, 
                                           fn = costFunctionReg,
                                           gr = gradfunctionReg,
                                           method = 'L-BFGS-B',
                                           X = X, 
                                           y = y, 
                                           L1_in = 400,
                                           L1_out = 25,
                                           L2_in = 25,
                                           L2_out = length(unique(y)),
                                           lambda = 1))
        # value: 0.3837699! (Andrew's weights gained a cost of 0.383770 given lambda == 1)
      RollingBack <- function(Theta_vector, 
                              L1_in, L1_out,
                              L2_in, L2_out){
        L1_in1 <- L1_in + 1
        L2_in1 <- L2_in + 1
        
        THETA1 <- matrix(Theta_vector[1:(L1_in1*L1_out)], nrow = L1_out, ncol = L1_in1)
        THETA2 <- matrix(Theta_vector[((L1_in1*L1_out)+1):((L1_in1*L1_out)+(L2_in1*L2_out))], nrow = L2_out, ncol = L2_in1)
        return(list(THETA1, THETA2))
        }
      THETA_vec <- RollingBack(optim_resoults$par,
                               L1_in = 400,
                               L1_out = 25,
                               L2_in = 25,
                               L2_out = 10)
      calculated_THETA1 <- THETA_vec[[1]]
      calculated_THETA2 <- THETA_vec[[2]]
      
  NNpredict <- function(X, y, THETA1, THETA2) {
    k <-unique(y)
    m <- nrow(y)
    # turn y into matrix
    y_mat <- matrix(0, nrow = length(y), ncol = k)
    colnames(y_mat) <- paste("y_", 1:10, sep = "")
    for(i in 1:m) {
      y_mat[i,y[i]] <- 1
    }
    ones_mat_y <- matrix(1, ncol = ncol(y_mat), nrow = nrow(y_mat))
    
    ones <- matrix(1, ncol = 1, nrow = nrow(y_mat))
    a1 <- cbind(ones, X)
    z2 <- a1%*%t(THETA1)
    a2 <- sigmoid(z2)
    a2 <- cbind(ones,a2)
    z3 <- a2%*%t(THETA2)
    a3 <- sigmoid(z3)
    h_theta <- a3
    
    max_prob_col <- Rfast::rowMaxs(h_theta, value = F)    # fiding which of the numbers the weights predicted
    
    prob_with_pred <- matrix(nrow = 0, ncol = (k+1+1))
    i <- 1
    for (i in 1:nrow(X)){
      tmpmat <- prob_with_pred
      tmpval <- cbind(matrix(h_theta[i,], ncol = 10, nrow = 1), 
                      matrix(max_prob_col[i], ncol = 1, nrow = 1), # y_mat is organized and 10 means 0. 1 is 1 etc...
                      matrix(y[i], ncol = 1, nrow = 1))            # the actual actual y
      prob_with_pred <- rbind(tmpmat, tmpval)
    }
    
    colnames(prob_with_pred) <- c(paste("category_", 1:10, sep = ""), "category_predicted", "actual_label")
    
    true_prediction_ratio <- 1-sum(prob_with_pred[,11] != prob_with_pred[,12])/nrow(prob_with_pred)
    false_prediction_ratio <- sum(prob_with_pred[,11] != prob_with_pred[,12])/nrow(prob_with_pred)
    
    return(list(prob_with_pred,true_prediction_ratio,false_prediction_ratio))
  }
  
  NNpredict(X = X, y = y, THETA1 = THETA1, THETA2 = THETA2)[[2]]
  NNpredict(X = X, y = y, THETA1 = calculated_THETA1, THETA2 = calculated_THETA2)[[2]]
    # About 0.0248 difference between Anrew's weights and mine.

# Part 3: Visualizing the hidden layer
display_digits(16, calculated_THETA1[,2:ncol(calculated_THETA2)])
  # Not very beutiful, as in Andrew's pdf.

