# David Harar

# PS 3 - Machine Learning - Multi-class Classifcation and Neural Networks
rm(list = ls())
gc()
options(scipen = 999)
library(dplyr)
library(ggpubr)
library(ggplot2)

# PART 1: Multi-Class Classification ------
# Since Andrew's code meant to work with Matlab, I had to creat my own function to print
# digits.
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


# work directory
setwd("C:/Users/davidh/Desktop/ml/ex3/")
# home directory
# setwd("C:/Users/Daivid Harar/Desktop/ML/course 1 - standford/ex3/")

# 1.1 Multi-class Classification unsing One-vs-all ----
ex3data1 <- R.matlab::readMat("./ex3data1.MAT")
X <- ex3data1$X
Y <- ex3data1$y

display_digits(number = 100, X= X)

# 1.2a: Vectorize Logistic Regression ------
# You will be using multiple one-vs-all logistic regression models to build a
# multi-class classifier. Since there are 10 classes, you will need to train 10
# separate logistic regression classifiers.

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

costFunctionReg <- function(X,y,theta, lambda){
  # mannualy change X for now
  m <- length(y)
  ones <- rep(1,m)
  #X_norm <- normalize(X)[[1]]
  X_norm <- X
  
  theta_1 <- theta[-1] %>% as.matrix()
  
  #J <- (1/m)*((-1)*t(y)%*%log(sigmoid(X_norm%*%t(theta)))-t(ones-y)%*%log(ones-sigmoid(X_norm%*%t(theta))))
  J <- (1/m)*((-1)*t(y)%*%log(sigmoid(X_norm%*%theta))-t(ones-y)%*%log(ones-sigmoid(X_norm%*%theta)))+
    colSums((lambda/(2*m))*(theta_1^2))
  grad <- (1/m)*(t(X_norm)%*%sigmoid(X_norm%*%theta - y))
  #return(list(J, grad))
  return(J)
} 

h_theta <- function(theta, X){
  h_theta_vec <- sigmoid(X%*%theta)
  return(h_theta_vec)
}

gradfunctionReg <- function(X,y,theta, lambda) {
  
  m <- length(y)
  ones <- rep(1,m)
  #X_norm <- normalize(X)[1]
  
  normalized_gradient_j <- (1/m)*(t(X)%*%(h_theta(theta = theta, X= X)-y)) + (lambda/m)*theta
  normalized_gradient_j[1,1] <- (1/m)*(t(X)%*%(h_theta(theta = theta, X= X)-y))[1,1]
  
  return(normalized_gradient_j)
}

# Test case for CostFunction and gradfunctionReg
theta_t = matrix(c(-2,-1,1,2), ncol = 1)
X_t = cbind(rep(1,5), matrix(c(1:15)/10, ncol = 3, nrow = 5))
y_t = matrix(c(1,0,1,0,1), ncol = 1)
lambda_t = 3

# Note: I changed the functions that they would not normalize X and only than the 
# numbers were correct.

costFunctionReg(X = X_t,
                y= y_t,
                theta = theta_t,
                lambda = lambda_t) # = 2.534819
gradfunctionReg(X = X_t, 
                y = y_t,
                theta = theta_t,
                lambda = lambda_t)

# 1.2b: One-vs-All Training ----
lambda = 0.1
OneVsAll <- function(X, y, lambda) {
  # ONEVSALL trains multiple logistic regression classifiers and returns all
  # the classifiers in a matrix all_theta, where the i-th row of all_theta 
  # corresponds to the classifier for label i
  #    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
  #    logistic regression classifiers and returns each of these classifiers
  #    in a matrix all_theta, where the i-th row of all_theta corresponds 
  #    to the classifier for label i
  
  # Note: X should contain 1's column BEFORE entering this function
  
  m <- nrow(X)
  n <- ncol(X)
  k <- length(unique(y))
  y_actual <- matrix(0, nrow = k, ncol = 1)
  
  initial_theta_mat <- matrix(0, nrow = k, ncol = (n+1))
  initial_theta_vec <- matrix(0, nrow = 1, ncol = (n+1))
  # Theta (initial) is of dim K*(n+1)
  # Add ones to the X data matrix
  X <- cbind(matrix(1, nrow = nrow(X), ncol = 1), X)
  
  for (j in 1:length(unique(y))) {
    y_j <- ifelse(y == unique(y)[j],1,0) %>% as.matrix()
    
    optim_resoults_j <- (result <- optim(initial_theta_vec, 
                                         fn = costFunctionReg,
                                         gr = gradfunctionReg,
                                         method = 'L-BFGS-B',
                                         X = X, y = y_j, lambda = 0.1))
    
    theta_j <- optim_resoults_j$par
    initial_theta_mat[j,] <- theta_j
    
    y_actual[j,1] <- unique(y)[j]
    
  }
  return(list(initial_theta_mat,y_actual))
}

# Test for 100 random 100 observations
X_large <- X

digits_sample <- sample(1:nrow(X_large), 100)
X <- X_large[digits_sample,]
y <- Y[digits_sample,]

THETA <- OneVsAll(X = X,
                  y = y,
                  lambda = 0.01)[[1]]
y_actual <- OneVsAll(X = X,
                     y = y,
                     lambda = 0.01)[[2]]


predictOneVsAll <- function(X, y, theta_mat, y_pred) {
  
  # X: data matrix
  # y: labels vector
  # theta_mat: matrix for coefficients/wights
  # y_pred: the y value each of the *rows* of theta_mat is trying to predict
  
  
  k <- length(unique(y))
  X <- cbind(matrix(1, nrow = nrow(X), ncol = 1), X)
  z <- X%*%t(THETA)
  prob <- sigmoid(z)
  max_prob_col <- Rfast::rowMaxs(prob, value = F)    # fiding which of the numbers the weights predicted
  
  prob_with_pred <- matrix(nrow = 0, ncol = (k+1+1)) # k for prediction for each category, 1 for predicted and 1 for actual
  
  for (i in 1:nrow(X)){
    tmpmat <- prob_with_pred
    tmpval <- cbind(t(as.matrix(prob[i,])), 
                    as.matrix(t(y_actual)[max_prob_col[i]], ncol = 1), # prediction (think of another name for this y)
                    as.matrix(y[i]))                                   # the actual actual y
    prob_with_pred <- rbind(tmpmat, tmpval)
  }
  
  colnames(prob_with_pred) <- c(paste("category_", 1:k, sep = ""), "category_predicted", "label")
  return(prob_with_pred)
}

digits_sample <- sample(1:nrow(X_large), 500)
X <- X_large[digits_sample,]
y <- Y[digits_sample,]

THETA <- OneVsAll(X = X,
                  y = y,
                  lambda = 0.01)[[1]]
y_actual <- OneVsAll(X = X,
                     y = y,
                     lambda = 0.01)[[2]]

solution <- predictOneVsAll(X = X,
                            y = y,
                            theta_mat = THETA,
                            y_pred = y_actual)
nrow(solution[solution[,11] != solution[,12],]) # no mistakes, seems wrong, recheck tommarow

# all sample
THETA <- OneVsAll(X = X_large,
                  y = Y,
                  lambda = 0.01)[[1]]
y_actual <- OneVsAll(X = X_large,
                     y = Y,
                     lambda = 0.01)[[2]]

solution_all <- predictOneVsAll(X = X_large,
                                y = Y,
                                theta_mat = THETA,
                                y_pred = y_actual)
nrow(solution_all[solution_all[,11] != solution_all[,12],]) # 183 mistakes :)
solution_all %>% View()

# Bonus - Viewing the wronge predictions and see whether the handwrite there is a mess ----
solution_all <- cbind(solution_all, 1:nrow(solution_all))
wrong_pred_indecies <- solution_all[solution_all[,11] != solution_all[,12],] %>% .[,13]
wrong_pred_data <- X_large[wrong_pred_indecies,]

# Let's take a subset of this subsample, the closest number with an integer root (13^12 = 169)
display_digits(number = 169, X = wrong_pred_data)
# They do ugly, I can't say they the ugliest.

# PART 2: Neural Networks -----
rm(list = setdiff(ls(), lsf.str())) # remove all but the functions
gc()

input_layer_size  = 400          # 20x20 Input Images of Digits
hidden_layer_size = 25           # 25 hidden units
num_labels = length(unique(y))   # 10 labels, from 1 to 10   
# work directory
setwd("C:/Users/davidh/Desktop/ml/ex3/")
# home directory
# setwd("C:/Users/Daivid Harar/Desktop/ML/course 1 - standford/ex3/")    
ex3data1 <- R.matlab::readMat("./ex3data1.MAT")
X <- ex3data1$X
Y <- ex3data1$y
# Part 2.1: Loading and Visualizing Data ----
display_digits(number = 100, X= X)
# Part 2.2: Loading Pameters
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.
ex3weights <- R.matlab::readMat("./ex3weights.mat")
dim(ex3weights[[1]]) # 25 rows, 401 cols
dim(ex3weights[[2]]) # 10 rows, 26 cols

ex3weights[[1]] %>% View()
ex3weights[[2]] %>% View()

# Part 2.3: Implement Predict -------
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.

THETA1 <- ex3weights[[1]]
THETA2 <- ex3weights[[2]]

dim(X_first_stage_NN)
dim(X_first_stage_NN%*%t(THETA1))


A_2 <- sigmoid(X_first_stage_NN%*%t(THETA1))
# Note: A_2 is a marix of (a_1)^2, (a_2)^2, ..., (a_25)^2
A_2 <- cbind(matrix(1, nrow = nrow(A_2), ncol = 1),
             A_2)

A_3 <- sigmoid(A_2%*%t(THETA2)) 
# Note: Now, for each observation we have 10(=k) predictions. Similarely to OneVsAll, we will 
#       predict the category with the highest predicted value.

THETA1 <- ex3weights[[1]]
THETA2 <- ex3weights[[2]]


NNpredict <- function(Theta1, Theta2, X, y){
  
  X_first_stage_NN <- cbind(matrix(1, nrow = nrow(X), ncol = 1),
                            X)
  
  #dim(X_first_stage_NN)
  #dim(X_first_stage_NN%*%t(THETA1))
  
  A_2 <- sigmoid(X_first_stage_NN%*%t(THETA1))
  # Note: A_2 is a marix of (a_1)^2, (a_2)^2, ..., (a_25)^2
  A_2 <- cbind(matrix(1, nrow = nrow(A_2), ncol = 1),  # add a bias vector
               A_2)
  
  A_3 <- sigmoid(A_2%*%t(THETA2)) 
  # Note: Now, for each observation we have 10(=k) predictions. Similarely to OneVsAll, we will 
  #       predict the category with the highest predicted value.
  
  y_pred <- Rfast::rowMaxs(A_3, value = F)
  
  output <- cbind(matrix(1:nrow(A_3), ncol = 1, nrow = nrow(A_3)), # if we do not sample Xs
                  A_3,                                             # the matrix with predictions 
                  y_pred,                                          # *assuming* labels ordered
                  y)                                               # the actual label
  return(output)
}

X <- X_large

NN <- NNpredict(Theta1 = THETA1,
                Theta2 = THETA2,
                X= X,
                y = Y)
NNdf <- as.data.frame(NN)

1-nrow(NNdf[NNdf$y_pred != NNdf$V13,])/nrow(NNdf) # Approx 97.5% as stated in the PS


