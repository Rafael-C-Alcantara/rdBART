#ifndef GUARD_gibbs_h
#define GUARD_gibbs_h
#include <RcppArmadillo.h>

// Obtain draws of theta for each node
Rcpp::List thetaDraw(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& y, int m, double sigma, Rcpp::List theta, Rcpp::List Lambda);
// Assign the correct draw for each y
arma::mat yDraw(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& yFit, Rcpp::List draws);
// Assign the correct fit for each y by bottom node
arma::vec yPred(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& yFit, arma::mat yDraws);
// Calculate partial residuals given list of trees
arma::vec partialResid(const arma::vec& y, const arma::mat& yPredMat, int remove);
// Compute residuals sum of squares for given tree
double treeRSS(const arma::vec& yPred, const arma::vec& y);
// Compute residual sum of squares for tree list
double treeListRSS(const arma::mat& yPredMat, const arma::vec& y);
// Sample sigma
double sigPost(double nu, double lambda, int m, int nobs, const arma::mat& yPredMat, const arma::vec& y);
#endif
