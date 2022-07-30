#ifndef GUARD_metropolis_h
#define GUARD_metropolis_h
#include <RcppArmadillo.h>

// Calculate posterior covariance matrix
arma::mat lambdaPosteriorOne(const arma::mat& X, int m, double sigma);
Rcpp::List lambdaPosterior(Rcpp::List tree, const arma::mat& W, const arma::mat& X, int m, double sigma);
// Calculate posterior mean vector
arma::vec thetaPosteriorOne(const arma::mat& X, const arma::vec& y, const arma::mat& Lambda, double sigma);
Rcpp::List thetaPosterior(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& y, Rcpp::List Lambda, double sigma);
// Calculate tree log-likelihood
double logLikTree(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& y, int m, double sigma, Rcpp::List theta, Rcpp::List Lambda);
//Calculate MH ratio
double mhRatio(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, const arma::mat& X, const arma::vec y, int m, double sigma, double alpha, double beta, double ll0);
// Sample new tree
Rcpp::List newTree(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, const arma::mat& X, const arma::vec y, int m, double sigma, double alpha, double beta, double ll0);

#endif
