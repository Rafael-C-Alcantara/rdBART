#ifndef GUARD_metropolis_h
#define GUARD_metropolis_h
#include <RcppArmadillo.h>

arma::mat lambdaPosteriorOne(const arma::mat& X, int m, double sigma);
Rcpp::List lambdaPosterior(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, int m, double sigma);
arma::vec thetaPosteriorOne(const arma::mat& X, const arma::vec& y, const arma::mat& Lambda, double sigma);
Rcpp::List thetaPosterior(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, Rcpp::List Lambda, double sigma);
double logLikTree(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, Rcpp::List theta, Rcpp::List Lambda);
double mhRatio(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, double alpha, double beta);
Rcpp::List newTree(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, double alpha, double beta);

#endif
