#ifndef GUARD_gibbs_h
#define GUARD_gibbs_h
#include <RcppArmadillo.h>

Rcpp::List thetaDraw(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma);
arma::mat yDraw(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma, Rcpp::List draws);
arma::vec partialResid(Rcpp::List treeList, int remove, const arma::mat& W, Rcpp::NumericMatrix X, const Rcpp::NumericVector& y, int m, double sigma, Rcpp::List draws);
Rcpp::List thetaDrawsList(Rcpp::List treeList, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma);

#endif
