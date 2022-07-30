#ifndef GUARD_utilities_h
#define GUARD_utilities_h
#include <RcppArmadillo.h>

arma::vec whichCpp(const arma::vec& vec, double value);
Rcpp::List whichList(const arma::vec& vec, const arma::vec& values);
Rcpp::List splitMatrix(const arma::mat& M, const arma::vec& vec, const arma::vec& values);
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma);

#endif
