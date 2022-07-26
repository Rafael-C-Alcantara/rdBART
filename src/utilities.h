#ifndef GUARD_utilities_h
#define GUARD_utilities_h
#include <RcppArmadillo.h>

Rcpp::NumericVector whichCpp(Rcpp::NumericVector vec, double value);
Rcpp::List whichList(const arma::vec& vec, Rcpp::NumericVector values);
Rcpp::List splitMatrix(Rcpp::NumericMatrix M, const arma::vec& vec, Rcpp::NumericVector values);
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma);

#endif
