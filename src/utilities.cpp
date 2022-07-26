#include "utilities.h"

// Split matrix for tapply-like functions
// Find rows to split (equivalent to R function which)
// [[Rcpp::export]]
Rcpp::NumericVector whichCpp(const arma::vec& vec, double value)
{
  arma::uvec out = arma::find(vec == value);
  Rcpp::NumericVector outR = Rcpp::NumericVector(out.begin(),out.end());
  return(outR);
}
// [[Rcpp::export]]
Rcpp::List whichList(const arma::vec& vec, Rcpp::NumericVector values)
{
  Rcpp::List out;
  for (int i=0; i<values.length(); i++) out.push_back(whichCpp(vec,values(i)));
  return out;
}
// [[Rcpp::export]]
Rcpp::List splitMatrix(Rcpp::NumericMatrix M, const arma::vec& vec, Rcpp::NumericVector values)
{
  Rcpp::List rows = whichList(vec,values);
  Rcpp::List out;
  for (int i=0; i<rows.length(); i++)
    {
      Rcpp::NumericVector rowsVec = rows[i];
      Rcpp::NumericMatrix X(rowsVec.length(),M.ncol());
      for (int j=0; j<rowsVec.length(); j++)
	{
	  X.row(j) = M.row(rowsVec[j]);
	}
      out.push_back(X);
    }
  return(out);
}

// Draw from MVN
// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
   int ncols = sigma.n_cols;
   arma::mat Y = arma::randn(n, ncols);
   return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}
