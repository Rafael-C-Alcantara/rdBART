#include "utilities.h"

// Split matrix for tapply-like functions
// Find rows to split (equivalent to R function which)
// [[Rcpp::export]]
arma::vec whichCpp(const arma::vec& vec, double value)
{
  arma::uvec which = arma::find(vec == value);
  // Return position of elements (0-index)
  arma::vec ret = arma::conv_to<arma::vec>::from(which);
  return(ret);
}
// [[Rcpp::export]]
Rcpp::List whichList(const arma::vec& vec, const arma::vec& values)
{
  Rcpp::List out;
  for (int i=0; i<values.n_elem; i++) out.push_back(whichCpp(vec,values(i)));
  return out;
}
// [[Rcpp::export]]
Rcpp::List splitMatrix(const arma::mat& M, const arma::vec& vec, const arma::vec& values)
{
  Rcpp::List rows = whichList(vec,values);
  Rcpp::List out;
  for (int i=0; i<rows.length(); i++)
    {
      arma::vec rowsVec = rows[i];
      arma::mat X(rowsVec.n_elem,M.n_cols);
      for (int j=0; j<rowsVec.n_elem; j++)
	{
	  X.row(j) = M.row(rowsVec(j));
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
