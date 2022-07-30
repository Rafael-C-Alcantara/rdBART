#include "tree.h"
#include "treeProposal.h"
#include "utilities.h"
#include "metropolis.h"
#include "gibbs.h"
//#include <Rcpp/Benchmark/Timer.h>

// Obtain draws of theta for each node
// [[Rcpp::export]]
Rcpp::List thetaDraw(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& y, int m, double sigma, Rcpp::List theta, Rcpp::List Lambda)
{
  Rcpp::List out;
  for (int i=0; i<theta.length(); i++)
    {
      out.push_back(mvrnormArma(1,theta[i],Lambda[i]));
    }
  return out;
}

// Assign the correct draw for each y
// [[Rcpp::export]]
arma::mat yDraw(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& yFit, Rcpp::List draws)
{
  arma::vec bn = bottomNodes(tree);
  int nobs = W.n_rows;
  int npar = X.n_cols;
  arma::mat out(nobs,npar);
  for (int i=0; i<nobs; i++)
    {
      arma::rowvec w = W.row(i);
      arma::uvec which = arma::find(bn == yFit[i]);
      int whichInt = arma::conv_to<int>::from(which);
      arma::mat draw = draws[whichInt];
      out.row(i) = draw;
    }
  return(out);
}

// Assign the correct fit for each y by bottom node
// [[Rcpp::export]]
arma::vec yPred(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& yFit, arma::mat yDraws)
{
  arma::mat out = yDraws%X;
  return(arma::sum(out,1));
}

// Calculate partial residuals given list of trees
// [[Rcpp::export]]
arma::vec partialResid(const arma::vec& y, const arma::mat& yPredMat, int remove)
{
  arma::vec out = y - arma::sum(yPredMat,1) + yPredMat.col(remove);
  return out;
}

// Compute residuals sum of squares for given tree
// [[Rcpp::export]]
double treeRSS(const arma::vec& yPred, const arma::vec& y)
{
  arma::vec resid = y-yPred;
  return(dot(resid,resid));
}

// Compute residual sum of squares for tree list
// [[Rcpp::export]]
double treeListRSS(const arma::mat& yPredMat, const arma::vec& y)
{
  double out = 0.0;
  for (int i=0; i<yPredMat.n_cols; i++) out += treeRSS(yPredMat.col(i),y);
  return out;
}

// Sample sigma
// [[Rcpp::export]]
double sigPost(double nu, double lambda, int m, int nobs, const arma::mat& yPredMat, const arma::vec& y)
{
  double a = 0.5*(nu+m*nobs);
  double b = 0.5*(nu*lambda + treeListRSS(yPredMat,y));
  double g = R::rgamma(a,1/b);
  return 1/g;
}
