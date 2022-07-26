#include "tree.h"
#include "treeProposal.h"
#include "utilities.h"
#include "metropolis.h"
#include "gibbs.h"

// Obtain draws of theta for each node
// [[Rcpp::export]]
Rcpp::List thetaDraw(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma)
{
  Rcpp::List out;
  Rcpp::List Lambda = lambdaPosterior(tree,W,X,m,sigma);
  Rcpp::List theta  = thetaPosterior(tree,W,X,y,Lambda,sigma);
  for (int i=0; i<theta.length(); i++)
    {
      out.push_back(mvrnormArma(1,theta[i],Lambda[i]));
    }
  return out;
}
// Assign the correct draw for each y
// [[Rcpp::export]]
arma::mat yDraw(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma, Rcpp::List draws)
{
  Rcpp::IntegerVector yFit = fit(tree,W);
  Rcpp::NumericVector bn = bottomNodes(tree);
  int params = X.ncol();
  int obs = X.nrow();
  arma::mat out(obs,params);
  
  for (int i=0; i<yFit.length(); i++)
    {
      for (int j=0; j<bn.length(); j++)
	{
	  arma::mat draw = draws[j];
	  if (yFit[i] == bn[j]) out.row(i) = draw;
	}
    }
  return out;
}
// Calculate draws for list of trees
// [[Rcpp::export]]
Rcpp::List thetaDrawsList(Rcpp::List treeList, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma)
{
  Rcpp::List out;
  for (int i=0; i<treeList.length(); i++)
    {
      Rcpp::List tree = treeList[i];
      out.push_back(thetaDraw(tree,W,X,y,m,sigma));
    }
  return out;
}
// Calculate partial residuals given list of trees
// [[Rcpp::export]]
arma::vec partialResid(Rcpp::List treeList, int remove, const arma::mat& W, Rcpp::NumericMatrix X, const Rcpp::NumericVector& y, int m, double sigma, Rcpp::List draws)
{
  arma::mat armaX = Rcpp::as<arma::mat>(Rcpp::wrap(X));
  Rcpp::NumericVector outRcpp = Rcpp::clone(y);
  arma::vec out = Rcpp::as<arma::vec>(Rcpp::wrap(outRcpp));
  for (int i=0; i<treeList.length(); i++)
    {
      if (i!=remove-1)
	{
	  Rcpp::List tree   = treeList[i];
	  Rcpp::List draw   = draws[i];
	  arma::mat yDraws  = yDraw(tree,W,X,y,m,sigma,draw);
	  arma::mat treeFit = yDraws%armaX;
	  arma::mat treeFitSum = arma::cumsum(treeFit,1);
	  arma::vec treeFitCol = treeFitSum.col(0);
	  out -= treeFitCol;
	} else
	{
	  continue;
	}
    }
  return out;
}
