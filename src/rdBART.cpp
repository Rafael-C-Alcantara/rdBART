//#include <RcppArmadillo.h>
#include "tree.h"
#include "treeProposal.h"
#include "utilities.h"
#include "metropolis.h"
#include "gibbs.h"
#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::export]]
Rcpp::NumericVector rdBART(const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, double alpha, double beta, int m, int nDraws, int burn)
{
  Rcpp::Timer timer;
  timer.step("start");
  int nobs = X.nrow();
  int npar = X.ncol();
  Rcpp::NumericVector rcppY = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(y));
  // To update in the inner loop
  Rcpp::List treeList;
  for (int i=0; i<m; i++) treeList.push_back(node(1,NA_INTEGER,NA_INTEGER));
  Rcpp::List thetaList = thetaDrawsList(treeList,W,X,y,m,10);
  Rcpp::List yFitList;
  for (int i=0; i<m; i++) yFitList.push_back(yDraw(treeList[i],W,X,y,m,10,thetaList[i]));
  arma::mat accept(nDraws,m);
  // To update in the outter loop
  arma::vec sigDraws(nDraws,arma::fill::value(10.0));
  // Output
  Rcpp::List yFitDraws;
  timer.step("pre Loop");
  for (int i=0; i<nDraws; i++)
    {
      double sigma = sigDraws[i];
      arma::mat yFitDraw(nobs,npar);
      timer.step("time untill start inner loop");
      for (int j=0; j<m; j++)
	{
	  arma::vec residual = partialResid(treeList,j,W,X,rcppY,m,sigma,thetaList);
	  Rcpp::List propTree = proposal(treeList[j],W);
	  Rcpp::List nt = newTree(treeList[j],propTree,W,X,y,m,sigma,alpha,beta);
	  treeList[j]  = nt["tree"];
	  accept(i,j)  = nt["Acc"];
	  thetaList[j] = thetaDraw(treeList[j],W,X,y,m,sigma);
	  arma::mat yFitj = yDraw(treeList[j],W,X,y,m,sigma,thetaList[j]);
	  yFitList[j]  = yFitj;
	  yFitDraw += yFitj;
	}
      yFitDraws.push_back(yFitDraw);
    }
  timer.step("Time until end inner loop");
  /*
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("Theta") = yFitDraws, Rcpp::Named("Accept") = accept);
  return out;
  */
  Rcpp::NumericVector res(timer);   // 
  return res;
}
