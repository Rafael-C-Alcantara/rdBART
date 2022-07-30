//#include <RcppArmadillo.h>
#include "tree.h"
#include "treeProposal.h"
#include "utilities.h"
#include "metropolis.h"
#include "gibbs.h"
//#include <Rcpp/Benchmark/Timer.h>

// [[Rcpp::export]]
Rcpp::List rdBART(const arma::mat& W, const arma::mat& X, const arma::vec& y, double alpha, double beta, double nu, double lambda, int m, int nDraws, int burn)
{
  // Get X dimensions
  int nobs = X.n_rows;
  int npar = X.n_cols;

  // Create vector to store sigma draws
  arma::vec sigDraws(nDraws,arma::fill::value(10.0));

  // Create list of trees
  Rcpp::List treeList;
  for (int i=0; i<m; i++) treeList.push_back(node(1,NA_INTEGER,NA_INTEGER));

  // Create list of posterior covariance for each tree
  Rcpp::List LambdaPost;
  for (int i=0; i<m; i++) LambdaPost.push_back(lambdaPosterior(treeList[i],W,X,m,sigDraws(0)));

  // Create list of posterior mean for each tree
  Rcpp::List thetaPost;
  for (int i=0; i<m; i++) thetaPost.push_back(thetaPosterior(treeList[i],W,X,y,LambdaPost[i],sigDraws(0)));

  // Obtain draws of theta for initial tree list
  Rcpp::List thetaList;
  for (int i=0; i<m; i++) thetaList.push_back(thetaDraw(treeList[i],W,X,y,m,sigDraws(0), thetaPost[i], LambdaPost[i]));

  // Obtain log-likelihood for intial trees
  arma::vec logLikList(m);
  for (int i=0; i<m; i++) logLikList(i) = logLikTree(treeList[i],W,X,y,m,sigDraws(0),thetaPost[i],LambdaPost[i]);

  // Obtain matrix of fits
  arma::mat yFitMat(nobs,m);
  for (int i=0; i<m; i++) yFitMat.col(i) = fit(treeList[i],W);

  // Obtain list of matrices with draws for each y
  Rcpp::List yDrawsList;
  for (int i=0; i<m; i++) yDrawsList.push_back(yDraw(treeList[i],W,X,yFitMat.col(i),thetaList[i]));

  // Create matrix of predictions for each tree
  arma::mat yPredMat(nobs,m);
  for (int i=0; i<m; i++) yPredMat.col(i) = yPred(treeList[i],W,X,yFitMat.col(i),yDrawsList[i]);

  // Create list of posterior draws for each parameter (output)
  Rcpp::List drawsPost;

  // Create matrix to keep track of accept/reject for each tree (output)
  arma::mat accRatio(nDraws,m);

  // Start loop
  for (int i=0; i<nDraws; i++)
    {
      std::cout << "Iteration " << i+1 << " of MCMC\n";
      // Take sigma draw i
      double sig = sigDraws(i);
      // Create matrix to store sum of theta draws
      arma::mat drawsPostLoop(nobs,npar);

      // Start inner loop (tree list loop)
      for (int j=0; j<m; j++)
	{
	  // Obtain residual from other trees to use as y
	  arma::vec resid = partialResid(y,yPredMat,j);
	  // Obtain tree proposal
	  Rcpp::List treeProp = proposal(treeList[j],W);
	  // DEBUG
	  // if (i==523 && j==158)
	  //   {
	  //     Rcpp::List tn = treeProp["tree"];
	  //     Rcpp::List s = splits(W);
	  //     Rcpp::List as = avalSplits(tn,s);
	  //     std::cout << as.length() << '\n';
	  //     for (int k=0; k<as.length(); k++)
	  // 	{
	  // 	  Rcpp::NumericVector ass = as[k];
	  // 	  std::cout << ass.length() << '\n';
	  // 	}
	  //   }
	  // DEBUG
	  // Select between old and proposal tree
	  Rcpp::List tOld = treeList[j];
	  Rcpp::List treeNew  = newTree(tOld,treeProp,W,X,resid,m,sig,alpha,beta,logLikList(j));
	  // If tree changed, update treeList, LambdaPost, thetaPost, logLikList and yFitMat
	  bool acc = treeNew["Acc"];
	  if (acc)
	    {
	      treeList[j]   = treeNew["tree"];
	      LambdaPost[j] = lambdaPosterior(treeList[j],W,X,m,sig);
	      thetaPost[j]  = thetaPosterior(treeList[j],W,X,resid,LambdaPost[j],sig);
	      logLikList(j) = logLikTree(treeList[j],W,X,resid,m,sig,thetaPost[j],LambdaPost[j]);
	      yFitMat.col(j) = fit(treeList[j],W);
	    }
	  // Draw new thetas for tree
	  thetaList[j] = thetaDraw(treeList[j],W,X,resid,m,sig,thetaPost[j],LambdaPost[j]);
	  // Update matrix of individual level thetas
	  yDrawsList[j] = yDraw(treeList[j],W,X,yFitMat.col(j),thetaList[j]);
	  // Update matrix of predictions
	  yPredMat.col(j) = yPred(treeList[j],W,X,yFitMat.col(j),yDrawsList[j]);
	  // Update sum of tree Draws
	  arma::mat drawsPostLoopMat = yDrawsList[j];
	  drawsPostLoop += drawsPostLoopMat;
	  // Update accept/reject matrix
	  accRatio(i,j) = acc;
	}
      // Push back drawsPostLoop to drawsPost
      drawsPost.push_back(drawsPostLoop);
      // Update sigma
      sigDraws(i) = sigPost(nu,lambda,m,nobs,yPredMat,y);
    }
  // Output list of created objects for test
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("draws") = drawsPost,
				      Rcpp::Named("sigma") = sigDraws,
				      Rcpp::Named("accRatio") = accRatio);
  return out;
}
