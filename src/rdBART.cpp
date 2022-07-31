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
  arma::vec sigDraws(nDraws,arma::fill::value(1.0));

  // Create list of trees
  Rcpp::List treeList;
  for (int i=0; i<m; i++) treeList.push_back(node(1,NA_INTEGER,NA_INTEGER));

  // Create list of posterior covariance for each tree
  Rcpp::List LambdaPost = Rcpp::clone(treeList);
  for (int i=0; i<m; i++)
    {
      Rcpp::List tree = treeList[i];
      LambdaPost[i]   = lambdaPosterior(tree,W,X,m,sigDraws(0));
    }

  // Create list of posterior mean for each tree
  Rcpp::List thetaPost = Rcpp::clone(treeList);
  for (int i=0; i<m; i++)
    {
      Rcpp::List tree = treeList[i];
      Rcpp::List Li   = LambdaPost[i];
      thetaPost[i]    = thetaPosterior(tree,W,X,y,Li,sigDraws(0));
    }

  // Obtain draws of theta for initial tree list
  Rcpp::List thetaList = Rcpp::clone(treeList);
  for (int i=0; i<m; i++)
    {
      Rcpp::List tree = treeList[i];
      Rcpp::List ti   = thetaPost[i];
      Rcpp::List Li   = LambdaPost[i];
      thetaList[i]    = thetaDraw(tree,W,X,y,m,sigDraws(0),ti,Li);
    }

  // Obtain log-likelihood for intial trees
  arma::vec logLikList(m);
  for (int i=0; i<m; i++)
    {
      Rcpp::List tree = treeList[i];
      Rcpp::List ti   = thetaPost[i];
      Rcpp::List Li   = LambdaPost[i];
      logLikList(i)   = logLikTree(tree,W,X,y,m,sigDraws(0),ti,Li);
    }

  // Obtain matrix of fits
  arma::mat yFitMat(nobs,m);
  for (int i=0; i<m; i++)
    {
      Rcpp::List tree = treeList[i];
      yFitMat.col(i) = fit(tree,W);
    }

  // Obtain list of matrices with draws for each y
  Rcpp::List yDrawsList = Rcpp::clone(treeList);
  for (int i=0; i<m; i++)
    {
      Rcpp::List tree = treeList[i];
      Rcpp::List ti   = thetaList[i];
      yDrawsList[i]   = yDraw(tree,W,X,yFitMat.col(i),ti);
    }

  // Create matrix of predictions for each tree
  arma::mat yPredMat(nobs,m);
  for (int i=0; i<m; i++)
    {
      yPredMat.col(i) = y/m;
    }

  // Create list of posterior draws for each parameter (output)
  arma::vec drawsPostArma(nDraws);
  Rcpp::List drawsPost = Rcpp::as<Rcpp::List>(Rcpp::wrap(drawsPostArma));
  
  // Create matrix to keep track of accept/reject for each tree (output)
  arma::mat accRatio(nDraws,m);

  // Create drawsPost iterator
  Rcpp::List::iterator drawsPostIt = drawsPost.begin();
  
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
	  Rcpp::List oldTree = treeList[j];
	  // Obtain residual from other trees to use as y
	  arma::vec resid = partialResid(y,yPredMat,j);
	  // Obtain tree proposal
	  Rcpp::List treeProp = proposal(oldTree,W);
	  // Select between old and proposal tree
	  Rcpp::List treeNew  = newTree(oldTree,treeProp,W,X,resid,m,sig,alpha,beta,logLikList(j));
	  // If tree changed, update treeList, LambdaPost, thetaPost, logLikList and yFitMat
	  bool acc = treeNew["Acc"];
	  if (acc)
	    {
	      // Update tree list
	      Rcpp::List tn = treeNew["tree"];
	      treeList[j] = tn;
	      // Update LambdaPost
	      LambdaPost[j] = lambdaPosterior(tn,W,X,m,sig);
	      // Update thetaPost
	      Rcpp::List lp = LambdaPost[j];
	      thetaPost[j] = thetaPosterior(tn,W,X,resid,lp,sig);
	      // Update logLikList
	      Rcpp::List tp = thetaPost[j];
	      logLikList(j) = logLikTree(tn,W,X,resid,m,sig,tp,lp);
	      // Update yFitMat
	      yFitMat.col(j) = fit(tn,W);
	    }
	  // Draw new thetas for tree
	  Rcpp::List tn = treeList[j];
	  // std::cout << "Passed tn\n";
	  Rcpp::List tp = thetaPost[j];
	  // std::cout << "Passed tp\n";
	  Rcpp::List lp = LambdaPost[j];
	  // std::cout << "Passed lp\n";
	  thetaList[j] = thetaDraw(tn,W,X,resid,m,sig,tp,lp);
	  // std::cout << "Passed thetaList\n";
	  // Update matrix of individual level thetas
	  Rcpp::List td = thetaList[j];
	  // std::cout << "Passed td\n";
	  yDrawsList[j] = yDraw(tn,W,X,yFitMat.col(j),td);
	  // std::cout << "Passed yDrawsList\n";
	  // Update matrix of predictions
	  arma::mat yd    = yDrawsList[j];
	  // std::cout << "Passed yd\n";
	  yPredMat.col(j) = yPred(tn,W,X,yFitMat.col(j),yd);
	  // std::cout << "Passed yPredMat\n";
	  // Update sum of tree Draws
	  drawsPostLoop += yd;
	  // std::cout << "Passed drawsPostLoop\n";
	  // Update accept/reject matrix
	  accRatio(i,j) = acc;
	  // std::cout << "Passed accRatio\n";
	}
      // std::cout << "Test\n";
      // std::cout << i << '\n';
      drawsPost[i] = drawsPostLoop;
      // std::cout << "Passed drawsPost.push_back\n";
      // Update sigma
      sigDraws(i) = sigPost(nu,lambda,m,nobs,yPredMat,y);
      // std::cout << "Passed sigPost\n";
    }
  // Output list of created objects for test
  Rcpp::List out = Rcpp::List::create(Rcpp::_("draws") = drawsPost,
				      Rcpp::_("sigma") = sigDraws,
				      Rcpp::_("accRatio") = accRatio);
  return out;
}
