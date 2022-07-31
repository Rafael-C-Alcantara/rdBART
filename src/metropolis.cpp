#include "tree.h"
#include "treeProposal.h"
#include "utilities.h"
#include "metropolis.h"

// Calculate posterior covariance matrix
// [[Rcpp::export]]
arma::mat lambdaPosteriorOne(const arma::mat& Xnode, int m, double sigma)
{
  int k = Xnode.n_cols;
  arma::mat priorInv = arma::eye(k,k);
  priorInv.diag() *= m;
  arma::mat XtX = Xnode.t()*Xnode/sigma;
  arma::mat out = arma::inv(priorInv+XtX);
  return(out);
}

// [[Rcpp::export]]
Rcpp::List lambdaPosterior(Rcpp::List tree, const arma::mat& W, const arma::mat& X, int m, double sigma)
{
  arma::vec yfit = fit(tree,W);
  arma::vec bn = bottomNodes(tree);
  Rcpp::List splitList = splitMatrix(X,yfit,bn);
  Rcpp::List out = Rcpp::clone(splitList); // So I don't have to create empty tree and push_back (less efficient)
  for (int i=0; i<out.length(); i++)
    {
      arma::mat Xi = splitList[i];
      out[i] = lambdaPosteriorOne(Xi,m,sigma);
    }
  return out;
}

// Calculate posterior mean vector
// [[Rcpp::export]]
arma::vec thetaPosteriorOne(const arma::mat& X, const arma::vec& y, const arma::mat& Lambda, double sigma)
{
  return Lambda*X.t()*y/sigma;
}

// [[Rcpp::export]]
Rcpp::List thetaPosterior(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& y, Rcpp::List Lambda, double sigma)
{
  arma::vec yfit = fit(tree,W);
  arma::vec bn = bottomNodes(tree);
  Rcpp::List splitListX = splitMatrix(X,yfit,bn);
  Rcpp::List splitListY = splitMatrix(y,yfit,bn);
  Rcpp::List out = Rcpp::clone(splitListX); // So I don't have to create empty list and push_back (less efficient)
  for (int i=0; i<out.length(); i++)
    {
      arma::mat Xi = splitListX[i];
      arma::vec Yi = splitListY[i];
      arma::mat Lambdai = Lambda[i];
      out[i] = thetaPosteriorOne(Xi,Yi,Lambdai,sigma);
    }
  return out;
}

// Calculate tree log-likelihood
//[[Rcpp::export]]
double logLikTree(Rcpp::List tree, const arma::mat& W, const arma::mat& X, const arma::vec& y, int m, double sigma, Rcpp::List theta, Rcpp::List Lambda)
{
  int k = X.n_cols;
  double logm = std::log(m)*k;
  arma::vec yfit = fit(tree,W);
  arma::vec bn = bottomNodes(tree);
  Rcpp::List splitListX = splitMatrix(X,yfit,bn);
  Rcpp::List splitListY = splitMatrix(y,yfit,bn);
  double a = 0.0;
  double b = 0.0;
  for (int i=0; i<splitListY.length(); i++)
    {
      arma::mat Xi = splitListX[i];
      arma::vec Yi = splitListY[i];
      arma::vec thetai = theta[i];
      arma::mat Lambdai = Lambda[i];
      arma::mat linv = arma::inv(Lambdai);
      double tLt = (thetai.t()*linv*thetai).eval()(0,0);
      double expTerm = arma::dot(Yi,Yi)/sigma - tLt;
      a += expTerm;
      double det;
      double sign;
      bool ok = arma::log_det(det,sign,Lambdai);
      b += det + logm;
    }
  a *= -0.5;
  b *= 0.5;
  return(a+b);
}

// Calculate MH ratio
// [[Rcpp::export]]
double mhRatio(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, const arma::mat& X, const arma::vec y, int m, double sigma, double alpha, double beta, double ll0)
{
  int move = treeNew["move"];
  Rcpp::List tree1 = treeNew["tree"];
  Rcpp::List Lambda1 = lambdaPosterior(tree1,W,X,m,sigma);
  Rcpp::List theta1 = thetaPosterior(tree1,W,X,y,Lambda1,sigma);
  double ll1 = logLikTree(tree1,W,X,y,m,sigma,theta1,Lambda1);
  if (move==1)
    {
      int node = treeNew["node"];
      double p = pSplit(node,alpha,beta);
      int bot = nBottomNodes(tree);
      int nog = nNogNodes(tree1);
      return std::log(bot)-std::log(nog)+std::log(p)-std::log(1-p)+ll1-ll0;
    } else if (move == 2)
    {
      int node = treeNew["node"];
      double p = pSplit(node,alpha,beta);
      int bot = nBottomNodes(tree1);
      int nog = nNogNodes(tree);
      return std::log(nog)-std::log(bot)+std::log(1-p)-std::log(p)+ll1-ll0;
    } else if (move == 3)
    {
      Rcpp::List s = splits(W);
      Rcpp::List s1 = Rcpp::clone(s);
      double p0 = pTree(tree,s,alpha,beta);
      double p1 = pTree(tree1,s1,alpha,beta);
      return ll1-ll0+std::log(p1)-std::log(p0);
    } else
    {
      return ll1-ll0;
    }
}

// Sample new tree
// [[Rcpp::export]]
Rcpp::List newTree(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, const arma::mat& X, const arma::vec y, int m, double sigma, double alpha, double beta, double ll0)
{
  double prob  = R::runif(0.0,1.0);
  double ratio = mhRatio(tree,treeNew,W,X,y,m,sigma,alpha,beta,ll0);
  bool acc = std::log(prob) <= ratio;
  Rcpp::List out = Rcpp::List::create(Rcpp::_["tree"] = Rcpp::List::create(), Rcpp::_["Acc"] = NA_LOGICAL);
  if (acc)
    {
      Rcpp::List tree1 = treeNew["tree"];
      out["tree"] = tree1;
      out["Acc"]  = acc;
    } else
    {
      out["tree"] = tree;
      out["Acc"]  = acc;
    }
  return(out);
}
