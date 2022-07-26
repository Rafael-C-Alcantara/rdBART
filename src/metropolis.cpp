#include "tree.h"
#include "treeProposal.h"
#include "utilities.h"
#include "metropolis.h"

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
Rcpp::List lambdaPosterior(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, int m, double sigma)
{
  Rcpp::List out;
  Rcpp::IntegerVector yfit = fit(tree,W);
  arma::vec armaYfit = Rcpp::as<arma::vec>(Rcpp::wrap(yfit));
  Rcpp::NumericVector bn = bottomNodes(tree);
  Rcpp::List splitList = splitMatrix(X,armaYfit,bn);
  for (int i=0; i<splitList.length(); i++)
    {
      arma::mat armaX = Rcpp::as<arma::mat>(Rcpp::wrap(splitList[i]));
      out.push_back(lambdaPosteriorOne(armaX,m,sigma));
    }
  return out;
}
// [[Rcpp::export]]
arma::vec thetaPosteriorOne(const arma::mat& X, const arma::vec& y, const arma::mat& Lambda, double sigma)
{
  return Lambda*X.t()*y/sigma;
}

// [[Rcpp::export]]
Rcpp::List thetaPosterior(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, Rcpp::List Lambda, double sigma)
{
  Rcpp::List out;
  Rcpp::NumericMatrix rcppY = Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(y));
  Rcpp::IntegerVector yfit = fit(tree,W);
  arma::vec armaYfit = Rcpp::as<arma::vec>(Rcpp::wrap(yfit));
  Rcpp::NumericVector bn = bottomNodes(tree);
  Rcpp::List splitListX = splitMatrix(X,armaYfit,bn);
  Rcpp::List splitListY = splitMatrix(rcppY,armaYfit,bn);
  for (int i=0; i<splitListX.length(); i++)
    {
      Rcpp::NumericMatrix Xi = splitListX[i];
      Rcpp::NumericMatrix Yi = splitListY[i];
      arma::mat armaX = Rcpp::as<arma::mat>(Rcpp::wrap(Xi));
      arma::vec armaY = Rcpp::as<arma::vec>(Rcpp::wrap(Yi));
      arma::mat Lambdai = Lambda[i];
      out.push_back(thetaPosteriorOne(armaX,armaY,Lambdai,sigma));
    }
  return out;
}

//[[Rcpp::export]]
double logLikTree(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, Rcpp::List theta, Rcpp::List Lambda)
{
  int k = X.ncol();
  double logm = std::log(m)*k;
  Rcpp::NumericMatrix rcppY = Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(y));
  Rcpp::IntegerVector yfit = fit(tree,W);
  arma::vec armaYfit = Rcpp::as<arma::vec>(Rcpp::wrap(yfit));
  Rcpp::NumericVector bn = bottomNodes(tree);
  Rcpp::List splitListX = splitMatrix(X,armaYfit,bn);
  Rcpp::List splitListY = splitMatrix(rcppY,armaYfit,bn);
  double a = 0.0;
  double b = 0.0;
  for (int i=0; i<splitListY.length(); i++)
    {
      Rcpp::NumericMatrix Xi = splitListX[i];
      Rcpp::NumericMatrix Yi = splitListY[i];
      arma::mat armaX = Rcpp::as<arma::mat>(Rcpp::wrap(Xi));
      arma::vec armaY = Rcpp::as<arma::vec>(Rcpp::wrap(Yi));
      arma::vec thetai = theta[i];
      arma::mat Lambdai = Lambda[i];
      arma::mat linv = arma::inv(Lambdai);
      double tLt = (thetai.t()*linv*thetai).eval()(0,0);
      double expTerm = arma::dot(armaY,armaY)/sigma - tLt;
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
// [[Rcpp::export]]
double mhRatio(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, double alpha, double beta)
{
  int move = treeNew["move"];
  Rcpp::List tree1 = treeNew["tree"];
  Rcpp::List Lambda0 = lambdaPosterior(tree,W,X,m,sigma);
  Rcpp::List theta0 = thetaPosterior(tree,W,X,y,Lambda0,sigma);
  Rcpp::List Lambda1 = lambdaPosterior(tree1,W,X,m,sigma);
  Rcpp::List theta1 = thetaPosterior(tree1,W,X,y,Lambda1,sigma);
  double ll0 = logLikTree(tree,W,X,y,m,sigma,theta0,Lambda0);
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
    } else
    {
      Rcpp::List s = splits(W);
      double p0 = pTree(tree,s,alpha,beta);
      double p1 = pTree(tree1,s,alpha,beta);
      return ll1-ll0+std::log(p1)-std::log(p0);
    }
}
// [[Rcpp::export]]
Rcpp::List newTree(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, double alpha, double beta)
{
  Rcpp::NumericVector prob  = Rcpp::runif(1);
  double lp    = std::log(prob[0]);
  double ratio = mhRatio(tree,treeNew,W,X,y,m,sigma,alpha,beta);
  bool acc = lp <= ratio;
  Rcpp::List out;
  if (acc)
    {
      Rcpp::List tree1 = treeNew["tree"];
      out.push_back(tree1,"tree");
      out.push_back(acc, "Acc");
    } else
    {
      out.push_back(tree,"tree");
      out.push_back(acc, "Acc");
    }
  return(out);
}
