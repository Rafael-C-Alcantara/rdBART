#include "tree.h"

// [[Rcpp::export]]
Rcpp::List node(int nodeID, int splitVar, int splitVal) // Create node
{
  Rcpp::List node = Rcpp::List::create(Rcpp::Named("id") = nodeID, Rcpp::Named("var") = splitVar, Rcpp::Named("val") = splitVal,
				       Rcpp::Named("left") = Rcpp::List::create(), Rcpp::Named("right") = Rcpp::List::create());
  return(node);
}
// [[Rcpp::export]]
Rcpp::List splits(const arma::mat& W)
{
  Rcpp::List s = Rcpp::List::create();
  for (int i=0; i<W.n_cols; i++)
    {
      s.push_back(W.col(i),std::to_string(i+1));
    }
  return(s);
}

void getAvalSplits(Rcpp::List tree, Rcpp::List& splits)
{
  int var = tree["var"];
  int val = tree["val"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() == 0) return; // Reached bottom node
  Rcpp::NumericVector svar = splits[var-1];
  if (svar.length() == 0) splits.erase(var-1); // No available values for split: remove variable
  svar.erase(val-1);
  splits[var-1] = svar;
  getAvalSplits(left, splits);
  getAvalSplits(right, splits);
}

Rcpp::List avalSplits(Rcpp::List tree, Rcpp::List& splits)
{
  int nvar = splits.length();
  for (int i=0; i<nvar; i++)
    {
      Rcpp::NumericVector svar = splits[i];
      double max = Rcpp::max(svar);
      double min = Rcpp::min(svar);
      for (int j=0; j<svar.length(); j++)
	{
	  // Remove max and min to avoid empty nodes
	  if (svar[j] == max|svar[j] == min) svar.erase(j);
	}
      splits[i] = svar;
    }
  getAvalSplits(tree,splits);
  return(splits);
}

void getBottomNodes(Rcpp::List tree, Rcpp::NumericVector& botVec)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() > 0)
    {
      getBottomNodes(left, botVec);
      getBottomNodes(right, botVec);
    } else
    {
      botVec.push_back(id);
    }
}

Rcpp::NumericVector bottomNodes(Rcpp::List tree)
{
  Rcpp::NumericVector out;
  getBottomNodes(tree,out);
  return(out);
}

int nBottomNodes(Rcpp::List tree)
{
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() > 0) return(nBottomNodes(left)+nBottomNodes(right));
  return(1);
}

void getNogNodes(Rcpp::List tree, Rcpp::NumericVector& nogVec)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() > 0) // Node has children
    {
      Rcpp::List left2  = left["left"];
      Rcpp::List right2 = right["left"];
      if (left2.length() == 0 && right2.length() == 0) nogVec.push_back(id);
      if (left2.length() > 0) getNogNodes(left, nogVec);
      if (right2.length() > 0) getNogNodes(right, nogVec);
    }
}

Rcpp::NumericVector nogNodes(Rcpp::List tree)
{
  Rcpp::NumericVector out;
  getNogNodes(tree,out);
  return(out);
}

int nNogNodes(Rcpp::List tree)
{
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() > 0)
    {
      Rcpp::List left2  = left["left"];
      Rcpp::List right2 = right["left"];
      if (left2.length() > 0 && right2.length() > 0) return(nNogNodes(left)+nNogNodes(right));
      if (left2.length() > 0) return(nNogNodes(left));
      if (right2.length() > 0) return(nNogNodes(right));
      return(1);
    }
  return(0);
}

void getInternalNodes(Rcpp::List tree, Rcpp::NumericVector& intVec)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() > 0)
    {
      if (id > 1)
	{
	  intVec.push_back(id);
	  getInternalNodes(left,intVec);
	  getInternalNodes(right,intVec);
	} else
	{
	  getInternalNodes(left,intVec);
	  getInternalNodes(right,intVec);
	}
    }
}

Rcpp::NumericVector internalNodes(Rcpp::List tree)
{
  Rcpp::NumericVector out;
  getInternalNodes(tree,out);
  return(out);
}

int nInternalNodes(Rcpp::List tree)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() == 0) return(0);
  if (id == 1) return(nInternalNodes(left) + nInternalNodes(right));
  return(1 + nInternalNodes(left) + nInternalNodes(right));
}

// List and count parent-child node pairs
void getPCNodes(Rcpp::List tree, Rcpp::List& pcList)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() > 0) // Node is not terminal
    {
      if (id > 1) // Node is not root
	{
	  Rcpp::List left2  = left["left"];
	  Rcpp::List right2 = right["left"];
	  if (left2.length() > 0) // Left child is not terminal
	    {
	      int id2 = left["id"];
	      Rcpp::IntegerVector v = {id,id2};
	      pcList.push_back(v);
	    }
	  if (right2.length() > 0) // Right child is not terminal
	    {
	      int id2 = right["id"];
	      Rcpp::IntegerVector v = {id,id2};
	      pcList.push_back(v);
	    }
	}
      getPCNodes(left,pcList);
      getPCNodes(right,pcList);

    }
}

Rcpp::List pcNodes(Rcpp::List tree)
{
  Rcpp::List out;
  getPCNodes(tree,out);
  return(out);
}

int nPCNodes(Rcpp::List tree)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() > 0) // Node is not terminal
    {
      if (id == 1) return(nPCNodes(left)+nPCNodes(right)); // Root node
      Rcpp::List left2  = left["left"];
      Rcpp::List right2 = right["left"];
      if (left2.length() > 0) return(1+nPCNodes(left)+nPCNodes(right)); // Left child is not terminal
      if (right2.length() > 0) return(1+nPCNodes(left)+nPCNodes(right)); // Right child is not terminal
    }
  return(0);
}

int nodeDepth(int nodeID)
{
  // This depends on the structure of the tree being: node i is parent of node 2i and 2i+1
  int val = nodeID/2;
  if (val == 0) return(0);
  return(1 + nodeDepth(val));
}

double pSplit(int nodeID, double alpha, double beta)
{
  int d = nodeDepth(nodeID);
  return(alpha/std::pow(1+d,beta));
}

double pAllSplits(Rcpp::List tree, double alpha, double beta)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() == 0) return(1-pSplit(id,alpha,beta)); // Reached bottom node
  return(pSplit(id,alpha,beta)*pAllSplits(left,alpha,beta)*pAllSplits(right,alpha,beta));
}

double pRules(Rcpp::List tree, Rcpp::List& splits)
{
  int id  = tree["id"];
  int var = tree["var"];
  int val = tree["val"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() == 0) return(1); // Reached bottom node
  double nvar = splits.length();
  Rcpp::NumericVector svar = splits[var-1];
  if (svar.length() == 2) splits.erase(var-1); // If all possible values have been chosen for var, remove it from available splits
  double nval = svar.length() - 2; // Remove max and min values to avoid empty nodes
  double total = nvar*nval;
  svar.erase(val-1);
  splits[var-1] = svar;
  return(pRules(left,splits)*pRules(right,splits)/total);
}

double pTree(Rcpp::List tree, Rcpp::List splits, double alpha, double beta)
{
  return(pAllSplits(tree,alpha,beta)*pRules(tree,splits));
}
