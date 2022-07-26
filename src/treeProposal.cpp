#include "tree.h"
#include "treeProposal.h"

// Grow
void splitNode(Rcpp::List tree, int nodeID, int var, int val)
{
  int id = tree["id"];
  if (id == nodeID)
    {
      // Found node
      tree["var"] = var;
      tree["val"] = val;
      tree["left"] = node(2*id,NA_INTEGER,NA_INTEGER);
      tree["right"] = node(2*id+1,NA_INTEGER,NA_INTEGER);
      return; // Stop looking if already found right node
    }
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() == 0) return; // Reached bottom node but haven't found right node yet
  splitNode(left, nodeID, var, val);
  splitNode(right, nodeID, var, val);
}
// [[Rcpp::export]]
Rcpp::List grow(Rcpp::List tree, Rcpp::List splits)
{
  Rcpp::List treeNew = Rcpp::clone(tree);
  Rcpp::NumericVector bn = bottomNodes(treeNew);
  Rcpp::List s = avalSplits(treeNew, splits);
  Rcpp::NumericVector bot = Rcpp::sample(bn, 1);
  int outBot = bot[0];
  int nvar = s.length();;
  Rcpp::IntegerVector seqVar = Rcpp::seq(1,nvar);
  Rcpp::IntegerVector var = Rcpp::sample(seqVar, 1);
  int outVar = var[0];
  Rcpp::List svar = s[outVar-1];
  int nval = svar.length();
  Rcpp::IntegerVector seqVal = Rcpp::seq(1,nval);
  Rcpp::IntegerVector val = Rcpp::sample(seqVal, 1);
  int outVal = val[0];
  splitNode(treeNew,outBot,outVar,outVal);
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("tree") = treeNew, Rcpp::Named("node") = outBot);
  return(output);
}

// Prune
void pruneNode(Rcpp::List tree, int nodeID)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() == 0) return; // Reached bottom node but haven't found right node yet
  if (id == nodeID)
    {
      // Found the right node
      tree["var"]   = NA_INTEGER;
      tree["val"]   = NA_INTEGER;
      tree["left"]  = Rcpp::List::create();
      tree["right"] = Rcpp::List::create();
    }
  pruneNode(left, nodeID);
  pruneNode(right, nodeID);
}
// [[Rcpp::export]]
Rcpp::List prune(Rcpp::List tree)
{
  Rcpp::List treeNew = Rcpp::clone(tree);
  Rcpp::NumericVector nn  = nogNodes(treeNew);
  Rcpp::NumericVector nog = Rcpp::sample(nn,1);
  int outBot = nog[0];
  pruneNode(treeNew,outBot);
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("tree") = treeNew, Rcpp::Named("node") = outBot);
  return(output);
}

// Change
void changeNode(Rcpp::List tree, int nodeID, int var, int val)
{
  int id = tree["id"];
  Rcpp::List left = tree["left"];
  Rcpp::List right = tree["right"];
  if (id == nodeID)
    {
      // Found right node
      tree["var"] = var;
      tree["val"] = val;
    }
  if (left.length() == 0) return; // Reached bottom node bu haven't found right node yet
  changeNode(left, nodeID, var, val);
  changeNode(right, nodeID, var, val);
}
// [[Rcpp::export]]
Rcpp::List change(Rcpp::List tree, Rcpp::List splits)
{
  // Sample internal node to change
  Rcpp::List treeNew = Rcpp::clone(tree);
  Rcpp::NumericVector nodes  = internalNodes(treeNew);
  Rcpp::NumericVector intNod = Rcpp::sample(nodes,1);
  int nod = intNod[0];
  // Sample variable to choose for split
  Rcpp::List s = avalSplits(treeNew, splits);
  int nvar     = s.length();
  Rcpp::IntegerVector seqVar = Rcpp::seq(1,nvar);
  Rcpp::IntegerVector var = Rcpp::sample(seqVar, 1);
  int outVar = var[0];
  Rcpp::List svar = s[outVar-1];
  int nval = svar.length();
  Rcpp::IntegerVector seqVal = Rcpp::seq(1,nval);
  Rcpp::IntegerVector val = Rcpp::sample(seqVal, 1);
  int outVal = val[0];
  changeNode(treeNew,nod,outVar,outVal);
  return(treeNew);
}

// Swap
void swapNode(Rcpp::List tree, Rcpp::List nodes)
{
  int id = tree["id"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (id == nodes[0])
    {
      // Found parent node
      int parentVar = tree["var"];
      int parentVal = tree["val"];
      int lID = left["id"];
      int rID = right["id"];
      if (lID == nodes[1])
	{
	  int childVar = left["var"];
	  int childVal = left["val"];
	  left["var"]  = parentVar;
	  left["val"]  = parentVal;
	  tree["var"] = childVar;
	  tree["val"] = childVal;
	}
      if (rID == nodes[1])
	{
	  int childVar = right["var"];
	  int childVal = right["val"];
	  right["var"]  = parentVar;
	  right["val"]  = parentVal;
	  tree["var"] = childVar;
	  tree["val"] = childVal;
	}
    }
  if (left.length() == 0) return; // Reached bottom node but haven't found right node yet
  swapNode(left, nodes);
  swapNode(right, nodes);
}
// [[Rcpp::export]]
Rcpp::List swap(Rcpp::List tree)
{
  Rcpp::List treeNew = Rcpp::clone(tree);
  Rcpp::List nodes = pcNodes(treeNew);
  Rcpp::List pair = Rcpp::sample(nodes,1);
  Rcpp::List outPair = pair[0];
  swapNode(treeNew,outPair);
  return(treeNew);
}

// Proposal distribution
// Fit tree: will be used to make sure we don't create trees with empty bottom nodes
int fitObs(Rcpp::List tree, const arma::rowvec& w, const arma::mat& W)
{
  int id  = tree["id"];
  arma::uword var = tree["var"];
  arma::uword val = tree["val"];
  Rcpp::List left  = tree["left"];
  Rcpp::List right = tree["right"];
  if (left.length() == 0) return(id); // Reached bottom node
  double wi = w(var-1);
  double Wi = W(val-1,var-1);
  if (wi <= Wi)
    {
      return(fitObs(left,w,W));
    } else
    {
      return(fitObs(right,w,W));
    }
}
// [[Rcpp::export]]
Rcpp::IntegerVector fit(Rcpp::List tree, const arma::mat& W)
{
  int nobs = W.n_rows;
  Rcpp::IntegerVector out(nobs);
  for (int i=0; i<nobs; i++)
    {
      arma::rowvec w = W.row(i);
      out[i] = fitObs(tree,w,W);
    }
  return(out);
}

int proposalMove(Rcpp::List tree, const arma::mat& W)
{
  // Function to sample valid moves for proposal
  Rcpp::IntegerVector fitVec = fit(tree,W);
  Rcpp::IntegerVector fitTab = Rcpp::table(fitVec);
  int minFit = Rcpp::min(fitTab);
  int nbot = nBottomNodes(tree);
  int nint = nInternalNodes(tree);
  int npc  = nPCNodes(tree);
  Rcpp::NumericVector probs = {.25,.25,.4,.1};
  Rcpp::IntegerVector moveVec = Rcpp::sample(4,1,probs);
  int move = moveVec[0];
  bool invalid = (move > 1 && nbot == 1)|(move > 2 && nint == 0)|(move == 4 && npc == 0)|(move == 1 && minFit == 1);
  if (invalid) return(proposalMove(tree,W));
  return(move);
}

// [[Rcpp::export]]
Rcpp::List proposal(Rcpp::List tree, const arma::mat& W)
{
  int move = proposalMove(tree,W);
  if (move == 1)
    {
      Rcpp::List s = splits(W);
      Rcpp::List treeNew = grow(tree,s);
      treeNew.push_back(move,"move");
      return(treeNew);
    } else if (move == 2)
    {
      Rcpp::List treeNew = prune(tree);
      treeNew.push_back(move,"move");
      return(treeNew);
    } else if (move == 3)
    {
      Rcpp::List s = splits(W);
      Rcpp::List treeNew = change(tree,s);
      Rcpp::List output = Rcpp::List::create(Rcpp::Named("tree") = treeNew, Rcpp::Named("move") = move);
      return(output);
    } else
    {
      Rcpp::List treeNew = swap(tree);
      Rcpp::List output = Rcpp::List::create(Rcpp::Named("tree") = treeNew, Rcpp::Named("move") = move);
      return(output);
    }
}
