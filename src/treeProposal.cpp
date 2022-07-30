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

  // Select bottom node
  arma::vec bn = bottomNodes(treeNew);
  Rcpp::IntegerVector bnR = Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(bn));
  int bot = bnR[0];
  
  // Select variable
  Rcpp::List s = avalSplits(treeNew, splits);
  int nvar = s.length();;
  Rcpp::IntegerVector seqVar = Rcpp::seq(1,nvar);
  Rcpp::IntegerVector var = Rcpp::sample(seqVar, 1);
  int outVar = var[0];

  // Select value
  Rcpp::List svar = s[outVar-1];
  int nval = svar.length();
  Rcpp::IntegerVector seqVal = Rcpp::seq(1,nval);
  Rcpp::IntegerVector val = Rcpp::sample(seqVal, 1);
  int outVal = val[0];

  // Grow tree
  splitNode(treeNew,bot,outVar,outVal);
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("tree") = treeNew, Rcpp::Named("node") = bot);
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

  // Select node
  arma::vec nn  = nogNodes(treeNew);
  Rcpp::IntegerVector nnR = Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(nn));
  int nog = nnR[0];

  // Prune tree
  pruneNode(treeNew,nog);
  Rcpp::List output = Rcpp::List::create(Rcpp::Named("tree") = treeNew, Rcpp::Named("node") = nog);
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
  arma::vec nodes  = internalNodes(treeNew);
  Rcpp::IntegerVector nodesR = Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(nodes));
  int node = nodesR[0];
  
  // Sample variable to choose for split
  Rcpp::List s = avalSplits(treeNew, splits);
  int nvar     = s.length();
  Rcpp::IntegerVector seqVar = Rcpp::seq(1,nvar);
  Rcpp::IntegerVector var = Rcpp::sample(seqVar, 1);
  int outVar = var[0];

  // Sample value for split
  Rcpp::List svar = s[outVar-1];
  int nval = svar.length();
  Rcpp::IntegerVector seqVal = Rcpp::seq(1,nval);
  Rcpp::IntegerVector val = Rcpp::sample(seqVal, 1);
  int outVal = val[0];

  // Change tree
  changeNode(treeNew,node,outVar,outVal);
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
arma::vec fit(Rcpp::List tree, const arma::mat& W)
{
  int nobs = W.n_rows;
  arma::vec out(nobs);
  for (int i=0; i<nobs; i++)
    {
      arma::rowvec w = W.row(i);
      out(i) = fitObs(tree,w,W);
    }
  return(out);
}

int proposalMove(Rcpp::List tree, const arma::mat& W)
{
  // Function to sample valid moves for proposal
  arma::vec fitVec = fit(tree,W);
  Rcpp::IntegerVector rcppFit = Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(fitVec));
  Rcpp::IntegerVector fitTab = Rcpp::table(rcppFit);
  int minFit = Rcpp::min(fitTab);
  int nbot = nBottomNodes(tree);
  int nint = nInternalNodes(tree);
  int npc  = nPCNodes(tree);

  // Setting probabilities of each move
  Rcpp::NumericVector probs = {.25,.25,.4,.1};

  // Sample move
  Rcpp::IntegerVector moveVec = Rcpp::sample(4,1,probs);
  int move = moveVec[0];

  // List invalid moves: 1: not grow when nbot=1; 2: change or swap when there are no internal nodes; 3: swap when there are no pcPair nodes; 4: grow when there are bottom nodes with 1 obs (avoid empty nodes)
  bool invalid = (move > 1 && nbot == 1)|(move > 2 && nint == 0)|(move == 4 && npc == 0)|(move == 1 && minFit == 1);

  // Recursion: if move is invalid, sample again until move is valid
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
