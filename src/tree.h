#ifndef GUARD_tree_h
#define GUARD_tree_h
#include <RcppArmadillo.h>
// Create node
Rcpp::List node(int nodeID, int splitVar, int splitVal);
// Data information
Rcpp::List splits(const arma::mat& W);
void getAvalSplits(Rcpp::List tree, Rcpp::List& splits);
Rcpp::List avalSplits(Rcpp::List tree, Rcpp::List& splits);
// Tree information
// List and count bottom nodes
void getBottomNodes(Rcpp::List tree, std::vector<int>& botVec);
arma::vec bottomNodes(Rcpp::List tree);
int nBottomNodes(Rcpp::List tree);
// List and count no-grandchildren nodes
void getNogNodes(Rcpp::List tree, std::vector<int>& nogVec);
arma::vec nogNodes(Rcpp::List tree);
int nNogNodes(Rcpp::List tree);
// List and count internal nodes
void getInternalNodes(Rcpp::List tree, std::vector<int>& intVec);
arma::vec internalNodes(Rcpp::List tree);
int nInternalNodes(Rcpp::List tree);
// List and count parent-child node pairs
void getPCNodes(Rcpp::List tree, Rcpp::List& pcList);
Rcpp::List pcNodes(Rcpp::List tree);
int nPCNodes(Rcpp::List tree);
// Node depth
int nodeDepth(int nodeID);
// Node split probability
double pSplit(int nodeID, double alpha, double beta);
// Tree Probability
// Probability of all splits
double pAllsplits(Rcpp::List tree, double alpha, double beta);
// Probability of chosen split rules
double pRules(Rcpp::List tree, Rcpp::List& splits);
// Tree probability
double pTree(Rcpp::List tree, Rcpp::List splits, double alpha, double beta);
#endif
