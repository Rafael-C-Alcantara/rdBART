#ifndef GUARD_treeProposal_h
#define GUARD_treeProposal_h
#include <RcppArmadillo.h>

void splitNode(Rcpp::List tree, int nodeID, int var, int val);
Rcpp::List grow(Rcpp::List tree, Rcpp::List splits);
void pruneNode(Rcpp::List tree, int nodeID);
Rcpp::List prune(Rcpp::List tree);
void changeNode(Rcpp::List tree, int nodeID, int var, int val);
Rcpp::List change(Rcpp::List tree, Rcpp::List splits);
void swapNode(Rcpp::List tree, Rcpp::List nodes);
Rcpp::List swap(Rcpp::List tree);
int fitObs(Rcpp::List tree, const arma::vec& w, const arma::mat& W);
arma::vec fit(Rcpp::List tree, const arma::mat& W);
int proposalMove(Rcpp::List tree, const arma::mat& W);
Rcpp::List proposal(Rcpp::List tree, const arma::mat& W);

#endif
