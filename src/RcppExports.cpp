// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// thetaDraw
Rcpp::List thetaDraw(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma);
RcppExport SEXP _rdBART_thetaDraw(SEXP treeSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP mSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(thetaDraw(tree, W, X, y, m, sigma));
    return rcpp_result_gen;
END_RCPP
}
// yDraw
arma::mat yDraw(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma, Rcpp::List draws);
RcppExport SEXP _rdBART_yDraw(SEXP treeSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP mSEXP, SEXP sigmaSEXP, SEXP drawsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type draws(drawsSEXP);
    rcpp_result_gen = Rcpp::wrap(yDraw(tree, W, X, y, m, sigma, draws));
    return rcpp_result_gen;
END_RCPP
}
// thetaDrawsList
Rcpp::List thetaDrawsList(Rcpp::List treeList, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, int m, double sigma);
RcppExport SEXP _rdBART_thetaDrawsList(SEXP treeListSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP mSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type treeList(treeListSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(thetaDrawsList(treeList, W, X, y, m, sigma));
    return rcpp_result_gen;
END_RCPP
}
// partialResid
arma::vec partialResid(Rcpp::List treeList, int remove, const arma::mat& W, Rcpp::NumericMatrix X, const Rcpp::NumericVector& y, int m, double sigma, Rcpp::List draws);
RcppExport SEXP _rdBART_partialResid(SEXP treeListSEXP, SEXP removeSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP mSEXP, SEXP sigmaSEXP, SEXP drawsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type treeList(treeListSEXP);
    Rcpp::traits::input_parameter< int >::type remove(removeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type draws(drawsSEXP);
    rcpp_result_gen = Rcpp::wrap(partialResid(treeList, remove, W, X, y, m, sigma, draws));
    return rcpp_result_gen;
END_RCPP
}
// lambdaPosteriorOne
arma::mat lambdaPosteriorOne(const arma::mat& Xnode, int m, double sigma);
RcppExport SEXP _rdBART_lambdaPosteriorOne(SEXP XnodeSEXP, SEXP mSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Xnode(XnodeSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(lambdaPosteriorOne(Xnode, m, sigma));
    return rcpp_result_gen;
END_RCPP
}
// lambdaPosterior
Rcpp::List lambdaPosterior(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, int m, double sigma);
RcppExport SEXP _rdBART_lambdaPosterior(SEXP treeSEXP, SEXP WSEXP, SEXP XSEXP, SEXP mSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(lambdaPosterior(tree, W, X, m, sigma));
    return rcpp_result_gen;
END_RCPP
}
// thetaPosteriorOne
arma::vec thetaPosteriorOne(const arma::mat& X, const arma::vec& y, const arma::mat& Lambda, double sigma);
RcppExport SEXP _rdBART_thetaPosteriorOne(SEXP XSEXP, SEXP ySEXP, SEXP LambdaSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Lambda(LambdaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(thetaPosteriorOne(X, y, Lambda, sigma));
    return rcpp_result_gen;
END_RCPP
}
// thetaPosterior
Rcpp::List thetaPosterior(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, Rcpp::List Lambda, double sigma);
RcppExport SEXP _rdBART_thetaPosterior(SEXP treeSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP LambdaSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type Lambda(LambdaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(thetaPosterior(tree, W, X, y, Lambda, sigma));
    return rcpp_result_gen;
END_RCPP
}
// logLikTree
double logLikTree(Rcpp::List tree, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, Rcpp::List theta, Rcpp::List Lambda);
RcppExport SEXP _rdBART_logLikTree(SEXP treeSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP mSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP LambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type Lambda(LambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(logLikTree(tree, W, X, y, m, sigma, theta, Lambda));
    return rcpp_result_gen;
END_RCPP
}
// mhRatio
double mhRatio(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, double alpha, double beta);
RcppExport SEXP _rdBART_mhRatio(SEXP treeSEXP, SEXP treeNewSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP mSEXP, SEXP sigmaSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type treeNew(treeNewSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(mhRatio(tree, treeNew, W, X, y, m, sigma, alpha, beta));
    return rcpp_result_gen;
END_RCPP
}
// newTree
Rcpp::List newTree(Rcpp::List tree, Rcpp::List treeNew, const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec y, int m, double sigma, double alpha, double beta);
RcppExport SEXP _rdBART_newTree(SEXP treeSEXP, SEXP treeNewSEXP, SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP mSEXP, SEXP sigmaSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type treeNew(treeNewSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(newTree(tree, treeNew, W, X, y, m, sigma, alpha, beta));
    return rcpp_result_gen;
END_RCPP
}
// rdBART
Rcpp::NumericVector rdBART(const arma::mat& W, Rcpp::NumericMatrix X, const arma::vec& y, double alpha, double beta, int m, int nDraws, int burn);
RcppExport SEXP _rdBART_rdBART(SEXP WSEXP, SEXP XSEXP, SEXP ySEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP mSEXP, SEXP nDrawsSEXP, SEXP burnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type nDraws(nDrawsSEXP);
    Rcpp::traits::input_parameter< int >::type burn(burnSEXP);
    rcpp_result_gen = Rcpp::wrap(rdBART(W, X, y, alpha, beta, m, nDraws, burn));
    return rcpp_result_gen;
END_RCPP
}
// node
Rcpp::List node(int nodeID, int splitVar, int splitVal);
RcppExport SEXP _rdBART_node(SEXP nodeIDSEXP, SEXP splitVarSEXP, SEXP splitValSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type nodeID(nodeIDSEXP);
    Rcpp::traits::input_parameter< int >::type splitVar(splitVarSEXP);
    Rcpp::traits::input_parameter< int >::type splitVal(splitValSEXP);
    rcpp_result_gen = Rcpp::wrap(node(nodeID, splitVar, splitVal));
    return rcpp_result_gen;
END_RCPP
}
// splits
Rcpp::List splits(const arma::mat& W);
RcppExport SEXP _rdBART_splits(SEXP WSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    rcpp_result_gen = Rcpp::wrap(splits(W));
    return rcpp_result_gen;
END_RCPP
}
// grow
Rcpp::List grow(Rcpp::List tree, Rcpp::List splits);
RcppExport SEXP _rdBART_grow(SEXP treeSEXP, SEXP splitsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type splits(splitsSEXP);
    rcpp_result_gen = Rcpp::wrap(grow(tree, splits));
    return rcpp_result_gen;
END_RCPP
}
// prune
Rcpp::List prune(Rcpp::List tree);
RcppExport SEXP _rdBART_prune(SEXP treeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    rcpp_result_gen = Rcpp::wrap(prune(tree));
    return rcpp_result_gen;
END_RCPP
}
// change
Rcpp::List change(Rcpp::List tree, Rcpp::List splits);
RcppExport SEXP _rdBART_change(SEXP treeSEXP, SEXP splitsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type splits(splitsSEXP);
    rcpp_result_gen = Rcpp::wrap(change(tree, splits));
    return rcpp_result_gen;
END_RCPP
}
// swap
Rcpp::List swap(Rcpp::List tree);
RcppExport SEXP _rdBART_swap(SEXP treeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    rcpp_result_gen = Rcpp::wrap(swap(tree));
    return rcpp_result_gen;
END_RCPP
}
// fit
Rcpp::IntegerVector fit(Rcpp::List tree, const arma::mat& W);
RcppExport SEXP _rdBART_fit(SEXP treeSEXP, SEXP WSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    rcpp_result_gen = Rcpp::wrap(fit(tree, W));
    return rcpp_result_gen;
END_RCPP
}
// proposal
Rcpp::List proposal(Rcpp::List tree, const arma::mat& W);
RcppExport SEXP _rdBART_proposal(SEXP treeSEXP, SEXP WSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type W(WSEXP);
    rcpp_result_gen = Rcpp::wrap(proposal(tree, W));
    return rcpp_result_gen;
END_RCPP
}
// whichCpp
Rcpp::NumericVector whichCpp(const arma::vec& vec, double value);
RcppExport SEXP _rdBART_whichCpp(SEXP vecSEXP, SEXP valueSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type vec(vecSEXP);
    Rcpp::traits::input_parameter< double >::type value(valueSEXP);
    rcpp_result_gen = Rcpp::wrap(whichCpp(vec, value));
    return rcpp_result_gen;
END_RCPP
}
// whichList
Rcpp::List whichList(const arma::vec& vec, Rcpp::NumericVector values);
RcppExport SEXP _rdBART_whichList(SEXP vecSEXP, SEXP valuesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type vec(vecSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type values(valuesSEXP);
    rcpp_result_gen = Rcpp::wrap(whichList(vec, values));
    return rcpp_result_gen;
END_RCPP
}
// splitMatrix
Rcpp::List splitMatrix(Rcpp::NumericMatrix M, const arma::vec& vec, Rcpp::NumericVector values);
RcppExport SEXP _rdBART_splitMatrix(SEXP MSEXP, SEXP vecSEXP, SEXP valuesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type M(MSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type vec(vecSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type values(valuesSEXP);
    rcpp_result_gen = Rcpp::wrap(splitMatrix(M, vec, values));
    return rcpp_result_gen;
END_RCPP
}
// mvrnormArma
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma);
RcppExport SEXP _rdBART_mvrnormArma(SEXP nSEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(mvrnormArma(n, mu, sigma));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rdBART_thetaDraw", (DL_FUNC) &_rdBART_thetaDraw, 6},
    {"_rdBART_yDraw", (DL_FUNC) &_rdBART_yDraw, 7},
    {"_rdBART_thetaDrawsList", (DL_FUNC) &_rdBART_thetaDrawsList, 6},
    {"_rdBART_partialResid", (DL_FUNC) &_rdBART_partialResid, 8},
    {"_rdBART_lambdaPosteriorOne", (DL_FUNC) &_rdBART_lambdaPosteriorOne, 3},
    {"_rdBART_lambdaPosterior", (DL_FUNC) &_rdBART_lambdaPosterior, 5},
    {"_rdBART_thetaPosteriorOne", (DL_FUNC) &_rdBART_thetaPosteriorOne, 4},
    {"_rdBART_thetaPosterior", (DL_FUNC) &_rdBART_thetaPosterior, 6},
    {"_rdBART_logLikTree", (DL_FUNC) &_rdBART_logLikTree, 8},
    {"_rdBART_mhRatio", (DL_FUNC) &_rdBART_mhRatio, 9},
    {"_rdBART_newTree", (DL_FUNC) &_rdBART_newTree, 9},
    {"_rdBART_rdBART", (DL_FUNC) &_rdBART_rdBART, 8},
    {"_rdBART_node", (DL_FUNC) &_rdBART_node, 3},
    {"_rdBART_splits", (DL_FUNC) &_rdBART_splits, 1},
    {"_rdBART_grow", (DL_FUNC) &_rdBART_grow, 2},
    {"_rdBART_prune", (DL_FUNC) &_rdBART_prune, 1},
    {"_rdBART_change", (DL_FUNC) &_rdBART_change, 2},
    {"_rdBART_swap", (DL_FUNC) &_rdBART_swap, 1},
    {"_rdBART_fit", (DL_FUNC) &_rdBART_fit, 2},
    {"_rdBART_proposal", (DL_FUNC) &_rdBART_proposal, 2},
    {"_rdBART_whichCpp", (DL_FUNC) &_rdBART_whichCpp, 2},
    {"_rdBART_whichList", (DL_FUNC) &_rdBART_whichList, 2},
    {"_rdBART_splitMatrix", (DL_FUNC) &_rdBART_splitMatrix, 3},
    {"_rdBART_mvrnormArma", (DL_FUNC) &_rdBART_mvrnormArma, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_rdBART(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
