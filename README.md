# MAB

R package MAB is created to
implement strategies for stationary and non-stationary multi-armed bandit problems. 
Various widely-used strategies and their ensembles are included in this package.
This package is designed to compare different strategies in multi-armed bandit problems
and help users to choose suitable strategies with suitable tuning parameters
in different scenarios. This is not an official Google product. 


Install the Package
-------------------

MAB depends on R package [emre](https://github.com/google/emre). To install MAB package, download the package and run the following code in 
command line:

R CMD INSTALL FILE.PATH

Another way is to install devtools package first and then run the following code
in R:

library(devtools)

install_github("google/MAB")


