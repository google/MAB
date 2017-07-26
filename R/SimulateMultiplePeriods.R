# Copyright 2012-2017 Google
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
  #     http://www.apache.org/licenses/LICENSE-2.0
#
  # Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#' Simulate strategies for Multi-Armed Bandit in multiple periods
#'
#' This function is aimed to simulate data to run
#' strategies of Multi-Armed Bandit in a sequence of periods. Weight plot 
#' and regret plot are provided if needed. In each period there could be multiple pulls and each method can only be applied once. The default setting is that in each period
#' there is only 1 pull, corresponding to continuous updating.
#'
#' Various methods have been implemented. "Epsilon-Greedy" and "Epsilon-Decreasing" allocates \eqn{1 - epsilon} traffic to the arm which has the largest average reward and equally distribute the traffic
#' to other arms. For "Epsilon-Greedy" epsilon in \code{method.par} serves as constant exploration rate . For "Epsilon-Decreasing" epsilon in \code{method.par} serves as exploration rate at period 1, 
#' while in period \eqn{t} exploration rate is \eqn{epsilon / t}. See \url{https://en.wikipedia.org/wiki/Multi-armed_bandit#Approximate_solutions} for more details about these strategies.
#'
#' "Thompson-Sampling" refers to Beta-Binomial Thompson Sampling using Beta(1, 1) as a prior. "Bayes-Poisson-TS" refers to Poisson-Gamma Thompson Sampling using a Bayesian Generalized Linear 
#' Mixed Effects Model to compute weights. "Bayes-Poisson-TS", "Greedy-Bayes-Poisson-TS" and "EXP3-Bayes-Poisson-TS" depends on the package "emre" to compute posterior distribution. For algorithm
#' details, see the paper \url{https://arxiv.org/abs/1602.00047}. 
#'
#' UCB (Upper Confidence Bound) is a classical method for Multi-Armed Bandit. For algorithm details, see the paper \url{http://personal.unileoben.ac.at/rortner/Pubs/UCBRev.pdf}. EXP3 is a method which
#' needs to specify exploration rate \code{gamma} and exploitation rate \code{eta}. For algorithm details, see the paper \url{https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf}.
#'
#' Ensemble methods are also implemented. "Greedy-Thompson-Sampling" and "Greedy-Bayes-Poisson-TS" allocate \eqn{1 - epsilon} traffic to the arm corresponding to the largest
#' Thompson sampling weight and allocate \eqn{epsilon} traffic corresponding to Thompson sampling weights. 
#' Instead of using average reward for each period to update weights in "EXP3", "EXP3-Thompson-Sampling" and "EXP3-Bayes-Poisson-TS" use Thompson sampling weights in the updating formula in "EXP3".
#' "HyperTS" is an ensemble by applying Thompson Sampling to selecting the best method in each period based on previous performance. For algorithm details, see the paper
#' \url{http://yxjiang.github.io/paper/RecSys2014-ensemble-bandit.pdf}.
#'
#' To measure the performance. Regret is computed by summing over the products of the number of pulls on one arm at one period and the difference of the mean reward of that arm compared with the largest one. 
#' Relative regret is 
#' computed by dividing the regret of a certain method over the regret of the benchmark method that allocates equal weights to each arm throughout all the periods. 
#'
#'
#' @param method A character string choosing from "Epsilon-Greedy", "Epsilon-Decreasing", "Thompson-Sampling",
#' "EXP3", "UCB", "Bayes-Poisson-TS", "Greedy-Thompson-Sampling", "EXP3-Thompson-Sampling", 
#' "Greedy-Bayes-Poisson-TS", "EXP3-Bayes-Poisson-TS" and "HyperTS". For details of these methods, see below. Default is "Thompson-Sampling".
#' @param method.par A list of parameters needed for different methods: 
#' 
#' \code{epsilon}: A real number between 0 and 1; needed for "Epsilon-Greedy", "Epsilon-Decreasing", "Greedy-Thompson-Sampling" and "Greedy-Bayes-Poisson-TS".
#' 
#' \code{ndraws.TS}: A positive integer specifying the number of random draws from the posterior; 
#' needed for "Thompson-Sampling", "Greedy-Thompson-Sampling" and "EXP3-Thompson-Sampling".  Default is 1000.
#' 
#' \code{EXP3}: A list consisting of two real numbers \code{eta} and \code{gamma}; \eqn{eta > 0} and \eqn{0 <= gamma < 1}; needed for "EXP3", "EXP3-Thompson-Sampling" and "EXP3-Bayes-Poisson-TS".
#' 
#' \code{BP}: A list consisting of three postive integers \code{iter.BP}, \code{ndraws.BP} and \code{interval.BP}; 
#' needed for "Bayes-Poisson-TS", "Greedy-Bayes-Poisson-TS" and "EXP3-Bayes-Poisson-TS"; \code{iter.BP} specifies the number of iterations to compute posterior; 
#' \code{ndraws.BP} specifies the number of posterior samples drawn from posterior distribution; \code{interval.BP} is specified to draw each posterior sample from
#'  a sample sequence of length \code{interval.BP}. 
#'
#' \code{HyperTS}: A list consisting of a vector \code{method.list}, needed for "HyperTS". \code{method.list} is a vector of character strings choosing from "Epsilon-Greedy", "Epsilon-Decreasing", "Thompson-Sampling",
#' "EXP3", "UCB", "Bayes-Poisson-TS", "Greedy-Thompson-Sampling", "EXP3-Thompson-Sampling",
#' "Greedy-Bayes-Poisson-TS" and "EXP3-Bayes-Poisson-TS". "HyperTS" will construct an ensemble consisting all the methods in \code{method.list}.
#' @param nburnin A positive integer specifying the number of periods to allocate each arm equal traffic before applying any strategy.
#' @param nperiod A positive integer specifying the number of periods to apply the strategy.
#' @param npulls.per.period  A positive integer  or a vector of positive integers. Default value is 1. If \code{npulls.per.period} is a positive integer,
#' the number of pulls is \code{npulls.per.period} for each period. If \code{npulls.per.period} is a vector, each element represents
#' the number of pulls for one period; the length of \code{npulls.per.period} should be equal to \code{nburnin} + \code{nperiod}.
#' @param reward.family A character string specifying the distribution family of reward. Available distribution includes
#'  "Bernoulli", "Poisson" and "Gaussian". If "Gaussian" is chosen to be the reward distribution,
#' a vector of standard deviation should be provided in \code{sd.reward}.
#' @param sd.reward A vector of non-negative numbers specifying standard deviation of each arm's reward distribution if "Gaussian" is chosen to be the reward distribution. Default to be NULL. 
#' See \code{reward.family}. 
#' @param mean.reward A vector or a matrix of real numbers specifying the mean reward of each arm. If \code{mean.reward} is a vector, each element is the mean reward 
#'  for each arm and the mean reward of each arm is unchanged throughout all periods (corresponding to the stationary Multi-Armed Bandit). 
#'  If \code{mean.reward} is a matrix, it should
#'  have (\code{nburnin} + \code{nperiod}) rows. The mean reward of each arm could change. Each row represents a mean reward vector for each period
#'   (corresponding to nonstationary and adversarial Multi-Armed Bandit).
#' @param weight.plot A logic value with FALSE as default. If TRUE, weight plot object for each arm is returned.
#' @param regret.plot A logic value with FALSE as default. If TRUE,  relative regret plot object is returned.
#'
#' @return a list consisting of:
#' \item{weight}{A weight matrix whose each element is the allocated weight for each arm and period. Each row represents one arm and each column represents one period.}
#' \item{regret}{A relative regret vector whose each element is relative regret for each period. For definition of relative regret, see above.}
#' \item{weight.plot.object}{If weight.plot = TRUE, a ggplot object is returned.}
#' \item{regret.plot.object}{If regret.plot = TRUE, a ggplot object is returned.}
#' @export
#' @examples
#' ### Simulate Thompson-Sampling
#' set.seed(100)
#' res <- SimulateMultiplePeriods(method = "Thompson-Sampling", method.par = list(ndraws.TS = 1000), nburnin = 30, nperiod = 180, npulls.per.period = 5, reward.family = "Bernoulli",  mean.reward = runif(3, 0, 0.1), weight.plot = TRUE)
#' res$weight.plot.object
#' ### Simulate EXP3-Thompson-Sampling
#' set.seed(100)
#' res <- SimulateMultiplePeriods(method = "EXP3-Thompson-Sampling", method.par = list(ndraws.TS = 1000, EXP3 = list(gamma = 0, eta = 0.1)), nburnin = 30, nperiod = 180, npulls.per.period = 5, 
#'               reward.family = "Bernoulli",  mean.reward = runif(3, 0, 0.1), weight.plot = TRUE)
#' res$weight.plot.object
#' ### Simulate ensemble method HyperTS given "Thompson-Sampling", "Epsilon-Greedy" and "Epsilon-Decreasing"
#' set.seed(100)
#' res <- SimulateMultiplePeriods(method = "HyperTS", method.par = list(ndraws.TS = 1000, epsilon = 0.1, HyperTS = list(method.list = c("Thompson-Sampling", "Epsilon-Greedy", "Epsilon-Decreasing"))),
#'                                nburnin = 30, nperiod = 180, npulls.per.period = 5, reward.family = "Poisson",  mean.reward = runif(3, 0, 0.1), weight.plot = TRUE)
#' res$weight.plot.object




SimulateMultiplePeriods <- function(method = "Thompson-Sampling", method.par = list(ndraws.TS = 1000), nburnin, nperiod, reward.family,  mean.reward, sd.reward = NULL, npulls.per.period = 1,
                           weight.plot = FALSE, regret.plot = FALSE){
  if (! method %in% c("Epsilon-Greedy", "Epsilon-Decreasing", "Thompson-Sampling","EXP3", "UCB", "Bayes-Poisson-TS", 
                      "Greedy-Thompson-Sampling", "EXP3-Thompson-Sampling",  "Greedy-Bayes-Poisson-TS", "EXP3-Bayes-Poisson-TS", "HyperTS")){
    stop("Please specify correct method names!")
  }

  if (is.vector(mean.reward)){
    rewardVec <- mean.reward
  }


  if (length(npulls.per.period) == 1){
    npullsVec <- npulls.per.period
  }else if(length(npulls.per.period) != nburnin + nperiod | min(npulls.per.period) <= 0){
    stop("Please specify correct number of pulls per period!")
  }

  if (is.vector(mean.reward)){
    bestReward <- max(rewardVec)
    n <- length(rewardVec)
  }else{
    bestReward <- apply(mean.reward, 1, max)
    n <- dim(mean.reward)[2]
  }


  if (reward.family == "Gaussian" & (length(sd.reward) != n | anyNA(sd.reward))){
    stop("Please specify correct standard deviation for Gaussian reward family!")
  }

  if (is.vector(mean.reward)){
    if (length(npulls.per.period) == 1){
      burninTrial <- rmultinom(1, nburnin * npullsVec, rep(1 / n, n))
    }else{
      burninTrial <- rmultinom(1, sum(npulls.per.period[1:nburnin]), rep(1 / n, n))
    }
    burninReward <- apply(cbind(burninTrial, rewardVec, sd.reward), 1, GetReward, reward.family)
  }else{
    burninTrial <- rep(0, n)
    burninReward <- rep(0, n)
    if (length(npulls.per.period) == 1){
      for (period in 1:nburnin){
        tempTrial <- rmultinom(1, npullsVec, rep(1 / n, n))
        tempReward <- apply(cbind(tempTrial, mean.reward[period, ], sd.reward), 1,  GetReward, reward.family)
        burninTrial <- burninTrial + tempTrial
        burninReward <- burninReward + tempReward
      }
    }else{
      for (period in 1:nburnin){
        tempTrial <- rmultinom(1, npulls.per.period[period], rep(1 / n, n))
        tempReward <- apply(cbind(tempTrial, mean.reward[period, ], sd.reward), 1, GetReward, reward.family)
        burninTrial <- burninTrial + tempTrial
        burninReward <- burninReward + tempReward
      }
    }
  }
  burninEvent <- data.frame(trial = burninTrial, reward = burninReward)


  equalDailyRegret <- rep(0, nperiod)
  for (period in 1:nperiod){
    if (length(npulls.per.period) == 1){
      dailyTrial <- rep(npullsVec / n, n)
    }else{
      dailyTrial <- rep(npulls.per.period[nburnin + period] / n, n)
    }
    if (is.vector(mean.reward)){
      equalDailyRegret[period] <- sum(dailyTrial * (bestReward - rewardVec))
    }else{
      equalDailyRegret[period] <- sum(dailyTrial * (bestReward[nburnin + period] - mean.reward[nburnin + period, ]))
    }
  }


  allWeight <- cbind()
  all.event <- burninEvent
  dailyRegret <- rep(0, nperiod)

  weight <- as.vector(CalculateWeight(method = "Thompson-Sampling", sd.reward = sd.reward, reward.family = reward.family, all.event = all.event, method.par = list(ndraws.TS = 1000)))
  EXP3Info <- list(prevWeight = weight, EXP3Trial = burninTrial, EXP3Reward = burninReward)

  if (method == "HyperTS"){
    nmethod <- length(method.par$HyperTS$method.list)
    total.reward <- rep(0, nmethod)
    if (reward.family == "Bernoulli"){
      total.trial <- rep(0, nmethod)
    }
    if (reward.family == "Gaussian" | reward.family == "Poisson"){
      total.trial <- rep(1, nmethod)
    }
  }




  for (period in 1:nperiod){
    if (method == "HyperTS"){
      ndraws <- 1000
      ans <- matrix(nrow = ndraws, ncol = nmethod)
      if (reward.family == "Bernoulli"){
        for (i in 1:nmethod) ans[ ,i] <- rbeta(ndraws, total.reward[i] + 1,
                                               total.trial[i] - total.reward[i] + 1)
      }
      if (reward.family == "Gaussian"){
        for (i in 1:nmethod) ans[ ,i] <- rnorm(ndraws, total.reward[i] / total.trial[i], sd.reward[i] / sqrt(total.trial[i]))
      }
      if (reward.family == "Poisson"){
        for (i in 1:nmethod) ans[ ,i] <- rgamma(ndraws, shape = total.reward[i] + 1, scale = 1 / total.trial[i])
      }
      method.index <- which.max(as.vector(table(factor(max.col(ans), levels = 1:nmethod))))
      method.chosen <- method.par$HyperTS$method.list[method.index]
      weight <- CalculateWeight(method.chosen, all.event = all.event, sd.reward = sd.reward, reward.family = reward.family, method.par = method.par, period = period, EXP3Info = EXP3Info)
    }

    if (method != "HyperTS"){
      weight <- CalculateWeight(method, all.event = all.event, sd.reward = sd.reward, reward.family = reward.family, method.par = method.par, period = period, EXP3Info = EXP3Info)
    }

    allWeight <- cbind(allWeight, weight)
    if (length(npulls.per.period) == 1){
      dailyTrial <- rmultinom(1, npullsVec, weight)
    }else{
      dailyTrial <- rmultinom(1, npulls.per.period[nburnin + period], weight)
    }
    if (is.vector(mean.reward)){
      dailyReward <- apply(cbind(dailyTrial, rewardVec, sd.reward), 1, GetReward, reward.family)
    }else{
      dailyReward <- apply(cbind(dailyTrial, mean.reward[nburnin + period, ], sd.reward), 1, GetReward, reward.family)
    }

    all.event$trial <- all.event$trial + dailyTrial
    all.event$reward <- all.event$reward + dailyReward
    if (is.vector(mean.reward)){
      dailyRegret[period] <- sum(dailyTrial * (bestReward - rewardVec))
    }else{
      dailyRegret[period] <- sum(dailyTrial * (bestReward[nburnin + period] - mean.reward[nburnin + period, ]))
    }
    EXP3Info = list(prevWeight = weight, EXP3Trial = dailyTrial, EXP3Reward = dailyReward)

    if (method == "HyperTS"){
      total.reward[method.index] <- total.reward[method.index] + sum(dailyReward)
      total.trial[method.index] <- total.trial[method.index] + sum(dailyTrial)
    }
  }
  relativeRegret <- dailyRegret / equalDailyRegret

  if (weight.plot == TRUE){
    weightVector <- c(t(allWeight))
    if (is.vector(mean.reward)){
      names <- rep(sapply(rewardVec, function(x) paste("Mean Reward =", x)), each = nperiod)
    }else{
      names <- rep(sapply(1:n, function(x) paste("Arm", x)), each = nperiod)
    }

    periodSeq <- rep(1:nperiod, n)
    graphData <- data.frame(names, periodSeq, weightVector)
    weight.plot.object <- ggplot(graphData, aes(x = periodSeq, y = weightVector, colour = names)) + geom_line() + ggtitle("Weight Plot") +
      theme(legend.text = element_text(size = 12, face = "bold"), axis.text.y = element_text(size = 12), legend.title = element_blank()) + 
      labs(y = "weight", x = "period")
  }
  if (regret.plot == TRUE){
    periodSeq <- 1:nperiod
    graphData <- data.frame(periodSeq, relativeRegret)
    regret.plot.object <- ggplot(graphData, aes(x = periodSeq, y = relativeRegret)) + geom_line() + ggtitle("Regret Plot") +
      theme(legend.text = element_text(size = 12, face = "bold"), axis.text.y = element_text(size = 12), legend.title = element_blank()) + 
      labs(y = "Relative Regret", x = "period")
  }

  if (weight.plot == TRUE & regret.plot == TRUE){
    return(list(weight = allWeight, regret = relativeRegret, weight.plot.object = weight.plot.object, regret.plot.object = regret.plot.object))
  }
  if (weight.plot == FALSE & regret.plot == TRUE){
    return(list(weight = allWeight, regret = relativeRegret, regret.plot.object = regret.plot.object))
  }
  if (weight.plot == TRUE & regret.plot == FALSE){
    return(list(weight = allWeight, regret = relativeRegret, weight.plot.object = weight.plot.object))
  }
  if (weight.plot == FALSE & regret.plot == FALSE){
    return(list(weight = allWeight, regret = relativeRegret))
  }
}




GetReward <- function(x, reward.family, sd.reward){
  if (reward.family == "Bernoulli") {
    return(rbinom(n = 1, size = x[1], prob = x[2]))
  }
  if (reward.family == "Gaussian") {
    return(rnorm(n = 1, mean = x[1] * x[2], sd = x[3] * sqrt(x[1])))
  }
  if (reward.family == "Poisson"){
    return(rpois(n = 1, lambda = x[1] * x[2]))
  }
}





