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

#' Calculate the probability of pulling each arm in the next period for various
#' strategies
#'
#' This function is aimed to compute the probability of pulling each arm for
#' various methods in Multi-Armed Bandit given the total reward and the number
#' of trials for each arm.
#'
#' @param method A character string choosing from "Epsilon-Greedy",
#' "Epsilon-Decreasing", "Thompson-Sampling",
#' "EXP3", "UCB", "Bayes-Poisson-TS", "Greedy-Thompson-Sampling",
#' "EXP3-Thompson-Sampling",
#' "Greedy-Bayes-Poisson-TS" and "EXP3-Bayes-Poisson-TS".
#' See \code{\link{SimulateMultiplePeriods}} for more details.
#' Default is "Thompson-Sampling".
#' @param method.par A list of parameters needed for different methods:
#'
#' \code{epsilon}: A real number between 0 and 1; needed for "Epsilon-Greedy",
#' "Epsilon-Decreasing", "Greedy-Thompson-Sampling" and "Greedy-Bayes-Poisson-TS".
#'
#' \code{ndraws.TS}: A positive integer specifying the number of random draws
#' from the posterior;
#' needed for "Thompson-Sampling", "Greedy-Thompson-Sampling" and
#' "EXP3-Thompson-Sampling".  Default is 1000.
#'
#' \code{EXP3}: A list consisting of two real numbers \code{eta} and \code{gamma};
#' \eqn{eta > 0} and \eqn{0 <= gamma < 1}; needed for "EXP3",
#' "EXP3-Thompson-Sampling" and "EXP3-Bayes-Poisson-TS".
#'
#' \code{BP}: A list consisting of three postive integers \code{iter.BP},
#' \code{ndraws.BP} and \code{interval.BP};
#' needed for "Bayes-Poisson-TS", "Greedy-Bayes-Poisson-TS" and
#' "EXP3-Bayes-Poisson-TS"; \code{iter.BP} specifies the number of iterations
#' to compute posterior;
#' \code{ndraws.BP} specifies the number of posterior samples drawn from
#' posterior distribution; \code{interval.BP} is specified to draw each
#' posterior sample from
#'  a sample sequence of length \code{interval.BP}.
#' @param all.event A data frame containing two columns \code{trial} and
#' \code{reward} with the number of rows equal to the number of arms.
#' Each element of \code{trial} and \code{reward} represents the number of trials
#' and the total reward for each arm respectively.
#' @param reward.family A character string specifying the distribution family
#' of reward. Available distribution includes
#'  "Bernoulli", "Poisson" and "Gaussian". If "Gaussian" is chosen to be the
#' reward distribution,
#' a vector of standard deviation should be provided in \code{sd.reward}.
#' @param sd.reward A vector of non-negative numbers specifying standard
#' deviation of each arm's reward distribution if "Gaussian" is chosen to be
#' the reward distribution. Default to be NULL.
#' See \code{reward.family}.
#' @param period A positive integer specifying the period index. Default to be 1.
#' @param EXP3Info A list of three vectors \code{prevWeight}, \code{EXP3Trial}
#' and \code{EXP3Reward} with dimension equal to the number of arms,
#' needed for "EXP3", "EXP3-Thompson-Sampling" and "EXP3-Bayes-Poisson-TS":
#'
#' \code{prevWeight}: the weight vector in the previous EXP3 iteration.
#'
#' \code{EXP3Trial} and \code{EXP3Reward}: vectors representing
#' the number of trials and the total reward for each arm in
#' the previous period respectively.
#'
#' See \code{\link{SimulateMultiplePeriods}} for more details.
#' @return A normalized weight vector for future randomized allocation.
#' @export
#' @examples
#' ### Calculate weights using Thompson Sampling if reward follows Poisson distribution
#' set.seed(100)
#' CalculateWeight(method = "Thompson-Sampling",
#'                 method.par = list(ndraws.TS = 1000),
#'                 all.event = data.frame(reward = 1:3, trial = rep(10, 3)),
#'                 reward.family = "Poisson")
#' ### Calculate weights using EXP3
#' CalculateWeight(method = "EXP3",
#'                 method.par = list(EXP3 = list(gamma = 0.01, eta =0.1)),
#'                 all.event = data.frame(reward = 1:3, trial = rep(10, 3)),
#'                 reward.family = "Bernoulli",
#'                 EXP3Info = list(prevWeight = rep(1, 3), EXP3Trial = rep(5, 3), EXP3Reward = 0:2))



CalculateWeight <- function(method = "Thompson-Sampling",
                            method.par = list(ndraws.TS = 1000),
                            all.event,
                            reward.family,
                            sd.reward = NULL,
                            period = 1,
                            EXP3Info = NULL){
  method.name <- c("Epsilon-Greedy", "Epsilon-Decreasing", "Thompson-Sampling",
                   "EXP3", "UCB", "Bayes-Poisson-TS",
                   "Greedy-Thompson-Sampling", "EXP3-Thompson-Sampling",
                   "Greedy-Bayes-Poisson-TS", "EXP3-Bayes-Poisson-TS")
  if (! method %in% method.name){
    stop("Please specify correct method names!")
  }
  if (! reward.family %in% c("Bernoulli", "Poisson", "Gaussian")){
    stop("Please specify correct reward family!")
  }
  
  
  if (method == method.name[1]){
    if (! is.number(method.par$epsilon) ){
      stop("Please specify correct parameters for Epsilon-Greedy!")
    }
    eps <- method.par$epsilon
    rate <- ifelse(all.event$trial == 0, 0, all.event$reward / all.event$trial)
    n <- length(rate)
    maxIdx <- which(rate == max(rate))
    maxCount <- length(maxIdx)
    weight <- rep(eps / (n - maxCount), n)
    weight[maxIdx] <- (1 - eps) / maxCount
    return(weight)
  }
  
  
  if (method == method.name[2]){
    if ( ! is.number(method.par$epsilon)){
      stop("Please specify correct parameters for Epsilon-Decreasing!")
    }
    eps <- method.par$epsilon / period
    rate <- ifelse(all.event$trial == 0, 0, all.event$reward / all.event$trial)
    n <- length(rate)
    maxIdx <- which(rate == max(rate))
    maxCount <- length(maxIdx)
    weight <- rep(eps / (n - maxCount), n)
    weight[maxIdx] <- (1 - eps) / maxCount
    return(weight)
  }
  
  
  if (method == method.name[3] |
      method == method.name[7] |
      method == method.name[8]){
    if (! is.number(method.par$ndraws.TS)){
      stop("Please specify correct parameters for Thompson-Sampling!")
    }
    ndraws.TS <- method.par$ndraws.TS
    reward <- all.event$reward
    trial <- all.event$trial
    n <- length(reward)
    ans <- matrix(nrow = ndraws.TS, ncol = n)
    if (reward.family == "Bernoulli"){
      failure <- trial - reward
      for (i in 1:n) ans[ ,i] <- rbeta(ndraws.TS, shape1 = reward[i] + 1,
                                       shape2 = failure[i] + 1)
    }
    if (reward.family == "Gaussian"){
      for (i in 1:n) ans[ ,i] <- rnorm(ndraws.TS, mean = reward[i] / trial[i],
                                       sd = sd.reward[i] / sqrt(trial[i]))
    }
    if (reward.family == "Poisson"){
      for (i in 1:n) ans[ ,i] <- rgamma(ndraws.TS, shape = reward[i] + 1,
                                        scale = 1 / trial[i])
    }
    
    w <- table(factor(max.col(ans), levels = 1:n))
    
    
    if (method == method.name[3]){
      return(as.vector(w / sum(w)))
    }
    
    if (method == method.name[7]){
      if (! is.number(method.par$epsilon)){
        stop("Please specify correct parameters for Greedy-Thompson-Sampling!")
      }
      n <- length(w)
      maxIdx <- which(w == max(w))
      maxCount <- length(maxIdx)
      maxVector <- rep(0, n)
      maxVector[maxIdx] <- 1 / maxCount
      eps <- method.par$epsilon
      return(as.vector(eps * w / sum(w) + (1 - eps) * maxVector))
    }
    
    if (method == method.name[8]){
      if ( ! is.number(method.par$EXP3$gamma) |
           ! is.number(method.par$EXP3$eta) ){
        stop("Please specify correct parameters for EXP3-Thompson-Sampling!")
      }
      if (reward.family != "Bernoulli"){
        stop("Please use Bernoulli Reward Family to run EXP3")
      }
      
      eta <- method.par$EXP3$eta
      gamma <- method.par$EXP3$gamma
      prevWeight <- EXP3Info$prevWeight
      temp <- prevWeight * exp(eta * w / sum(w))
      
      return(as.vector((1 - gamma) * temp / sum(temp) + gamma / n))
    }
  }
  
  
  if (method == method.name[4]){
    if (! is.number(method.par$EXP3$gamma) | ! is.number(method.par$EXP3$eta)){
      stop("Please specify correct parameters for EXP3!")
    }
    if (reward.family != "Bernoulli"){
      stop("Please use Bernoulli Reward Family to run EXP3")
    }
    prevWeight <- EXP3Info$prevWeight
    EXP3Trial <- EXP3Info$EXP3Trial
    EXP3Reward <- EXP3Info$EXP3Reward
    eta <- method.par$EXP3$eta
    gamma <- method.par$EXP3$gamma
    
    EXP3Rate <- ifelse(EXP3Trial == 0, 0, EXP3Reward / EXP3Trial)
    temp <- prevWeight *
      exp(eta * (EXP3Rate - max(EXP3Rate)) * sum(EXP3Trial) / length(EXP3Trial))
    
    return(as.vector((1 - gamma) * temp / sum(temp) + gamma / length(EXP3Trial)))
  }
  
  
  if (method == method.name[5]){
    reward <- all.event$reward
    trial <- all.event$trial
    rate <- reward / trial
    UCB <- rate + sqrt(2 * log(period) / trial)
    maxIdx <- which(UCB == max(UCB))
    maxCount <- length(maxIdx)
    maxVector <- rep(0, length(reward))
    maxVector[maxIdx] <- 1 / maxCount
    return(maxVector)
  }
  
  if (method == method.name[6] |
      method == method.name[9] |
      method == method.name[10]){
    if ( ! is.number(method.par$BP$iter.BP) |
         ! is.number(method.par$BP$ndraws.BP) |
         ! is.number(method.par$BP$interval.BP)){
      stop("Please specify correct parameters for Bayes-Poisson-TS!")
    }
    if (reward.family == "Gaussian"){
      stop("Please not use Gaussian Reward Family to run Bayes-Poisson-TS!")
    }
    iter.BP <- method.par$BP$iter.BP
    ndraws.BP <- method.par$BP$ndraws.BP
    interval.BP <- method.par$BP$interval.BP
    
    n <- length(all.event$trial)
    temp <- all.event
    temp$Id <- sapply(1:n, function(x) paste("Arm", x))
    
    mdl <- SetupEMREoptim(
      "reward ~ 1 + (1|Id) + offset(trial)",
      data = temp, model.constructor = PoissonEMRE,
      burnin = iter.BP - ndraws.BP * interval.BP,
      thinning.interval = 1, llik.interval = 1)
    mdlResult <- FitEMRE(mdl, max.iter = iter.BP, debug = FALSE)
    posterior <- mdlResult$snapshots
    useIdx <- seq(1, ndraws.BP * interval.BP, interval.BP)
    armPos <- posterior[["1__Id"]][useIdx, ]
    biasPos <- posterior[["__bias__"]][useIdx, ]
    winnerPred <- sapply(1:length(biasPos), function(k) {
      biasK <- biasPos[k]
      armK <- armPos[k, ]
      predK <- biasK  * armK
      return(which(predK == max(predK))[1])
    })
    w <- table(factor(winnerPred, levels = 1:n))
    
    if (method == method.name[6]){
      return(as.vector(w / sum(w)))
    }
    
    if (method == method.name[9]){
      if ( ! is.number(method.par$epsilon)){
        stop("Please specify correct parameters for Greedy-Bayes-Poisson-TS!")
      }
      n <- length(w)
      maxIdx <- which(w == max(w))
      maxCount <- length(maxIdx)
      maxVector <- rep(0, n)
      maxVector[maxIdx] <- 1 / maxCount
      eps <- method.par$epsilon
      return(as.vector(eps * w / sum(w) + (1 - eps) * maxVector))
    }
    
    if (method == method.name[10]){
      if ( ! is.number(method.par$EXP3$gamma) |
           ! is.number(method.par$EXP3$eta) ){
        stop("Please specify correct parameters for EXP3-Bayes-Poisson-TS!")
      }
      if (reward.family != "Bernoulli"){
        stop("Please use Bernoulli Reward Family to run Bayes-Poisson-TS!")
      }
      
      eta <- method.par$EXP3$eta
      gamma <- method.par$EXP3$gamma
      prevWeight <- EXP3Info$prevWeight
      temp <- prevWeight * exp(eta * w / sum(w))
      
      return(as.vector((1 - gamma) * temp / sum(temp) + gamma / n))
    }
  }
}
