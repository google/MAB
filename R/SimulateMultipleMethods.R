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

#' Compare various strategies for Multi-Armed Bandit in stationary and non-stationary scenarios
#'
#' This function is aimed to simulate data in different scenarios to compare various strategies in Multi-Armed Bandit. 
#' Users can specify the distribution of the number of arms, the distribution of mean reward, the distribution of the number of pulls in one period and the stationariness to simulate different scenarios.
#'  Relative regret is returned and average relative regret plot is returned if needed.
#' See \code{\link{SimulateMultiplePeriods}} for more details.
#' @param method A vector of character strings choosing from "Epsilon-Greedy", "Epsilon-Decreasing", "Thompson-Sampling",
#' "EXP3", "UCB", "Bayes-Poisson-TS", "Greedy-Thompson-Sampling", "EXP3-Thompson-Sampling", 
#' "Greedy-Bayes-Poisson-TS", "EXP3-Bayes-Poisson-TS"  and "HyperTS". See \code{\link{SimulateMultiplePeriods}} for more details. Default is "Thompson-Sampling".
#' @param method.par A list of parameters needed for different methods: 
#' 
#' \code{epsilon}: A real number between 0 and 1; needed for "Epsilon-Greedy", "Epsilon-Decreasing", "Greedy-Thompson-Sampling" and "Greedy-Bayes-Poisson-TS".
#' 
#' \code{ndraws.TS}: A positive integer specifying the number of random draws from the posterior; 
#' needed for "Thompson-Sampling", "Greedy-Thompson-Sampling" and "EXP3-Thompson-Sampling". Default is 1000.
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
#' @param iter A positive integer specifying the number of iterations.
#' @param nburnin A positive integer specifying the number of periods to allocate each arm equal traffic before applying any strategy. 
#' @param nperiod A positive integer specifying the number of periods to apply various strategies.
#' 
#' @param reward.mean.family A character string specifying the distribution family to generate mean reward of each arm. Available distribution includes "Uniform", "Beta" and "Gaussian".
#' @param reward.family A character string specifying the distribution family of reward. Available distribution includes
#'  "Bernoulli", "Poisson" and "Gaussian". If "Gaussian" is chosen to be the reward distribution,
#'  a vector of standard deviation should be provided in \code{sd.reward} in \code{data.par}.
#' @param narms.family A character string specifying the distribution family of the number of arms. Available distribution includes "Poisson" and "Binomial". 
#' @param npulls.family A character string specifying the distribution family of the number of pulls per period. 
#' For continuous distribution, the number of pulls will be rounded up. Available distribution includes "Log-Normal" and "Poisson".
#' @param stationary A logic value indicating whether a stationary Multi-Armed Bandit is considered (corresponding to the case that the reward mean is unchanged). Default to be TRUE.
#' @param nonstationary.type A character string indicating how the mean reward varies. Available types include "Random Walk" and "Geometric Random Walk" 
#' (reward mean follows random walk in the log scale). Default to be NULL.
#' @param data.par A list of data generating parameters:
#' 
#' \code{reward.mean}: A list of parameters of \code{reward.mean.family}: \code{min} and \code{max} are two real numbers specifying
#' the bounds when \eqn{reward.mean.family = "Uniform"}; \code{shape1} and \code{shape2} are two shape parameters when \eqn{reward.mean.family = "Beta"}; 
#' \code{mean} and \code{sd} specify mean and standard deviation when \eqn{reward.mean.family = "Gaussian"}. 
#' 
#' \code{reward.family}: A list of parameters of \code{reward.family}: \code{sd} is a vector of non-negative numbers specifying standard deviation of each arm's reward distribution 
#' if "Gaussian" is chosen to be the reward distribution.
#' 
#'  \code{narms.family}: A list of parameters of \code{narms.family}: \code{lambda} is a positive parameter specifying the mean when \eqn{narms.family = "Poisson"}; \code{size} and \code{prob}
#' are 2 parameters needed when \eqn{narms.family = "Binomial"}.
#' 
#' \code{npulls.family}: A list of parameters of \code{npulls.family}: \code{meanlog} and \code{sdlog} are 2 positive parameters specifying the mean and standard deviation in the log scale
#'  when \eqn{npulls.family = "Log-Normal"}; \code{lambda} is a positive parameter specifying the mean when \eqn{npulls.family = "Poisson"}.
#' 
#'
#' \code{nonstationary.family}: A list of parameters of \code{nonstationary.type}: \code{sd} is a positive parameter specifying the standard deviation of white noise
#'  when \eqn{nonstationary.type = "Random Walk"}; \code{sdlog} is a positive parameter specifying the log standard deviation of white noise
#'  when \eqn{nonstationary.type = "Geometric Random Walk"}.
#' @param regret.plot A logic value indicating whether a average regret plot is returned. Default to be FALSE.
#' @return a list consisting of:
#' \item{regret.matrix}{A three-dimensional array with each dimension corresponding to the period, iteration and method.}
#' \item{regret.plot.object}{If regret.plot = TRUE, a ggplot object is returned.}
#' @export
#' @examples
#' ### Compare Epsilon-Greedy and Thompson Sampling in the stationary case.
#' set.seed(100)
#' res <- SimulateMultipleMethods(method = c("Epsilon-Greedy", "Thompson-Sampling"), method.par = list(epsilon = 0.1, ndraws.TS = 1000), iter = 100, nburnin = 30, nperiod = 180, 
#'                      reward.mean.family = "Uniform", reward.family = "Bernoulli", narms.family = "Poisson", npulls.family = "Log-Normal", 
#'                      data.par = list(reward.mean = list(min = 0, max = 0.1), npulls.family = list(meanlog = 3, sdlog = 1.5), narms.family = list(lambda = 5)), regret.plot = TRUE)
#' res$regret.plot.object
#' ### Compare Epsilon-Greedy, Thompson Sampling and EXP3 in the non-stationary case.
#' set.seed(100)
#' res <- SimulateMultipleMethods(method = c("Epsilon-Greedy", "Thompson-Sampling", "EXP3"), method.par = list(epsilon = 0.1, ndraws.TS = 1000, EXP3 = list(gamma = 0, eta = 0.1)), 
#'                      iter = 100, nburnin = 30, nperiod = 90, reward.mean.family = "Beta", reward.family = "Bernoulli", narms.family = "Binomial", 
#'                      npulls.family = "Log-Normal", stationary = FALSE, nonstationary.type = "Geometric Random Walk",
#'                      data.par = list(reward.mean = list(shape1 = 2, shape2 = 5), npulls.family = list(meanlog = 3, sdlog = 1), narms.family = list(size = 10, prob = 0.5), nonstationary.family = list(sdlog = 0.05)),
#'                      regret.plot = TRUE)
#' res$regret.plot.object



SimulateMultipleMethods <- function(method = "Thompson-Sampling", method.par = list(ndraws.TS = 1000), iter, nburnin, nperiod, reward.mean.family, reward.family, narms.family, npulls.family, 
                           stationary = TRUE, nonstationary.type = NULL, data.par, regret.plot = FALSE){
  if (! all(method %in% c("Epsilon-Greedy", "Epsilon-Decreasing", "Thompson-Sampling","EXP3", "UCB", "Bayes-Poisson-TS", 
                      "Greedy-Thompson-Sampling", "EXP3-Thompson-Sampling",  "Greedy-Bayes-Poisson-TS", "EXP3-Bayes-Poisson-TS", "HyperTS"))){
    stop("Please specify correct method names!")
  }
  if (! reward.family %in% c("Bernoulli", "Poisson", "Gaussian")){
    stop("Please specify correct reward family!")
  }
  if (! reward.mean.family %in% c("Uniform", "Beta", "Gaussian")){
    stop("Please specify correct mean reward family!")
  }
  if (! narms.family %in% c("Binomial", "Poisson")){
    stop("Please specify correct distribution family for the number of arms!")
  }
  if (! npulls.family %in% c("Log-Normal", "Poisson")){
    stop("Please specify correct distribution family for the number of pulls per period!")
  }


  nmethod <- length(method)
  regret.matrix <- array(0, c(nperiod, iter, nmethod))
  for (i in 1:iter){
    if (narms.family == "Poisson"){
      lambda <- data.par$narms.family$lambda
      while(TRUE){
        narms <- rpois(1, lambda)
        if (narms > 1){
          break
        }
      }
    }
    if (narms.family == "Binomial"){
      size <- data.par$narms.family$size
      prob <- data.par$narms.family$prob
      while(TRUE){
        narms <- rbinom(1, size, prob)
        if (narms > 1){
          break
        }
      }
    }


    if (reward.mean.family == "Uniform"){
      mean.reward <- runif(narms, min = data.par$reward.mean$min,  max = data.par$reward.mean$max)
    }
    if (reward.mean.family == "Beta"){
      mean.reward <- rbeta(narms, shape1 = data.par$reward.mean$shape1,  shape2 = data.par$reward.mean$shape2)
    }
    if (reward.mean.family == "Gaussian"){
      if (reward.family == "Bernoulli" | reward.family == "Poisson"){
        stop("Please not use Gaussian distribution to generate mean reward if reward family is Bernoulli or Poisson!")
      }
      mean.reward <- rnorm(narms, mean = data.par$reward.mean$mean,  sd = data.par$reward.mean$sd)
    }


    if (stationary == FALSE){
      mean.reward.matrix <- matrix(0, nrow = nburnin + nperiod, ncol = narms)
      mean.reward.matrix[1, ] <- mean.reward
      if (nonstationary.type == "Geometric Random Walk"){
        for (j in 2:(nburnin + nperiod)){
          if (reward.family == "Bernoulli"){
            mean.reward.matrix[j, ] <- sapply(mean.reward.matrix[j - 1, ] * rlnorm(narms, meanlog = 0, sdlog = data.par$nonstationary.family$sdlog), function(x) min(1, x))
          }else{
            mean.reward.matrix[j, ] <- mean.reward.matrix[j - 1, ] * rlnorm(narms, meanlog = 0, sdlog = data.par$nonstationary.family$sdlog)
          }
        }
        mean.reward <- mean.reward.matrix
      }
      if (nonstationary.type == "Random Walk"){
        if (reward.family == "Bernoulli" | reward.family == "Poisson"){
          stop("Please not use Random Walk to generate mean reward if reward family is Bernoulli or Poisson!")
        }
        for (j in 2:(nburnin + nperiod)){
          mean.reward.matrix[j, ] <- mean.reward.matrix[j - 1, ] + rnorm(narms, mean = 0, sd = data.par$nonstationary.family$sd)
        }
        mean.reward <- mean.reward.matrix
      }
    }


    if (npulls.family == "Log-Normal"){
      npulls.per.period <- ceiling(rlnorm(nburnin + nperiod, meanlog = data.par$npulls.family$meanlog, sdlog = data.par$npulls.family$sdlog))
    }
    if (npulls.family == "Poisson"){
      npulls.per.period <- rep(0, nburnin + nperiod)
      for (num in 1:(nburnin + nperiod)){
        while(TRUE){
          npulls.per.period[num] <- rpois(1, lambda = data.par$npulls.family$lambda)
          if (npulls.per.period[num] > 0){
            break
          }
        }
      }
    }

    for (k in 1:length(method)){
      regret <- SimulateMultiplePeriods(method = method[k], nburnin = nburnin, nperiod = nperiod, mean.reward = mean.reward, reward.family = reward.family, sd.reward = data.par$reward.family$sd,  
                         npulls.per.period = npulls.per.period, method.par = method.par)$regret
      regret.matrix[, i, k] <- regret
    }
  }

  if (regret.plot == TRUE){
    relativeRegret <- c(apply(regret.matrix, c(1,3), mean))
    methodName <- rep(method, each = nperiod)
    daySeq <- rep(1:nperiod, nmethod)
    graphData <- data.frame(daySeq, relativeRegret, methodName)
    regret.plot.object <- ggplot(graphData, aes(x = daySeq, y = relativeRegret, colour = factor(methodName))) + geom_line() + ggtitle("Regret Plot") +
      theme(legend.text = element_text(size = 12, face = "bold"), axis.text.y = element_text(size = 12), legend.title = element_blank()) + 
      labs(y = "Relative Regret", x = "day")
    return(list(regret.matrix = regret.matrix, regret.plot.object = regret.plot.object))
  }else{
    return(list(regret.matrix = regret.matrix))
  }
}


