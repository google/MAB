test_that("Test Epsilon-Greedy",{
  expect_equal(CalculateWeight(method = "Epsilon-Greedy", method.par = list(epsilon = 0.01), all.event = data.frame(reward = 1:3, trial = rep(10, 3)), reward.family = "Bernoulli"),
               c(0.005, 0.005, 0.990))})
test_that("Test Epsilon-Decreasing",{
  expect_equal(CalculateWeight(method = "Epsilon-Decreasing", method.par = list(epsilon = 0.01), all.event = data.frame(reward = 1:3, trial = rep(10, 3)), reward.family = "Bernoulli"),
               c(0.005, 0.005, 0.990))})
test_that("Test UCB",{
  expect_equal(CalculateWeight(method = "UCB", method.par = list(epsilon = 0.01), all.event = data.frame(reward = 1:3, trial = rep(10, 3)), reward.family = "Bernoulli"),
               c(0, 0, 1))})