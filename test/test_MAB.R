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

test_that("Test Epsilon-Greedy",{
  expect_equal(CalculateWeight(method = "Epsilon-Greedy",
                               method.par = list(epsilon = 0.01),
                               all.event = data.frame(reward = 1:3,
                                                      trial = rep(10, 3)),
                               reward.family = "Bernoulli"),
               c(0.005, 0.005, 0.990))})
test_that("Test Epsilon-Decreasing",{
  expect_equal(CalculateWeight(method = "Epsilon-Decreasing",
                               method.par = list(epsilon = 0.01),
                               all.event = data.frame(reward = 1:3,
                                                      trial = rep(10, 3)),
                               reward.family = "Bernoulli"),
               c(0.005, 0.005, 0.990))})
test_that("Test UCB",{
  expect_equal(CalculateWeight(method = "UCB",
                               method.par = list(epsilon = 0.01),
                               all.event = data.frame(reward = 1:3,
                                                      trial = rep(10, 3)),
                               reward.family = "Bernoulli"),
               c(0, 0, 1))})
