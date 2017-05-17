
// This simulation inspects how the size and shape of the sample affects
// the reward of a probability distribution built from the sample.
// The goal is to determine the relation between prediction error and
// sample size.
//
// By analysing the results with https://mycurvefit.com/,
// the results almost follow a rule:
//
//   loss(n) = 2 / n
//
// The further analysis shows an approximate relation:
//
//   loss(n) = 20 / x^0.95 - 7 * e^(-ln(x)^2)
//
// Thus, when the sample size doubles, the error is halved.

var _ = require('lodash');
var Categorical = require('lately').Categorical;


var A = 'a';
var B = 'b';
var C = 'c';
var D = 'd';
var E = 'e';

var populations = [
  [A, B],
  [A, B, C],
  [A, B, C, D, E],
  //[A, A, A, B, B],
  //[A, A, A, B, C],
  //[A, A, B, B, C, C],
  //[A, A, A, B, B, C],
  //[A, A, A, A, B, B, B, C, C, D],
  [A, A, A, A, A, A, A, B, C, D],
];

var SAMPLE_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512];
var SAMPLES_PER_SIZE = 1000;


var reward = function (prior, p) {
  // The greater the p to prior, the bigger the reward.
  // If p is smaller than prior, returns negative.
  // The difference in the low or high end of the probability range
  // yields greater rewards than in around 0.5.
  var prec = 0.0001;
  prior = _.clamp(prior, prec, 1 - prec);
  p = _.clamp(p, prec, 1 - prec);
  return Math.log(p / (1 - p)) - Math.log(prior / (1 - prior));
};

var slack = function (x) {
  // Parameters:
  //   x is number of observations
  var lnx = Math.log(x);
  return 0.95 * Math.exp(-lnx * lnx / 6);
};

var expectedReward = function (pop, sam) {
  // Parameters:
  //   pop
  //     Population categorical
  //   sam
  //     Sampled categorical

  // Get event types
  var evs = pop.events();

  // Return a weighted average reward.
  return evs.reduce(function (acc, ev) {
    var pp = pop.prob(ev);
    var p = sam.prob(ev);
    var rew = Math.abs(reward(pp, p));

    // Rare events happen more rarely, thus the reward comes more rarely.
    // However, the reward will be greater for rare events, thus canceling
    // out the weighting: rew * pp / pp = rew
    return acc + (rew / evs.length);
  }, 0);
};

// Algorithm
// - For each population pop
//   - train a categorical P from the pop.
//   - for each sample size S
//     - take SAMPLES_PER_SIZE S-sized samples from P
//     - for each S-sized sample X
//       - train a categorical C from X
//       - compute average reward AR given C and P
//       - return AR
//     - take average AAR of AR's
//     - return pair [S, AAR]
//   - return [pop, pairs]
// - Log results

var results = populations.map(function (pop) {

  var popcat = new Categorical();
  pop.forEach(function (ev) {
    popcat.learn(ev);
  });

  // For each sample size
  var triples = SAMPLE_SIZES.map(function (sampleSize) {

    // Gather a set of expected rewards for this sample size.
    var exRews = _.times(SAMPLES_PER_SIZE, function () {

      // Take the sample. Sample is an array.
      var sample = _.times(sampleSize, function () {
        return popcat.sample();
      });

      // Train a categorical from the sample
      var samcat = new Categorical();
      sample.forEach(function (ev) {
        samcat.learn(ev);
      });

      return expectedReward(popcat, samcat);
    });

    // Gather a set of expected rewards for this sample size
    // after childhood slack correction.
    var slackRews = _.times(SAMPLES_PER_SIZE, function () {

      // Take the sample. Sample is an array.
      var sample = _.times(sampleSize, function () {
        return popcat.sample();
      });

      // Train a categorical from the sample
      var samcat = new Categorical();
      sample.forEach(function (ev) {
        samcat.learn(ev);
      });

      // Compare
      var rew = expectedReward(popcat, samcat);

      // Give some slack, based on sample size
      var slackRew = rew + slack(samcat.weightSum());

      return slackRew;
    });

    // Take average
    var exexRew = _.mean(exRews);
    var exSlackRew = _.mean(slackRews);

    return [sampleSize, exexRew, exSlackRew, exRews];
  });

  var entropy = popcat.entropy();

  return [pop, triples, entropy];
});

results.forEach(function (popResults) {
  console.log('-------');
  console.log('Population:', popResults[0]);
  console.log('Entropy:', popResults[2]);

  var triples = popResults[1];

  triples.forEach(function (tri) {
    console.log('  Sample size:', tri[0]);
    console.log('  Avg. Expected reward:', tri[1]);
    //console.log('  Avg. Expected reward after childhood slack:', tri[2]);
    //console.log('  Exp. rewards:', tri[3]);
  });

  triples.forEach(function (tri) {
    console.log(String(tri[0]), String(tri[1].toPrecision(5)));
    //console.log('{' + tri[0] + ', ' + tri[1].toPrecision(5) + '},');
  });
});
