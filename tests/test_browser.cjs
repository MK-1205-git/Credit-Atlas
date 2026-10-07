const {test}=require('node:test');
const assert=require('node:assert/strict');
const {run,cdf,cholesky}=require('../dist/risk-engine.js');
const base=require('../dist/baseline.json');
const p={asset_shock:0,volatility_multiplier:1,market:.25,horizon:1,confidence:.99,lgd:.6,seed:42};
test('normal CDF and Cholesky agree with known values',()=>{
 assert.ok(Math.abs(cdf(1.96)-.97500210485)<1e-7);
 const l=cholesky([[1,.5],[.5,1]]);assert.ok(Math.abs(l[1][1]-Math.sqrt(.75))<1e-12);
});
test('100,000 browser paths match Python analytical risk and sampling bounds',()=>{
 const r=run(base,p);
 assert.equal(r.histogram.counts.reduce((a,b)=>a+b,0),100000);
 assert.ok(Math.abs(r.risk.expected_loss-base.risk.expected_loss)<1);
 assert.ok(Math.abs(r.risk.simulated_mean_loss-r.risk.expected_loss)<5*r.risk.mean_standard_error);
 assert.ok(r.risk.expected_shortfall>=r.risk.var);
 r.firms.forEach(f=>assert.ok(Math.abs(f.probability_of_default-f.simulated_default_rate)<5*Math.sqrt(f.probability_of_default*(1-f.probability_of_default)/100000)+.0001));
 const repeat=run(base,p);assert.deepEqual(r.histogram,repeat.histogram);
});
test('stress raises marginal credit risk; zero LGD gives zero losses',()=>{
 const r=run(base,{...p,asset_shock:-.3,volatility_multiplier:1.6,lgd:0});
 assert.equal(r.risk.expected_loss,0);assert.equal(r.risk.expected_shortfall,0);
 assert.equal(r.risk.zero_loss_probability,1);
 r.firms.forEach((f,i)=>assert.ok(f.probability_of_default>base.firms[i].probability_of_default));
});
