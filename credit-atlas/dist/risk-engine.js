/* Browser stress engine. Base asset parameters come from the Python/SciPy fit.
 * Risk-neutral PD, fixed LGD, seeded Gaussian copula, full Cholesky factorization.
 */
(function(root){
'use strict';
function cdf(x){const sign=x<0?-1:1;const z=Math.abs(x)/Math.SQRT2,t=1/(1+.3275911*z);const erf=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-z*z);return .5*(1+sign*erf);}
function cholesky(c){const n=c.length,L=Array.from({length:n},()=>Array(n).fill(0));for(let i=0;i<n;i++)for(let j=0;j<=i;j++){let s=c[i][j];for(let k=0;k<j;k++)s-=L[i][k]*L[j][k];if(i===j){if(s<=0)throw Error('Correlation matrix is not positive definite');L[i][j]=Math.sqrt(s);}else L[i][j]=s/L[j][j];}return L;}
function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};}
function run(base,p,onProgress){
 const start=Date.now(),fs=base.firms,n=fs.length,N=100000;
 if(!fs.length||fs.length>100||p.market<0||p.market>.8||p.horizon<=0||p.volatility_multiplier<=0||p.asset_shock<=-1||p.confidence<=0||p.confidence>=1||p.lgd<0||p.lgd>1)throw Error('Invalid simulation parameters');
 const firms=fs.map(f=>{let a=f.asset_value*(1+p.asset_shock),v=f.asset_volatility*p.volatility_multiplier;let dd=(Math.log(a/f.debt)+(base.panel.rate-.5*v*v)*p.horizon)/(v*Math.sqrt(p.horizon));return {...f,stressed_asset_value:a,stressed_asset_volatility:v,distance_to_default:dd,probability_of_default:cdf(-dd),expected_loss:f.exposure*p.lgd*cdf(-dd)};});
 const corr=fs.map((f,i)=>fs.map((g,j)=>i===j?1:p.market+(f.sector===g.sector?.15:0)));
 const L=cholesky(corr),u=rng(p.seed),losses=new Float64Array(N),counts=new Uint32Array(n),z=new Float64Array(n);
 let spare=null;function normal(){if(spare!==null){let v=spare;spare=null;return v;}let radius=Math.sqrt(-2*Math.log(Math.max(u(),1e-15))),angle=2*Math.PI*u();spare=radius*Math.sin(angle);return radius*Math.cos(angle);}
 let sum=0,sum2=0,zeros=0;
 for(let s=0;s<N;s++){
  for(let j=0;j<n;j++)z[j]=normal();let loss=0;
  for(let i=0;i<n;i++){let correlated=0;for(let j=0;j<=i;j++)correlated+=L[i][j]*z[j];if(correlated < -firms[i].distance_to_default){loss+=firms[i].exposure*p.lgd;counts[i]++;}}
  losses[s]=loss;sum+=loss;sum2+=loss*loss;if(loss===0)zeros++;
  if(s%20000===0&&onProgress)onProgress(s/N);
 }
 losses.sort();const index=Math.ceil(p.confidence*N)-1,varValue=losses[index],mass=(1-p.confidence)*N,whole=Math.floor(mass+1e-10),fraction=Math.max(0,mass-whole);let tail=0;
 for(let i=N-whole;i<N;i++)tail+=losses[i];if(fraction>1e-10)tail+=fraction*losses[N-whole-1];
 const total=fs.reduce((a,f)=>a+f.exposure,0),max=total*p.lgd,binN=40,edges=Array.from({length:binN+1},(_,i)=>max*i/binN),hist=Array(binN).fill(0);
 for(let i=0;i<N;i++)hist[Math.min(binN-1,Math.floor(losses[i]/(max||1)*binN))]++;
 firms.forEach((f,i)=>f.simulated_default_rate=counts[i]/N);
 return {schema_version:1,panel:{...base.panel,correlation:corr,correlation_method:'Assumed sector matrix: market '+p.market+' + sector 0.15'},firms,parameters:{...p,simulations:N},total_exposure:total,
 engine:'Browser Web Worker; Mulberry32/Box–Muller RNG (different from NumPy)',elapsed_ms:Date.now()-start,
 risk:{expected_loss:firms.reduce((s,f)=>s+f.expected_loss,0),simulated_mean_loss:sum/N,mean_standard_error:Math.sqrt(Math.max(0,(sum2-sum*sum/N)/(N-1))/N),var:varValue,expected_shortfall:tail/mass,zero_loss_probability:zeros/N,empirical_default_rates:Array.from(counts,v=>v/N)},
 histogram:{edges,counts:hist},method:'Risk-neutral terminal default; fixed LGD; Gaussian latent dependence; undiscounted default losses'};
}
root.CreditRisk={cdf,cholesky,run};if(typeof module!=='undefined')module.exports=root.CreditRisk;
})(typeof self!=='undefined'?self:globalThis);
