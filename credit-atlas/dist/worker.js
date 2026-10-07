importScripts('risk-engine.js');
self.onmessage=({data})=>{try{const result=CreditRisk.run(data.base,data.parameters,p=>self.postMessage({progress:p}));self.postMessage({result});}catch(error){self.postMessage({error:error.message});}};
