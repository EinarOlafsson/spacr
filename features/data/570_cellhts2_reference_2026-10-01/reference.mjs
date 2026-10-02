import {WebR} from './package/dist/webr.mjs';
import {readFile,writeFile} from 'node:fs/promises';
const root='/tmp/spacr-implementation-20261001/f570-reference/';
const r=new WebR({baseUrl:root+'package/dist/',interactive:false});
await r.init();
for(const name of ['perPlateScaling.R','adjustVariance.R','screen.csv','reference.R'])
  await r.FS.writeFile('/'+name,new Uint8Array(await readFile(root+name)));
await r.evalRVoid("source('/reference.R')");
for(const [name,dest] of [['reference.csv','reference.csv'],['session.txt','reference-session.txt']])
  await writeFile(root+dest,await r.FS.readFile('/'+name));
console.log(await r.evalRString('R.version.string'));
console.log('Original cellHTS2 Bscore and plate/experiment scaling executed for2readouts×1536wells.');
await r.close();
