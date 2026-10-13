import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const folder=path.resolve(ROOT,process.argv[2]||'dist');
const files=[];
function walk(dir){for(const entry of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,entry.name);entry.isDirectory()?walk(p):files.push(p);}}
walk(folder);
const errors=[];let links=0;
const htmlFiles=files.filter(p=>p.endsWith('.html'));
for(const file of htmlFiles){
 const html=fs.readFileSync(file,'utf8');
 if(!/<html lang="en"/.test(html))errors.push(`${file} has no language`);
 if(!/<title>[^<]+<\/title>/.test(html))errors.push(`${file} has no title`);
 const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
 if(new Set(ids).size!==ids.length)errors.push(`${file} has duplicate IDs`);
 for(const match of html.matchAll(/\b(?:href|src)="([^"]*)"/g)){
  const link=match[1].replace(/&amp;/g,'&');
  if(/^(https?:|mailto:|data:)/.test(link))continue;
  if(!link||link==='#'){errors.push(`${file} contains an empty link`);continue;}
  links++;
  const [withQuery,hash]=link.split('#');const pathname=decodeURIComponent(withQuery.split('?')[0]);
  let target=pathname?(pathname.startsWith('/')?path.join(folder,pathname):path.resolve(path.dirname(file),pathname)):file;
  if(fs.existsSync(target)&&fs.statSync(target).isDirectory())target=path.join(target,'index.html');
  if(!fs.existsSync(target)){errors.push(`${file} has missing target ${link}`);continue;}
  if(hash && target.endsWith('.html')){
   const destination=fs.readFileSync(target,'utf8');
   if(!destination.includes(`id="${decodeURIComponent(hash)}"`))errors.push(`${file} has missing anchor ${link}`);
  }
 }
 if(/you@example\.com|Paper title|Your Department|lorem ipsum/i.test(html))errors.push(`${file} contains placeholder profile content`);
 if(!html.includes('http-equiv="refresh"')){
  if(!html.includes('property="og:image"'))errors.push(`${file} has no social image`);
  if(!html.includes('id="main"'))errors.push(`${file} has no main target`);
 }
}
for(const p of ['feed.xml','feeds/architecture.xml','feeds/practice.xml','sitemap.xml','robots.txt','assets/social-card.png'])if(!fs.existsSync(path.join(folder,p)))errors.push(`Missing ${p}`);
if(errors.length){console.error(errors.join('\n'));process.exit(1);}
console.log(JSON.stringify({htmlPages:htmlFiles.length,internalLinksChecked:links,errors:0}));
