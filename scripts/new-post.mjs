import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [track,slug,...words]=process.argv.slice(2);
if(!['architecture','practice'].includes(track)||!slug||!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)||!words.length){
 console.error('Usage\n  npm run new -- architecture my-topic "My title"\n  npm run new -- practice week-two "Week 02 · A tested implementation"');process.exit(1);
}
const today=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
const target=path.join(root,'content/posts',`${today}-${slug}.md`);
if(fs.existsSync(target))throw new Error('A file with this name already exists.');
const metadata={title:words.join(' '),date:today,slug,track,kind:track==='practice'?'Weekly review':'Technical note',summary:'Write one sentence that states the question and what this note adds.',tags:[],published:false};
const body=track==='practice'?'## The goal\n\nState the question for the week.\n\n## What I completed\n\nLink the code, checks, and results. Keep planned work separate.\n\n## What failed\n\nDescribe one error and how you diagnosed it.\n\n## What I learned\n\nExplain what changed in your understanding.\n\n## Next week\n\nChoose a small set of concrete deliverables.\n':'## The question\n\nExplain the problem in plain language.\n\n## The mechanism\n\nDefine the inputs, objective, and outputs.\n\n## A small implementation\n\nInclude code or link to a reproducible example.\n\n## What I checked\n\nExplain the checks, controls, and limits.\n\n## Open questions\n\nState what remains uncertain.\n\n## Sources\n\nLink the original papers and official documentation.\n';
fs.writeFileSync(target,`---\n${JSON.stringify(metadata,null,2)}\n---\n\n${body}`,{flag:'wx'});
console.log(`Created ${path.relative(root,target)}\nThis is a draft. Set published to true when ready.`);
