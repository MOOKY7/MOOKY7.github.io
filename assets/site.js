document.documentElement.classList.add('js');
const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('#navigation');
menu?.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  nav.classList.toggle('is-open', open);
  menu.textContent = open ? 'Close' : 'Menu';
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && nav?.classList.contains('is-open')) {
    menu.click(); menu.focus();
  }
});
nav?.querySelectorAll('a').forEach(a => a.addEventListener('click', () => {
  nav.classList.remove('is-open'); menu.setAttribute('aria-expanded', 'false'); menu.textContent = 'Menu';
}));
const spaceButtons = [...document.querySelectorAll('[data-space]')];
spaceButtons.forEach(button => button.addEventListener('click', () => {
  const shared = button.dataset.space === 'shared';
  spaceButtons.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
  document.querySelectorAll('.space-point').forEach(p => {
    p.setAttribute('cx', p.dataset[shared ? 'sx' : 'ox']);
    p.setAttribute('cy', p.dataset[shared ? 'sy' : 'oy']);
  });
  document.querySelector('#space-caption').textContent = shared
    ? 'The goal is to preserve useful shared structure while accounting for measurement differences. Whether a model achieves this needs evidence.'
    : 'Different measurement settings can separate samples before a model learns anything useful about the scientific task.';
  document.querySelector('#space-title').textContent = shared ? 'A representation goal' : 'Different measurement settings';
}));
const posts = [...document.querySelectorAll('[data-post]')];
const search = document.querySelector('#note-search');
const filters = [...document.querySelectorAll('[data-filter]')];
let track = 'all';
function applyFilters(updateUrl = true) {
  const query = (search?.value || '').trim().toLowerCase();
  let count = 0;
  for (const post of posts) {
    const visible = (track === 'all' || post.dataset.track === track) && post.dataset.search.includes(query);
    post.hidden = !visible; if (visible) count++;
  }
  filters.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.filter === track)));
  const countNode = document.querySelector('#result-count');
  if (countNode) countNode.textContent = `${count} ${count === 1 ? 'note' : 'notes'}`;
  const empty = document.querySelector('#empty-state');
  if (empty) empty.hidden = count > 0;
  if (updateUrl && search) {
    const url = new URL(location.href);
    track === 'all' ? url.searchParams.delete('track') : url.searchParams.set('track', track);
    query ? url.searchParams.set('q', search.value) : url.searchParams.delete('q');
    history.replaceState(null, '', url);
  }
}
if (search) {
  const params = new URLSearchParams(location.search);
  if (['architecture','practice'].includes(params.get('track'))) track = params.get('track');
  search.value = params.get('q') || '';
  search.addEventListener('input', () => applyFilters());
  filters.forEach(b => b.addEventListener('click', () => {track = b.dataset.filter; applyFilters();}));
  document.querySelector('#clear-search')?.addEventListener('click', () => {search.value = ''; track = 'all'; applyFilters(); search.focus();});
  applyFilters(false);
}
document.querySelectorAll('pre').forEach(pre => {
  const code = pre.querySelector('code');
  if (!code || !navigator.clipboard) return;
  const button = document.createElement('button');
  button.className = 'copy-code'; button.textContent = 'Copy code'; button.type = 'button';
  button.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(code.textContent); button.textContent = 'Copied'; }
    catch { button.textContent = 'Select code to copy'; }
    setTimeout(() => {button.textContent = 'Copy code';}, 2000);
  });
  pre.prepend(button);
});
document.querySelector('[data-print]')?.addEventListener('click', () => window.print());
const progress = document.querySelector('.reading-progress');
if (progress) {
  const update = () => {const d = document.documentElement; const total = d.scrollHeight - innerHeight; progress.style.width = `${total > 0 ? Math.min(100, scrollY / total * 100) : 100}%`;};
  addEventListener('scroll', update, {passive:true}); addEventListener('resize', update); update();
}
