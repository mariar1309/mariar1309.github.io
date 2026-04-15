// ============================================================
// MAIN — runs on every page after components.js + data files
// ============================================================

const _inPages = window.location.pathname.replace(/\\/g, '/').includes('/pages/');
const _root = (document.body.getAttribute('data-root') !== null)
  ? document.body.getAttribute('data-root')
  : (_inPages ? '../' : '');

// ← Back button
document.querySelectorAll('.project-back a').forEach((a) => {
  a.href = _root + 'index.html';
  if (document.referrer) {
    a.addEventListener('click', (e) => { e.preventDefault(); window.history.back(); });
  }
});

// Sticky back bar
(function initStickyBack() {
  const backEl = document.querySelector('.project-back');
  if (!backEl) return;
  const srcLink = backEl.querySelector('a');
  if (!srcLink) return;

  const bar = document.createElement('div');
  bar.className = 'sticky-back';
  const barLink = document.createElement('a');
  barLink.textContent = srcLink.textContent;
  barLink.href = srcLink.href;
  barLink.addEventListener('click', (e) => {
    if (document.referrer) { e.preventDefault(); window.history.back(); }
  });
  bar.appendChild(barLink);
  document.body.appendChild(bar);

  const obs = new IntersectionObserver(([entry]) => {
    bar.classList.toggle('sticky-back--visible', !entry.isIntersecting);
  }, { threshold: 0 });
  obs.observe(backEl);
})();

// ============================================================
// Merge project arrays from whichever _index.js files were loaded
// ============================================================
const PROJECTS = [
  ...(typeof CV_PROJECTS !== 'undefined' ? CV_PROJECTS : []),
].sort((a, b) => (b.year || 0) - (a.year || 0));

// ============================================================
// CARD RENDERER
// ============================================================
function renderCards(projects, containerId) {
  const grid = document.getElementById(containerId);
  if (!grid) return;
  if (!projects.length) {
    grid.innerHTML = `<p style="color:var(--text-muted);padding:8px 0">No projects yet — check back soon!</p>`;
    return;
  }
  grid.innerHTML = projects.map((p) => `
    <article class="card" data-id="${p.id}" tabindex="0" role="link" aria-label="${p.title}">
      <div class="card-thumb">
        <img src="${_root + p.image}" alt="${p.title}" loading="lazy" />
      </div>
      <div class="card-body">
        <div class="card-category">${p.category}</div>
        <h3 class="card-title">${p.title}</h3>
        <p class="card-desc">${p.description}</p>
        <div class="card-tags">${p.tags.map((t) => `<span class="tag">${t}</span>`).join('')}</div>
      </div>
    </article>`).join('');

  grid.querySelectorAll('.card').forEach((card) => {
    const project = PROJECTS.find((p) => p.id === card.dataset.id);
    const go = () => { window.location.href = _root + project.page; };
    card.addEventListener('click', go);
    card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') go(); });
  });
}

// ============================================================
// PUBLICATION LIST RENDERER
// ============================================================
function renderPublications(containerId) {
  if (typeof PUBLICATIONS === 'undefined') return;
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!PUBLICATIONS.length) {
    el.innerHTML = `<p style="color:var(--text-muted);padding:24px 0;font-style:italic">Coming soon.</p>`;
    return;
  }
  el.innerHTML = PUBLICATIONS.map((p) => `
    <div class="pub-item">
      <div class="pub-year">${p.year}</div>
      <div>
        <div class="pub-title">${p.title}</div>
        <div class="pub-meta">${p.authors} — ${p.venue}</div>
        <div class="pub-links">
          ${p.links.map((l) =>
            `<a class="pub-link${l.style === 'orange' ? ' orange' : ''}" href="${l.url}"${/^https?:\/\//.test(l.url) ? ' target="_blank" rel="noopener"' : ''}>${l.label}</a>`
          ).join('')}
        </div>
      </div>
    </div>`).join('');
}

// ============================================================
// HOME PAGE — featured projects
// ============================================================
if (document.getElementById('home-grid')) {
  renderCards(PROJECTS.filter((p) => p.featured), 'home-grid');
}

// ============================================================
// PROJECTS PAGE — all projects with filter bar
// ============================================================
if (document.getElementById('projects-grid')) {
  const bar = document.getElementById('projects-filter');

  function applyProjectFilter(slug) {
    if (bar) bar.querySelectorAll('.filter-btn').forEach((b) =>
      b.classList.toggle('active', b.dataset.filter === slug)
    );
    const url = new URL(window.location.href);
    if (slug === 'all') url.searchParams.delete('filter');
    else url.searchParams.set('filter', slug);
    window.history.replaceState({}, '', url.toString());
    renderCards(
      slug === 'all' ? PROJECTS : PROJECTS.filter((p) => p.categorySlug === slug),
      'projects-grid'
    );
  }

  applyProjectFilter(new URLSearchParams(window.location.search).get('filter') || 'all');

  if (bar) bar.querySelectorAll('.filter-btn').forEach((btn) => {
    btn.addEventListener('click', () => applyProjectFilter(btn.dataset.filter));
  });
}

// ============================================================
// PUBLICATIONS PAGE
// ============================================================
if (document.getElementById('pub-list')) renderPublications('pub-list');

// ============================================================
// LIGHTBOX — project page images
// ============================================================
(function initLightbox() {
  const navImgs = Array.from(document.querySelectorAll('.project-figure img'));
  const heroImg = document.querySelector('.project-hero-img img');
  if (!navImgs.length && !heroImg) return;

  const overlay = document.createElement('div');
  overlay.className = 'lb-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.innerHTML = `
    <div class="lb-box">
      <img class="lb-img" src="" alt="" />
      <div class="lb-caption"></div>
    </div>
    <button class="lb-close" aria-label="Close">\u2715</button>
    <button class="lb-nav lb-prev" aria-label="Previous">\u2039</button>
    <button class="lb-nav lb-next" aria-label="Next">\u203a</button>`;
  document.body.appendChild(overlay);

  const lbImg     = overlay.querySelector('.lb-img');
  const lbCaption = overlay.querySelector('.lb-caption');
  const lbClose   = overlay.querySelector('.lb-close');
  const lbPrev    = overlay.querySelector('.lb-prev');
  const lbNext    = overlay.querySelector('.lb-next');

  let currentIdx = 0;
  let isHero = false;

  function setImage(img) {
    lbImg.src = img.src;
    lbImg.alt = img.alt || '';
    const fig = img.closest('figure');
    const cap = fig ? fig.querySelector('figcaption') : null;
    lbCaption.textContent = cap ? cap.textContent.trim() : '';
    lbCaption.style.display = lbCaption.textContent ? '' : 'none';
  }

  function open(idx, hero) {
    isHero = !!hero;
    currentIdx = isHero ? 0 : idx;
    setImage(isHero ? heroImg : navImgs[currentIdx]);
    const showNav = !isHero && navImgs.length > 1;
    lbPrev.style.display = showNav ? '' : 'none';
    lbNext.style.display = showNav ? '' : 'none';
    overlay.style.display = 'flex';
    requestAnimationFrame(() => overlay.classList.add('lb-visible'));
    document.body.style.overflow = 'hidden';
  }

  function close() {
    overlay.classList.remove('lb-visible');
    setTimeout(() => { overlay.style.display = 'none'; lbImg.src = ''; }, 200);
    document.body.style.overflow = '';
  }

  function nav(dir) {
    if (isHero) return;
    currentIdx = (currentIdx + dir + navImgs.length) % navImgs.length;
    setImage(navImgs[currentIdx]);
  }

  navImgs.forEach((img, i) => {
    img.addEventListener('click', (e) => { e.stopPropagation(); open(i, false); });
  });
  if (heroImg) {
    heroImg.addEventListener('click', (e) => { e.stopPropagation(); open(0, true); });
  }

  lbClose.addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  lbPrev.addEventListener('click', (e) => { e.stopPropagation(); nav(-1); });
  lbNext.addEventListener('click', (e) => { e.stopPropagation(); nav(1); });

  document.addEventListener('keydown', (e) => {
    if (!overlay.classList.contains('lb-visible')) return;
    if (e.key === 'Escape')    close();
    if (e.key === 'ArrowLeft') nav(-1);
    if (e.key === 'ArrowRight') nav(1);
  });
})();
