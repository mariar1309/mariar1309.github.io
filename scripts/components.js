// ============================================================
// SHARED COMPONENTS — injected on every page
// ============================================================
(function () {

  // ── Google Fonts (Open Sans) ─────────────────────────────
  if (!document.querySelector('link[href*="fonts.googleapis.com"]')) {
    document.head.append(Object.assign(document.createElement('link'), {
      rel: 'stylesheet',
      href: 'https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600;700&display=swap',
    }));
  }

  // ── Root path resolution ─────────────────────────────────
  const explicit = document.body.getAttribute('data-root');
  const autoRoot = window.location.pathname.replace(/\\/g, '/').includes('/pages/') ? '../' : '';
  const root = explicit !== null ? explicit : autoRoot;

  // ── Nav link definitions ──────────────────────────────────
  const NAV_LINKS = [
    { label: 'About',         href: `${root}pages/about.html` },
    { label: 'Projects',      href: `${root}pages/projects.html` },
    { label: 'Resume',        href: `${root}pages/resume.html` },
    { label: 'Publications',  href: `${root}pages/publications.html` },
    { label: 'Book Reviews',  href: `${root}pages/book-reviews.html` },
    { label: 'Contact',       href: `${root}pages/contact.html` },
  ];

  // ── Active-link helper ────────────────────────────────────
  function refreshNavActive() {
    const curPage   = window.location.pathname.replace(/\\/g, '/').split('/').pop() || 'index.html';
    const curFilter = new URLSearchParams(window.location.search).get('filter') || '';
    document.querySelectorAll('.main-nav a').forEach((a) => {
      try {
        const u       = new URL(a.href);
        const aPage   = u.pathname.replace(/\\/g, '/').split('/').pop();
        const aFilter = u.searchParams.get('filter') || '';
        a.classList.toggle('active', aPage === curPage && aFilter === curFilter);
      } catch (_) {
        a.classList.remove('active');
      }
    });
  }

  window._refreshNavActive = refreshNavActive;

  // ── Build header ─────────────────────────────────────────
  const header = document.querySelector('.site-header');
  if (!header) return;

  const linksHTML = NAV_LINKS.map(({ label, href }) =>
    `<li><a href="${href}">${label}</a></li>`
  ).join('');

  header.innerHTML = `
    <div class="header-inner">
      <a class="logo" href="${root}index.html">🚀 Maria Rufova</a>
      <span class="header-email">mariarufova@berkeley.edu</span>
      <nav class="main-nav"><ul>${linksHTML}</ul></nav>
      <button class="nav-toggle" aria-label="Toggle menu">&#9776;</button>
    </div>`;

  header.querySelector('.nav-toggle').addEventListener('click', () => {
    header.querySelector('.main-nav').classList.toggle('open');
  });

  refreshNavActive();
})();
