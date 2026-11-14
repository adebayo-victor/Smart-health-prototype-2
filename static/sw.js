/* SmartHealth service worker. Served from /sw.js so its scope is the whole site. */
const VERSION = 'v2';
const STATIC_CACHE = `smarthealth-static-${VERSION}`;
const PAGE_CACHE = `smarthealth-pages-${VERSION}`;
const TAILWIND = 'https://cdn.tailwindcss.com';

const PRECACHE = [
  '/offline',
  '/static/app.js',
  '/static/app.css',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(STATIC_CACHE);
    await cache.addAll(PRECACHE);
    // Styling must work offline. The CDN script is cross-origin, so fetch it opaquely; ignore failure.
    try { await cache.add(new Request(TAILWIND, { mode: 'no-cors' })); } catch (_) {}
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keep = [STATIC_CACHE, PAGE_CACHE];
    for (const key of await caches.keys()) if (!keep.includes(key)) await caches.delete(key);
    await self.clients.claim();
  })());
});

// The login page asks us to forget cached pages so a shared phone never shows the last user's records.
self.addEventListener('message', (event) => {
  if (event.data === 'clear-pages') event.waitUntil(caches.delete(PAGE_CACHE));
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Never cache API calls or service-worker/health endpoints.
  if (url.origin === location.origin && (url.pathname.startsWith('/api/') || url.pathname === '/sw.js' || url.pathname === '/healthz')) return;

  if (req.mode === 'navigate') {
    event.respondWith(networkFirstPage(req, url));
    return;
  }

  if ((url.origin === location.origin && url.pathname.startsWith('/static/')) || url.href.startsWith(TAILWIND)) {
    event.respondWith(staleWhileRevalidate(req));
  }
});

async function networkFirstPage(req, url) {
  try {
    const res = await fetch(req);
    // Cache only real, same-origin pages: not redirects to login, not the login page itself.
    if (res.ok && !res.redirected && url.pathname !== '/login') {
      const copy = res.clone();
      caches.open(PAGE_CACHE).then((c) => c.put(req, copy));
    }
    return res;
  } catch (_) {
    return (await caches.match(req)) || (await caches.match('/offline')) || Response.error();
  }
}

async function staleWhileRevalidate(req) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(req);
  const refresh = fetch(req).then((res) => {
    if (res && (res.ok || res.type === 'opaque')) cache.put(req, res.clone());
    return res;
  }).catch(() => cached);
  return cached || refresh;
}
