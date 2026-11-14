/* SmartHealth shared client helpers. Loaded on every page as window.SH. */
(function () {
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';

  /** Escape text before putting it into innerHTML. */
  const esc = (s) =>
    String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /** fetch wrapper: adds the CSRF header, parses JSON, throws Error with a readable message. */
  async function api(url, { method = 'POST', json, form } = {}) {
    const opts = { method, headers: { 'X-CSRF-Token': csrf }, credentials: 'same-origin' };
    if (json !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(json);
    } else if (form) {
      opts.body = form;
    }
    let res;
    try {
      res = await fetch(url, opts);
    } catch (_) {
      throw new Error('No connection. Check your network and try again.');
    }
    if (res.status === 401) {
      location.href = '/login';
      throw new Error('Your session expired. Please log in again.');
    }
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) throw new Error((data && data.error) || 'Something went wrong. Please try again.');
    return data;
  }

  /** Short message at the top of the screen. kind: info | success | error */
  function toast(message, kind = 'info') {
    const host = document.getElementById('toasts');
    if (!host) return alert(message);
    const el = document.createElement('div');
    el.className = `toast toast-${kind}`;
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => el.classList.add('toast-out'), kind === 'error' ? 5000 : 2800);
    setTimeout(() => el.remove(), kind === 'error' ? 5400 : 3200);
  }

  /** Disable a button and swap its label while an async job runs. */
  async function busy(btn, label, job) {
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>${esc(label)}</span>`;
    try { return await job(); }
    finally { btn.disabled = false; btn.innerHTML = original; }
  }

  /** Voice recorder that works on Chrome (webm) and Safari (mp4) and survives a quick press-and-release. */
  class Recorder {
    constructor() { this.rec = null; this.stream = null; this.chunks = []; this.stopWanted = false; this.starting = false; this.waiter = null; this.done = null; }
    static mime() {
      const options = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
      return options.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
    }
    get active() { return this.starting || (this.rec && this.rec.state === 'recording'); }
    async start() {
      if (this.active) return;
      this.starting = true; this.stopWanted = false; this.chunks = []; this.done = null;
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        this.starting = false;
        throw new Error('Microphone blocked. Allow microphone access in your browser settings.');
      }
      const mimeType = Recorder.mime();
      this.rec = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
      this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
      this.rec.onstop = () => {
        const type = this.rec.mimeType || 'audio/webm';
        const ext = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
        this.stream.getTracks().forEach((t) => t.stop());
        this.done = { blob: new Blob(this.chunks, { type }), filename: `recording.${ext}` };
        if (this.waiter) this.waiter(this.done);
      };
      this.rec.start();
      this.starting = false;
      if (this.stopWanted) this.rec.stop(); // released while the permission prompt was open
    }
    /** Resolves with { blob, filename }. Safe to call before start() has finished. */
    stop() {
      return new Promise((resolve) => {
        this.waiter = resolve;
        if (this.done) return resolve(this.done);
        if (this.starting) { this.stopWanted = true; return; }
        if (this.rec && this.rec.state === 'recording') this.rec.stop();
      });
    }
  }

  /** Online/offline pill in the header. */
  function watchConnection() {
    const pill = document.getElementById('netPill');
    if (!pill) return;
    const update = () => {
      const on = navigator.onLine;
      pill.dataset.online = on;
      pill.querySelector('span').textContent = on ? 'Online' : 'Offline';
    };
    addEventListener('online', update); addEventListener('offline', update); update();
  }

  /** Register the service worker from the site root so it controls every page. */
  function registerWorker() {
    if (!('serviceWorker' in navigator)) return;
    addEventListener('load', () => navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {}));
  }

  watchConnection();
  registerWorker();
  window.SH = { esc, api, toast, busy, Recorder, csrf };
})();
