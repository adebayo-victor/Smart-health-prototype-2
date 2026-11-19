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


  /**
   * Patient picker: search the facility's register, pick a patient, or add a new one.
   * Also shows the hospital ID with a one-tap way to add or change it, on every page that uses it.
   *   const picker = new SH.PatientPicker(el, { onChange(patient) {} });
   *   picker.value -> selected patient or null;  picker.setPatient(p);  picker.suggest('name')
   */
  class PatientPicker {
    constructor(root, { onChange, label = 'Patient' } = {}) {
      this.root = root; this.onChange = onChange || (() => {}); this.patient = null; this.editing = null; this.timer = null;
      root.innerHTML = `
        <div data-sel class="hidden bg-white rounded-xl border-2 border-brand/40 p-4">
          <div class="flex items-start gap-3">
            <div class="flex-1 min-w-0">
              <p class="text-xs font-semibold text-slate-500">${esc(label)}</p>
              <p data-name class="text-lg font-black leading-tight"></p>
              <p data-meta class="text-sm text-slate-600 mt-0.5"></p>
            </div>
            <div class="flex flex-col items-end gap-1">
              <button type="button" data-change class="min-h-10 px-2 text-sm font-bold text-brand-dark hover:underline">Change</button>
              <button type="button" data-edit class="min-h-10 px-2 text-sm font-bold text-slate-500 hover:underline">Edit details</button>
            </div>
          </div>
          <div class="mt-3 pt-3 border-t border-slate-100">
            <div data-hid-view class="flex items-center gap-2 flex-wrap">
              <span class="text-sm font-semibold text-slate-500">Hospital ID</span>
              <span data-hid class="px-2 py-0.5 rounded bg-slate-100 font-black tracking-wide text-sm"></span>
              <button type="button" data-hid-open class="min-h-10 px-2 text-sm font-bold text-brand-dark hover:underline"></button>
            </div>
            <div data-hid-edit class="hidden flex gap-2">
              <input data-hid-input type="text" maxlength="40" aria-label="Hospital ID" placeholder="Your hospital's ID for this patient"
                     class="flex-1 min-w-0 min-h-11 px-3 rounded-lg border border-slate-300 bg-surface font-bold">
              <button type="button" data-hid-save class="min-h-11 px-4 bg-brand text-white rounded-lg font-bold">Save</button>
              <button type="button" data-hid-cancel class="min-h-11 px-2 text-slate-500 font-bold" aria-label="Cancel">${'\u2715'}</button>
            </div>
          </div>
        </div>

        <div data-search>
          <label class="text-sm font-semibold text-slate-600 block mb-1">${esc(label)} *</label>
          <p class="text-xs text-slate-500 mb-1.5">Search for an existing patient. If they are new, tap "Add" below.</p>
          <input data-q type="search" autocomplete="off" placeholder="Search by name, phone or hospital ID"
                 class="w-full min-h-12 px-3 rounded-xl border border-slate-300 bg-white text-base font-semibold">
          <ul data-results class="mt-2 space-y-1.5"></ul>
          <p data-msg class="hidden text-sm text-slate-500 mt-2"></p>
          <button type="button" data-new class="mt-2 w-full min-h-12 rounded-xl border-2 border-dashed border-slate-300 text-slate-700 font-bold hover:bg-slate-50">+ Add a new patient</button>
        </div>

        <form data-form class="hidden bg-white rounded-xl border border-slate-200 p-4 space-y-3" novalidate>
          <p data-form-title class="font-black"></p>
          ${PatientPicker.field('full_name', 'Full name *', 'text', 'e.g. Amina Yusuf')}
          <div class="grid grid-cols-2 gap-3">
            ${PatientPicker.field('phone', 'Phone number', 'tel', '08012345678')}
            ${PatientPicker.field('age_years', 'Age (years)', 'number', '30')}
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="text-sm font-semibold text-slate-600 block mb-1" for="pp-sex">Sex</label>
              <select data-f="sex" id="pp-sex" class="w-full min-h-12 px-3 rounded-lg border border-slate-300 bg-surface font-semibold">
                <option value="">Not stated</option><option value="F">Female</option><option value="M">Male</option>
              </select>
            </div>
            ${PatientPicker.field('hospital_id', 'Hospital ID', 'text', 'Your facility ID')}
          </div>
          <details class="rounded-lg bg-surface border border-slate-200">
            <summary class="cursor-pointer min-h-11 px-3 flex items-center text-sm font-bold text-slate-600">More details (optional)</summary>
            <div class="p-3 pt-1 space-y-3">
              ${PatientPicker.field('village_lga', 'Village or LGA', 'text', '')}
              <div class="grid grid-cols-2 gap-3">
                ${PatientPicker.field('next_of_kin_name', 'Next of kin', 'text', '')}
                ${PatientPicker.field('next_of_kin_phone', 'Next of kin phone', 'tel', '')}
              </div>
              <div data-preg-wrap class="hidden">
                <label class="text-sm font-semibold text-slate-600 block mb-1" for="pp-preg">Pregnant?</label>
                <select data-f="is_pregnant" id="pp-preg" class="w-full min-h-12 px-3 rounded-lg border border-slate-300 bg-white font-semibold">
                  <option value="">Not stated</option><option value="yes">Yes</option><option value="no">No</option>
                </select>
              </div>
              ${PatientPicker.field('allergies', 'Known allergies', 'text', 'e.g. penicillin')}
            </div>
          </details>
          <p data-form-error role="alert" class="hidden text-sm font-semibold text-danger-dark"></p>
          <div class="flex gap-2">
            <button type="button" data-form-cancel class="min-h-12 px-4 bg-slate-200 rounded-xl font-bold">Cancel</button>
            <button type="submit" data-form-save class="flex-1 min-h-12 bg-brand text-white rounded-xl font-bold hover:bg-brand-dark">Save patient</button>
          </div>
        </form>`;
      this.$ = (sel) => root.querySelector(sel);
      this.bind();
      this.search('');
    }

    static field(name, label, type, ph) {
      const mode = type === 'number' ? 'numeric' : type === 'tel' ? 'tel' : 'text';
      return `<div><label class="text-sm font-semibold text-slate-600 block mb-1">${label}
        <input data-f="${name}" type="${type}" inputmode="${mode}" placeholder="${esc(ph)}" autocomplete="off"
               class="mt-1 w-full min-h-12 px-3 rounded-lg border border-slate-300 bg-surface text-base font-semibold"></label></div>`;
    }

    get value() { return this.patient; }

    bind() {
      this.$('[data-q]').addEventListener('input', (e) => { clearTimeout(this.timer); this.timer = setTimeout(() => this.search(e.target.value), 250); });
      this.$('[data-new]').addEventListener('click', () => this.openForm(null, this.$('[data-q]').value.trim()));
      this.$('[data-change]').addEventListener('click', () => this.clear());
      this.$('[data-edit]').addEventListener('click', () => this.openForm(this.patient));
      this.$('[data-form-cancel]').addEventListener('click', () => this.closeForm());
      this.$('[data-form]').addEventListener('submit', (e) => { e.preventDefault(); this.save(); });
      this.$('[data-f="sex"]').addEventListener('change', () => this.syncPregnancy());
      this.$('[data-results]').addEventListener('click', (e) => {
        const b = e.target.closest('[data-pick]'); if (b) return this.select(this.results[+b.dataset.pick]);
        if (e.target.closest('[data-addq]')) this.openForm(null, this.$('[data-q]').value.trim());
      });
      // hospital ID quick edit
      const hidEdit = (on) => { this.$('[data-hid-edit]').classList.toggle('hidden', !on); this.$('[data-hid-edit]').classList.toggle('flex', on); this.$('[data-hid-view]').classList.toggle('hidden', on); if (on) { this.$('[data-hid-input]').value = this.patient.hospital_id || ''; this.$('[data-hid-input]').focus(); } };
      this.$('[data-hid-open]').addEventListener('click', () => hidEdit(true));
      this.$('[data-hid-cancel]').addEventListener('click', () => hidEdit(false));
      this.$('[data-hid-input]').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this.$('[data-hid-save]').click(); } });
      this.$('[data-hid-save]').addEventListener('click', async (e) => {
        try {
          const p = await busy(e.currentTarget, 'Saving', () => api(`/api/patients/${this.patient.id}`, { method: 'PATCH', json: { hospital_id: this.$('[data-hid-input]').value } }));
          this.patient = p; this.paint(); hidEdit(false); toast('Hospital ID saved.', 'success'); this.onChange(p);
        } catch (err) { toast(err.message, 'error'); }
      });
    }

    async search(q) {
      const list = this.$('[data-results]'), msg = this.$('[data-msg]');
      try {
        this.results = await api('/api/patients?q=' + encodeURIComponent(q.trim()), { method: 'GET' });
      } catch (err) { list.innerHTML = ''; msg.textContent = err.message; return msg.classList.remove('hidden'); }
      list.innerHTML = this.results.map((p, i) => `
        <li><button type="button" data-pick="${i}" class="w-full text-left min-h-14 px-3 py-2 rounded-xl bg-white border border-slate-200 hover:border-brand hover:bg-brand-soft transition">
          <span class="block font-black">${esc(p.full_name)}${p.hospital_id ? ` <span class="ml-1 px-1.5 py-0.5 rounded bg-slate-100 text-xs font-bold">${esc(p.hospital_id)}</span>` : ''}</span>
          <span class="block text-sm text-slate-600">${esc(PatientPicker.meta(p) || 'No other details')}</span></button></li>`).join('');
      if (q.trim()) list.insertAdjacentHTML('beforeend', `<li><button type="button" data-addq class="w-full text-left min-h-14 px-3 py-2 rounded-xl bg-brand-soft border-2 border-brand/40 hover:bg-emerald-100 transition">
          <span class="block font-black text-brand-dark">+ Add &ldquo;${esc(q.trim())}&rdquo; as a new patient</span>
          <span class="block text-sm text-slate-600">Tap here, then fill in their details</span></button></li>`);
      msg.textContent = this.results.length ? '' : (q.trim() ? '' : 'No patients registered yet.');
      msg.classList.toggle('hidden', this.results.length > 0);
    }

    static meta(p) {
      return [p.age_years != null ? p.age_years + ' yrs' : '', p.sex === 'F' ? 'Female' : p.sex === 'M' ? 'Male' : '', p.phone, p.village_lga].filter(Boolean).join(', ');
    }

    /** Pages call this when the worker presses a main button with no patient chosen. Guides them instead of just failing. */
    require() {
      this.root.scrollIntoView({ behavior: 'smooth', block: 'center' });
      const q = this.$('[data-q]').value.trim();
      if (!this.$('[data-form]').classList.contains('hidden')) return toast('Tap "Save patient" to finish adding this patient.', 'error');
      if (q) { this.openForm(null, q); return toast(`Check ${q}'s details, then tap "Save patient".`, 'info'); }
      this.$('[data-q]').focus();
      toast('Search for the patient, or tap "Add a new patient".', 'error');
    }

    select(p) { this.patient = p; this.paint(); this.onChange(p); }
    setPatient(p) { this.patient = p; this.paint(); }
    clear() { this.patient = null; this.$('[data-sel]').classList.add('hidden'); this.$('[data-search]').classList.remove('hidden'); this.$('[data-q]').focus(); this.onChange(null); }
    suggest(name) { if (!name || this.patient) return; this.$('[data-q]').value = name; this.search(name); }

    paint() {
      const p = this.patient;
      this.$('[data-name]').textContent = p.full_name;
      this.$('[data-meta]').textContent = PatientPicker.meta(p) || 'No other details yet';
      const hid = this.$('[data-hid]');
      hid.textContent = p.hospital_id || 'None'; hid.classList.toggle('text-slate-400', !p.hospital_id);
      this.$('[data-hid-open]').textContent = p.hospital_id ? 'Change' : 'Add hospital ID';
      this.$('[data-hid-edit]').classList.add('hidden'); this.$('[data-hid-edit]').classList.remove('flex'); this.$('[data-hid-view]').classList.remove('hidden');
      this.$('[data-sel]').classList.remove('hidden'); this.$('[data-search]').classList.add('hidden'); this.$('[data-form]').classList.add('hidden');
    }

    syncPregnancy() { this.$('[data-preg-wrap]').classList.toggle('hidden', this.$('[data-f="sex"]').value !== 'F'); }

    openForm(p, presetName = '') {
      this.editing = p;
      const f = (n) => this.$(`[data-f="${n}"]`);
      ['full_name', 'phone', 'age_years', 'sex', 'hospital_id', 'village_lga', 'next_of_kin_name', 'next_of_kin_phone', 'allergies'].forEach((n) => { f(n).value = p ? (p[n] ?? '') : ''; });
      f('full_name').value = p ? p.full_name : presetName;
      f('is_pregnant').value = p && p.is_pregnant === true ? 'yes' : p && p.is_pregnant === false ? 'no' : '';
      this.syncPregnancy();
      this.$('[data-form-title]').textContent = p ? 'Edit patient details' : 'New patient';
      this.$('[data-form-error]').classList.add('hidden');
      this.$('[data-form]').classList.remove('hidden');
      this.$('[data-search]').classList.add('hidden'); this.$('[data-sel]').classList.add('hidden');
      f('full_name').focus();
    }

    closeForm() {
      this.$('[data-form]').classList.add('hidden');
      if (this.patient) this.paint(); else this.$('[data-search]').classList.remove('hidden');
    }

    async save(skipCheck = false) {
      const data = {
        full_name: this.$('name').value.trim(),
        phone: this.$('phone').value.trim() || null,
        age_years: this.$('age').value || null,
        sex: this.$('sex').value || null,
        hospital_id: this.$('hospitalId').value.trim() || null,
        village_lga: this.$('village').value.trim() || null,
        next_of_kin_name: this.$('nokName').value.trim() || null,
        next_of_kin_phone: this.$('nokPhone').value.trim() || null,
        is_pregnant: this.$('pregnant').checked,
        allergies: this.$('allergies').value.trim() || null,
      };

      if (!data.full_name) {
        return this.showError('Enter patient full name.');
      }

      // Check duplicates first unless explicitly bypassed
      if (!skipCheck) {
        try {
          const check = await SH.api('/api/patients/check-duplicates', { json: data });
          if (check.matches && check.matches.length > 0) {
            return this.renderDuplicateWarning(check.matches, data);
          }
        } catch (_) {}
      }

      try {
        const p = await SH.busy(this.$('saveBtn'), 'Saving...', () =>
          SH.api('/api/patients', { json: data })
        );
        this.setPatient(p);
        this.closeModal();
        SH.toast('Patient registered successfully.', 'success');
      } catch (err) {
        this.showError(err.message);
      }
    }

    renderDuplicateWarning(matches, pendingData) {
      const box = this.$('modalError') || this.createErrorBox();
      box.className = 'p-3.5 rounded-xl bg-warn-soft border border-warn/40 text-warn-dark text-xs space-y-2';
      box.innerHTML = `
        <p class="font-black uppercase tracking-wide">⚠️ Possible Existing Patient Found</p>
        <div class="space-y-1.5">
          ${matches.map(m => `
            <div class="p-2 bg-white rounded-lg border border-warn/30 flex items-center justify-between text-ink">
              <div>
                <strong class="block">${SH.esc(m.full_name)}</strong>
                <span class="text-slate-500">${m.age_years ? m.age_years + ' yrs · ' : ''}${m.phone || 'No phone'} (${SH.esc(m.match_reason || 'Match')})</span>
              </div>
              <button type="button" class="select-dupe-btn px-2.5 py-1.5 bg-brand text-white rounded-md font-bold text-[11px]" data-id="${m.id}">
                Use This
              </button>
            </div>
          `).join('')}
        </div>
        <button type="button" id="saveAnywayBtn" class="w-full min-h-9 bg-slate-200 text-ink rounded-lg font-bold hover:bg-slate-300 transition text-[11px]">
          No, Save as New Patient Anyway
        </button>
      `;

      box.querySelectorAll('.select-dupe-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const id = e.currentTarget.dataset.id;
          const chosen = matches.find(m => m.id == id);
          if (chosen) {
            this.setPatient(chosen);
            this.closeModal();
            SH.toast(`Selected ${chosen.full_name}.`, 'info');
          }
        });
      });

      box.querySelector('#saveAnywayBtn').addEventListener('click', () => {
        this.save(true); // Bypass check
      });
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
  window.SH = { esc, api, toast, busy, Recorder, PatientPicker, csrf };
})();
