/* SmartHealth shared client helpers with Offline In-App Help & Long-Press */
(function () {
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';

  const esc = (s) =>
    String(s ?? '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[c]));

  async function api(url, { method, json, form } = {}) {
    if (!method) {
      method = (json !== undefined || form) ? 'POST' : 'GET';
    }
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
    try { data = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) throw new Error((data && data.error) || 'Something went wrong. Please try again.');
    return data;
  }

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

  async function busy(btn, label, job) {
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>${esc(label)}</span>`;
    try { return await job(); }
    finally { btn.disabled = false; btn.innerHTML = original; }
  }

  class Recorder {
    constructor() {
      this.rec = null;
      this.stream = null;
      this.chunks = [];
      this.stopWanted = false;
      this.starting = false;
      this.waiter = null;
      this.done = null;
    }
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
      if (this.stopWanted) this.rec.stop();
    }
    stop() {
      return new Promise((resolve) => {
        this.waiter = resolve;
        if (this.done) return resolve(this.done);
        if (this.starting) { this.stopWanted = true; return; }
        if (this.rec && this.rec.state === 'recording') this.rec.stop();
      });
    }
  }

  class PatientPicker {
    constructor(root, { onChange, label = 'Patient' } = {}) {
      this.root = root;
      this.onChange = onChange || (() => {});
      this.patient = null;
      this.editing = null;
      this.timer = null;

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
              <input data-hid-input type="text" maxlength="40" aria-label="Hospital ID" placeholder="Hospital ID"
                class="flex-1 min-w-0 min-h-11 px-3 rounded-lg border border-slate-300 bg-surface font-bold text-sm">
              <button type="button" data-hid-save class="min-h-11 px-4 bg-brand text-white rounded-lg font-bold text-sm">Save</button>
              <button type="button" data-hid-cancel class="min-h-11 px-2 text-slate-500 font-bold" aria-label="Cancel">✕</button>
            </div>
          </div>
        </div>

        <div data-search>
          <label class="text-sm font-semibold text-slate-600 block mb-1">${esc(label)} *</label>
          <p class="text-xs text-slate-500 mb-1.5">Search for an existing patient. If new, tap "Add" below.</p>
          <input data-q type="search" autocomplete="off" placeholder="Search by name, phone or hospital ID"
            class="w-full min-h-12 px-3 rounded-xl border border-slate-300 bg-white text-base font-semibold">
          <ul data-results class="mt-2 space-y-1.5"></ul>
          <p data-msg class="hidden text-sm text-slate-500 mt-2"></p>
          <button type="button" data-new class="mt-2 w-full min-h-12 rounded-xl border-2 border-dashed border-slate-300 text-slate-700 font-bold hover:bg-slate-50 transition">
            + Add a new patient
          </button>
        </div>

        <form data-form class="hidden bg-white rounded-xl border border-slate-200 p-4 space-y-3" novalidate>
          <p data-form-title class="font-black text-sm"></p>
          ${PatientPicker.field('full_name', 'Full name *', 'text', 'e.g. Amina Yusuf')}
          <div class="grid grid-cols-2 gap-3">
            ${PatientPicker.field('phone', 'Phone number', 'tel', '08012345678')}
            ${PatientPicker.field('age_years', 'Age (years)', 'number', '30')}
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="text-sm font-semibold text-slate-600 block mb-1">Sex</label>
              <select data-f="sex" class="w-full min-h-12 px-3 rounded-lg border border-slate-300 bg-surface font-semibold text-sm">
                <option value="">Not stated</option>
                <option value="F">Female</option>
                <option value="M">Male</option>
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
                <label class="text-sm font-semibold text-slate-600 block mb-1">Pregnant?</label>
                <select data-f="is_pregnant" class="w-full min-h-12 px-3 rounded-lg border border-slate-300 bg-white font-semibold text-sm">
                  <option value="">Not stated</option>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>
              ${PatientPicker.field('allergies', 'Known allergies', 'text', 'e.g. penicillin')}
            </div>
          </details>

          <div data-form-error class="hidden text-xs"></div>

          <div class="flex gap-2">
            <button type="button" data-form-cancel class="min-h-12 px-4 bg-slate-200 rounded-xl font-bold text-sm">Cancel</button>
            <button type="submit" data-form-save class="flex-1 min-h-12 bg-brand text-white rounded-xl font-bold text-sm hover:bg-brand-dark transition">Save patient</button>
          </div>
        </form>
      `;

      this.$ = (sel) => root.querySelector(sel);
      this.bind();
      this.search('');
    }

    static field(name, label, type, ph) {
      const mode = type === 'number' ? 'numeric' : type === 'tel' ? 'tel' : 'text';
      return `<div><label class="text-sm font-semibold text-slate-600 block mb-1">${label}
        <input data-f="${name}" type="${type}" inputmode="${mode}" placeholder="${esc(ph)}" autocomplete="off"
          class="mt-1 w-full min-h-12 px-3 rounded-lg border border-slate-300 bg-surface text-base font-semibold">
      </label></div>`;
    }

    get value() { return this.patient; }

    bind() {
      this.$('[data-q]').addEventListener('input', (e) => {
        clearTimeout(this.timer);
        this.timer = setTimeout(() => this.search(e.target.value), 250);
      });
      this.$('[data-new]').addEventListener('click', () => this.openForm(null, this.$('[data-q]').value.trim()));
      this.$('[data-change]').addEventListener('click', () => this.clear());
      this.$('[data-edit]').addEventListener('click', () => this.openForm(this.patient));
      this.$('[data-form-cancel]').addEventListener('click', () => this.closeForm());
      this.$('[data-form]').addEventListener('submit', (e) => { e.preventDefault(); this.save(); });
      this.$('[data-f="sex"]').addEventListener('change', () => this.syncPregnancy());

      const hidOpen = this.$('[data-hid-open]');
      const hidView = this.$('[data-hid-view]');
      const hidEdit = this.$('[data-hid-edit]');
      const hidInput = this.$('[data-hid-input]');
      const hidSave = this.$('[data-hid-save]');
      const hidCancel = this.$('[data-hid-cancel]');

      if (hidOpen) {
        hidOpen.addEventListener('click', () => {
          hidView.classList.add('hidden');
          hidEdit.classList.remove('hidden');
          hidInput.value = this.patient?.hospital_id || '';
          hidInput.focus();
        });
        hidCancel.addEventListener('click', () => {
          hidEdit.classList.add('hidden');
          hidView.classList.remove('hidden');
        });
        hidSave.addEventListener('click', async () => {
          const val = hidInput.value.trim();
          try {
            const updated = await SH.busy(hidSave, 'Saving', () =>
              SH.api(`/api/patients/${this.patient.id}`, {
                method: 'PATCH',
                json: { hospital_id: val }
              })
            );
            this.setPatient(updated);
            hidEdit.classList.add('hidden');
            hidView.classList.remove('hidden');
            SH.toast('Hospital ID updated.', 'success');
          } catch (err) {
            SH.toast(err.message, 'error');
          }
        });
      }
    }

    async search(q) {
      const res = this.$('[data-results]');
      try {
        const list = await SH.api(`/api/patients?q=${encodeURIComponent(q.trim())}`, { method: 'GET' });
        if (!list || list.length === 0) {
          res.innerHTML = q.trim()
            ? `<li class="p-3 bg-brand-soft/40 border border-brand/30 rounded-xl flex items-center justify-between text-xs font-bold text-brand-dark cursor-pointer hover:bg-brand-soft" data-add-named>
                 <span>+ Add "${esc(q.trim())}" as a new patient</span>
                 <span class="text-[11px] underline">Add now</span>
               </li>`
            : '<li class="p-3 text-xs text-slate-400 font-semibold text-center">Type a name, phone, or hospital ID to search</li>';

          const addNamedBtn = res.querySelector('[data-add-named]');
          if (addNamedBtn) {
            addNamedBtn.addEventListener('click', () => this.openForm(null, q.trim()));
          }
          return;
        }

        res.innerHTML = list.map(p => `
          <li class="p-3 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl cursor-pointer flex items-center justify-between transition" data-id="${p.id}">
            <div class="min-w-0">
              <span class="font-bold text-sm text-ink block truncate">${esc(p.full_name)}</span>
              <span class="text-xs text-slate-500">${this.meta(p)}</span>
            </div>
            <span class="text-xs font-bold text-brand px-2.5 py-1 bg-brand-soft rounded-lg">Select</span>
          </li>
        `).join('');

        res.querySelectorAll('[data-id]').forEach(li => {
          li.addEventListener('click', () => {
            const chosen = list.find(p => p.id == li.dataset.id);
            this.select(chosen);
          });
        });
      } catch (_) {
        res.innerHTML = '<li class="p-2 text-xs text-danger-dark">Could not load patients.</li>';
      }
    }

    meta(p) {
      const parts = [];
      if (p.age_years) parts.push(`${p.age_years} yrs`);
      if (p.sex) parts.push(p.sex === 'F' ? 'Female' : 'Male');
      if (p.phone) parts.push(p.phone);
      if (p.hospital_id) parts.push(`ID: ${p.hospital_id}`);
      return parts.join(' · ') || 'No details';
    }

    require() {
      const q = this.$('[data-q]');
      if (q && q.value.trim()) {
        this.openForm(null, q.value.trim());
        SH.toast('Complete and save patient details first.', 'warn');
      } else {
        q?.focus();
        SH.toast('Search or tap "+ Add a new patient" to continue.', 'warn');
      }
    }

    select(p) { this.setPatient(p); }
    setPatient(p) { this.patient = p; this.paint(); this.onChange(p); }
    clear() { this.patient = null; this.paint(); this.onChange(null); this.search(''); }
    suggest(name) { if (name && !this.patient) { this.$('[data-q]').value = name; this.search(name); } }

    paint() {
      const has = !!this.patient;
      this.$('[data-sel]').classList.toggle('hidden', !has);
      this.$('[data-search]').classList.toggle('hidden', has);
      this.$('[data-form]').classList.add('hidden');
      if (has) {
        this.$('[data-name]').textContent = this.patient.full_name;
        this.$('[data-meta]').textContent = this.meta(this.patient);
        const hid = this.patient.hospital_id;
        this.$('[data-hid]').textContent = hid || 'None';
        this.$('[data-hid-open]').textContent = hid ? 'Change' : '+ Add';
      }
    }

    syncPregnancy() {
      const isFemale = this.$('[data-f="sex"]').value === 'F';
      this.$('[data-preg-wrap]').classList.toggle('hidden', !isFemale);
    }

    openForm(existing = null, defaultName = '') {
      this.editing = existing;
      this.$('[data-search]').classList.add('hidden');
      this.$('[data-sel]').classList.add('hidden');
      this.$('[data-form]').classList.remove('hidden');
      this.$('[data-form-title]').textContent = existing ? 'Edit patient details' : 'Register new patient';
      this.$('[data-form-error]').classList.add('hidden');

      const f = (k, v = '') => { const el = this.$(`[data-f="${k}"]`); if (el) el.value = v; };
      if (existing) {
        f('full_name', existing.full_name || '');
        f('phone', existing.phone || '');
        f('age_years', existing.age_years || '');
        f('sex', existing.sex || '');
        f('hospital_id', existing.hospital_id || '');
        f('village_lga', existing.village_lga || '');
        f('next_of_kin_name', existing.next_of_kin_name || '');
        f('next_of_kin_phone', existing.next_of_kin_phone || '');
        f('is_pregnant', existing.is_pregnant === true ? 'yes' : existing.is_pregnant === false ? 'no' : '');
        f('allergies', existing.allergies || '');
      } else {
        this.$('[data-form]').reset();
        f('full_name', defaultName);
      }
      this.syncPregnancy();
      this.$('[data-f="full_name"]').focus();
    }

    closeForm() {
      this.$('[data-form]').classList.add('hidden');
      this.paint();
    }

    async save(skipCheck = false) {
      const f = (k) => this.$(`[data-f="${k}"]`)?.value.trim() || null;
      const isPregVal = this.$('[data-f="is_pregnant"]')?.value;
      const data = {
        full_name: f('full_name'),
        phone: f('phone'),
        age_years: f('age_years'),
        sex: f('sex'),
        hospital_id: f('hospital_id'),
        village_lga: f('village_lga'),
        next_of_kin_name: f('next_of_kin_name'),
        next_of_kin_phone: f('next_of_kin_phone'),
        is_pregnant: isPregVal === 'yes' ? true : isPregVal === 'no' ? false : null,
        allergies: f('allergies'),
      };

      if (!data.full_name) {
        const errEl = this.$('[data-form-error]');
        errEl.textContent = 'Please enter patient full name.';
        errEl.className = 'text-xs text-danger-dark font-bold block';
        errEl.classList.remove('hidden');
        return;
      }

      if (!skipCheck && !this.editing) {
        try {
          const check = await SH.api('/api/patients/check-duplicates', { json: data });
          if (check.matches && check.matches.length > 0) {
            return this.renderDuplicateWarning(check.matches, data);
          }
        } catch (_) {}
      }

      try {
        const saveBtn = this.$('[data-form-save]');
        const endpoint = this.editing ? `/api/patients/${this.editing.id}` : '/api/patients';
        const method = this.editing ? 'PATCH' : 'POST';
        const saved = await SH.busy(saveBtn, 'Saving...', () =>
          SH.api(endpoint, { method, json: data })
        );
        this.setPatient(saved);
        this.closeForm();
        SH.toast('Patient saved successfully.', 'success');
      } catch (err) {
        const errEl = this.$('[data-form-error]');
        errEl.textContent = err.message;
        errEl.className = 'text-xs text-danger-dark font-bold block';
        errEl.classList.remove('hidden');
      }
    }

    renderDuplicateWarning(matches, pendingData) {
      const errEl = this.$('[data-form-error]');
      errEl.className = 'p-3 rounded-xl bg-warn-soft border border-warn/40 text-warn-dark text-xs space-y-2 block';
      errEl.innerHTML = `
        <p class="font-black uppercase tracking-wide">⚠️ Possible Existing Patient Found</p>
        <div class="space-y-1.5">
          ${matches.map(m => `
            <div class="p-2 bg-white rounded-lg border border-warn/30 flex items-center justify-between text-ink">
              <div>
                <strong class="block">${esc(m.full_name)}</strong>
                <span class="text-slate-500">${m.age_years ? m.age_years + ' yrs · ' : ''}${m.phone || 'No phone'} (${esc(m.match_reason || 'Match')})</span>
              </div>
              <button type="button" class="select-dupe-btn px-2.5 py-1.5 bg-brand text-white rounded-md font-bold text-[11px]" data-id="${m.id}">
                Use This
              </button>
            </div>
          `).join('')}
        </div>
        <button type="button" id="saveAnywayBtn" class="w-full min-h-9 bg-slate-200 text-ink rounded-lg font-bold hover:bg-slate-300 transition text-[11px] mt-1">
          Save as New Patient Anyway
        </button>
      `;

      errEl.querySelectorAll('.select-dupe-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const chosen = matches.find(m => m.id == btn.dataset.id);
          if (chosen) {
            this.setPatient(chosen);
            this.closeForm();
            SH.toast(`Selected existing patient: ${chosen.full_name}`, 'info');
          }
        });
      });

      errEl.querySelector('#saveAnywayBtn').addEventListener('click', () => {
        this.save(true);
      });
    }
  }

  // ==========================================
  // OFFLINE IN-APP HELP & LONG-PRESS ENGINE
  // ==========================================
  const HELP_DOCS = {
    translateBtn: {
      title: { en: 'Translate & Extract Data', ha: 'Fassara da Ciro Bayanai', pcm: 'Translate and Collect Info' },
      desc: {
        en: 'Transcribes patient speech, extracts mentioned vitals (BP, SpO2, temp), and prepares physical gestures for verification.',
        ha: 'Yana ɗaukar muryar mara lafiya, ciro ma\'aunin jini da zazzabi, tare da shirya gwajin tabbatarwa.',
        pcm: 'E dey record wetin patient talk, bring out BP and fever readings, and give actions to confirm am.'
      },
      faqs: [
        { q: { en: 'When do I use this?', ha: 'Yaushe zan yi amfani da wannan?', pcm: 'When I go use am?' },
          a: { en: 'Right after recording patient statement during intake.', ha: 'Bayan ka ɗauki muryar mara lafiya.', pcm: 'As you finish record the patient.' } },
        { q: { en: 'What if audio is unclear?', ha: 'Idan muryar ba ta fita da kyau ba fah?', pcm: 'Wetin if voice no clear?' },
          a: { en: 'Tap Clear and record again in a quiet area.', ha: 'Goge sannan ka sake rikodi a wuri mai natsuwa.', pcm: 'Clear am make you record again for quiet place.' } }
      ]
    },
    submitBtn: {
      title: { en: 'Calculate Triage Tier', ha: 'Lissafa Matakin Hatsari (Triage)', pcm: 'Calculate Triage Sickness Level' },
      desc: {
        en: 'Calculates clinical risk (RED = Emergency, YELLOW = Urgent, GREEN = Routine) using SpO2, blood pressure, heart rate, and temperature.',
        ha: 'Yana lissafin matakin hatsari (JA = Gaggawa, RAWAYA = Na kusa, KORE = Na al\'ada) ta amfani da ma\'aunan jini.',
        pcm: 'E go check BP, breath, and temperature know whether the sickness serious well-well (RED) or normal (GREEN).'
      },
      faqs: [
        { q: { en: 'What if result is RED?', ha: 'Menene zanyi idan ya nuna JA?', pcm: 'Wetin I go do if e show RED?' },
          a: { en: 'Immediately stabilise the patient and refer to hospital.', ha: 'Ka bashi taimakon gaggawa sannan ka tura shi asibiti.', pcm: 'Prepare patient quick for referral go hospital.' } }
      ]
    },
    analyzeBtn: {
      title: { en: 'Analyze Symptoms with AI', ha: 'Bincika Alamomin Cuta', pcm: 'Check Symptoms with AI' },
      desc: {
        en: 'Evaluates symptoms and vitals to suggest the most likely conditions in Nigeria (e.g. malaria, pneumonia).',
        ha: 'Yana duba alamomin cuta don ba da shawarar cututtuka da suka dace a Najeriya.',
        pcm: 'E go check the symptoms and vitals suggest which sickness fit cause am like malaria.'
      },
      faqs: [
        { q: { en: 'Can I type my own diagnosis?', ha: 'Zan iya rubuta nawa ra\'ayin?', pcm: 'I fit type my own diagnosis?' },
          a: { en: 'Yes, select "Other / Input Custom Diagnosis" below the AI list.', ha: 'Eh, zabi "Other" ka rubuta naka.', pcm: 'Yes, select "Other" type your own diagnosis.' } }
      ]
    },
    playInstructionBtn: {
      title: { en: 'Play Instructions to Patient', ha: 'Karanta Shawarwari ga Mara Lafiya', pcm: 'Play Instructions for Patient' },
      desc: {
        en: 'Speaks out the prescribed care plan and danger signs in Hausa, Pidgin, or English.',
        ha: 'Yana karanta yadda ake shan magunguna da gargadi a yaren da mara lafiya yake ji.',
        pcm: 'E dey read how to take medicine and warning signs for the language wey patient understand.'
      },
      faqs: [
        { q: { en: 'Can I edit instructions first?', ha: 'Zan iya gyara shawarwarin?', pcm: 'I fit edit am first?' },
          a: { en: 'Yes, edit the text box before pressing play.', ha: 'Eh, rubuta ko gyara kafin ka danna play.', pcm: 'Yes, change the text before you press play.' } }
      ]
    }
  };

  let helpLang = 'en';
  let helpModeActive = false;
  let activeHelpKey = 'submitBtn';

  function openHelp(key) {
    activeHelpKey = key || 'submitBtn';
    renderHelpModal();
    document.getElementById('helpModal')?.classList.remove('hidden');
  }

  function closeHelp() {
    document.getElementById('helpModal')?.classList.add('hidden');
  }

  function setHelpLang(lang) {
    helpLang = lang;
    ['En', 'Ha', 'Pcm'].forEach(l => {
      const btn = document.getElementById(`lang${l}Btn`);
      if (btn) {
        const isMatch = l.toLowerCase() === lang;
        btn.className = isMatch ? 'px-2 py-1 rounded bg-white text-ink shadow-xs' : 'px-2 py-1 rounded text-slate-500';
      }
    });
    renderHelpModal();
  }

  function renderHelpModal() {
    const doc = HELP_DOCS[activeHelpKey] || HELP_DOCS.submitBtn;
    const titleEl = document.getElementById('helpTitle');
    const descEl = document.getElementById('helpDescription');
    const faqList = document.getElementById('helpFaqList');
    if (!titleEl || !descEl || !faqList) return;

    titleEl.textContent = doc.title[helpLang] || doc.title.en;
    descEl.textContent = doc.desc[helpLang] || doc.desc.en;

    faqList.innerHTML = (doc.faqs || []).map(f => `
      <details class="p-2.5 rounded-xl bg-slate-50 border border-slate-200">
        <summary class="font-bold cursor-pointer text-slate-800">${esc(f.q[helpLang] || f.q.en)}</summary>
        <p class="mt-1.5 text-slate-600 font-medium">${esc(f.a[helpLang] || f.a.en)}</p>
      </details>
    `).join('');
  }

  function toggleHelpMode() {
    helpModeActive = !helpModeActive;
    const banner = document.getElementById('helpActiveBanner');
    const btn = document.getElementById('helpToggleBtn');
    if (banner) banner.classList.toggle('hidden', !helpModeActive);
    if (btn) btn.className = helpModeActive ? 'w-8 h-8 rounded-xl bg-warn text-white font-black text-xs flex items-center justify-center shadow-sm' : 'w-8 h-8 rounded-xl border border-slate-300 text-slate-700 font-black text-xs flex items-center justify-center hover:bg-slate-100 transition';
  }

  function setupHelpEvents() {
    document.getElementById('helpToggleBtn')?.addEventListener('click', toggleHelpMode);
    document.getElementById('exitHelpBtn')?.addEventListener('click', toggleHelpMode);

    // Attach long-press & help mode interceptor to main buttons
    const targets = ['submitBtn', 'analyzeBtn', 'translateBtn', 'playInstructionBtn'];
    targets.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;

      let pressTimer = null;
      let isLong = false;

      el.addEventListener('pointerdown', () => {
        isLong = false;
        pressTimer = setTimeout(() => {
          isLong = true;
          openHelp(id);
        }, 550);
      });

      el.addEventListener('pointerup', () => clearTimeout(pressTimer));
      el.addEventListener('pointercancel', () => clearTimeout(pressTimer));

      // Click interceptor when Help Mode is ON
      el.addEventListener('click', (e) => {
        if (helpModeActive || isLong) {
          e.preventDefault();
          e.stopImmediatePropagation();
          openHelp(id);
        }
      }, true);
    });
  }

  function watchConnection() {
    const badge = document.getElementById('connBadge');
    if (!badge) return;
    const set = () => {
      const on = navigator.onLine;
      badge.textContent = on ? 'Online' : 'Offline';
      badge.className = `conn-badge ${on ? 'conn-on' : 'conn-off'}`;
    };
    window.addEventListener('online', set);
    window.addEventListener('offline', set);
    set();
  }

  function registerWorker() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {});
    }
  }

  window.SH = {
    esc,
    api,
    toast,
    busy,
    Recorder,
    PatientPicker,
    openHelp,
    closeHelp,
    setHelpLang
  };

  document.addEventListener('DOMContentLoaded', () => {
    watchConnection();
    registerWorker();
    setupHelpEvents();
  });
})();
