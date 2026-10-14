'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const htmlPath = path.join(__dirname, '..', 'siphon-desktop-dark-opendesign.html');
const html = fs.readFileSync(htmlPath, 'utf8');

function productionFunction(name) {
  const start = html.indexOf(`  function ${name}(`);
  assert.notEqual(start, -1, `could not find production function ${name}`);
  const lineEnd = html.indexOf('\n', start + 1);
  const declarationLine = html.slice(start, lineEnd === -1 ? html.length : lineEnd).trim();
  if (declarationLine.endsWith('}')) return declarationLine;
  const end = html.slice(start).match(/\n  }\r?(?=\n|$)/);
  assert.ok(end, `could not find end of production function ${name}`);
  return html.slice(start, start + end.index + end[0].length).trim();
}

function productionConst(name) {
  const match = html.match(new RegExp(`^  const ${name}=.*;$`, 'm'));
  assert.ok(match, `could not find production constant ${name}`);
  return match[0].trim();
}

function makeElement(id) {
  const classes = new Set();
  const attributes = new Map();
  const listeners = new Map();
  let value = '0';
  let max = ({ settingVram: '4096', settingPinned: '16384', settingCacheRam: '32768' })[id] || '32768';
  const min = id === 'settingVram' ? '512' : '0';
  const range = ['settingVram', 'settingPinned', 'settingCacheRam'].includes(id);
  const element = {
    get value() { return value; },
    set value(next) {
      const text = String(next);
      if (!range || !Number.isFinite(Number(text))) value = text;
      else value = String(Math.min(Number(max), Math.max(Number(min), Number(text))));
    },
    min,
    get max() { return max; },
    set max(next) {
      max = String(next);
      if (range && Number.isFinite(Number(value))) {
        value = String(Math.min(Number(max), Math.max(Number(min), Number(value))));
      }
    },
    disabled: false,
    hidden: false,
    open: false,
    focus() { this.focused = true; },
    showModal() { this.open = true; },
    close() { this.open = false; },
    addEventListener(type, listener) { listeners.set(type, [...(listeners.get(type) || []), listener]); },
    dispatch(type, data = {}) { const event = { target: this, preventDefault() { this.defaultPrevented = true; }, ...data }; for (const listener of listeners.get(type) || []) listener(event); return event; },
    click() { if (!this.disabled) this.dispatch('click'); },
    textContent: '',
    firstChild: { textContent: '' },
    style: {},
    options: [],
    get children() { return this.options; },
    appendChild(child) { this.options.push(child); },
    replaceChildren(...children) { this.options = children; },
    classList: {
      add: (...names) => names.forEach(name => classes.add(name)),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      toggle: (name, force) => {
        const enabled = force === undefined ? !classes.has(name) : force;
        if (enabled) classes.add(name);
        else classes.delete(name);
        return enabled;
      },
      contains: name => classes.has(name),
    },
    setAttribute(name, next) { attributes.set(name, String(next)); },
    getAttribute(name) { return attributes.get(name) ?? null; },
    removeAttribute(name) { attributes.delete(name); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
  };
  return element;
}

function makeModelItem(key, initialState = 'ready') {
  const classes = new Set(['active']);
  const state = makeElement('modelState');
  state.textContent = initialState;
  const title = makeElement('modelTitle');
  const summary = makeElement('modelSummary');
  const caps = [makeElement('modelCapArtifact'), makeElement('modelCapLayers')];
  return {
    dataset: { model: key },
    className: 'model-item',
    disabled: false,
    querySelector(selector) {
      if (selector === 'strong') return title;
      if (selector === '.model-state') return state;
      if (selector === 'small') return summary;
      return null;
    },
    querySelectorAll(selector) { return selector === '.model-cap span' ? caps : []; },
    classList: {
      add: (...names) => names.forEach(name => classes.add(name)),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      toggle(name, force) {
        const enabled = force === undefined ? !classes.has(name) : force;
        if (enabled) classes.add(name);
        else classes.delete(name);
        return enabled;
      },
      contains: name => classes.has(name),
    },
    addEventListener() {},
    remove() {},
    state,
    capText() { return caps.map(cap => cap.textContent); },
  };
}

function createHarness({ ramSafe = 4096, baseOverrides = {} } = {}) {
  const elements = new Map();
  const modelItems = new Map([['test', makeModelItem('test')]]);
  modelItems.get('test').remove = () => modelItems.delete('test');
  const notifications = [];
  const routingDots = makeElement('routingDots');
  const routingMarkup = html.match(/<div class="routing-dots"[^>]*>(.*?)<\/div>/)?.[1] || '';
  routingDots.replaceChildren(...Array.from(routingMarkup.matchAll(/<i\b/g), () => makeElement('i')));
  elements.set('routingDots', routingDots);
  const modelList = makeElement('modelList');
  modelList.querySelector = selector => {
    const match = selector.match(/data-model=["']([^"']+)["']/);
    return match ? modelItems.get(match[1]) || null : null;
  };
  modelList.querySelectorAll = () => [...modelItems.values()];
  modelList.appendChild = item => { item.remove = () => modelItems.delete(item.dataset.model); modelItems.set(item.dataset.model, item); };
  const document = {
    getElementById(id) {
      if (id === 'modelList') return modelList;
      if (!elements.has(id)) elements.set(id, makeElement(id));
      return elements.get(id);
    },
    createElement(tag) { return ['option', 'i'].includes(tag) ? makeElement(tag) : makeModelItem('', 'reading'); },
    querySelector(selector) {
      if (selector === '#models .settings-panel') return document.getElementById('settingsPanel');
      if (selector === '.model-item.active .model-state') {
        return [...modelItems.values()].find(item => item.classList.contains('active'))?.state || null;
      }
      const match = selector.match(/data-model=["']([^"']+)["']/);
      return match ? modelItems.get(match[1]) || null : null;
    },
    querySelectorAll(selector) {
      if (selector === '.routing-dots i') return routingDots.children;
      return selector === '#modelList .model-item' ? [...modelItems.values()] : [];
    },
  };
  const models = {
    test: {
      name: 'Test MoE', arch: 'MoE', meta: 'test model', layers: 40,
      denseWeights: '1 GiB', expertWeights: '2 GiB', nativeContext: 8192,
      contextTokens: 4096, nativeK: 4, expertK: 4, l1Cache: 1024,
      l2Cache: 4096, cacheRam: 4096, residency: '— / —',
      ...baseOverrides,
    },
  };
  document.getElementById('settingKv').options = ['f16', 'tq3_0', 'q8_0'].map(value => ({ value }));
  const select = document.getElementById('modelSelect');
  select.options = [{ value: 'test', remove() { select.options = select.options.filter(option => option !== this); } }];
  select.appendChild = option => { option.remove = () => { select.options = select.options.filter(item => item !== option); }; select.options.push(option); };
  select.replaceChildren = (...options) => { select.options = []; options.forEach(option => select.appendChild(option)); };
  const context = vm.createContext({
    document,
    modelItems,
    CSS: { escape: value => value },
    Number,
    Math,
    String,
    notifications,
    Date,
  });
  const functions = [
    'emptyModel', 'setModelItemStatus', 'bindModelItem', 'applyCatalogue', 'setModel', 'clearDesktopRuntimeTelemetry', 'showEmptyCatalogue', 'setValue', 'clampPct', 'updateSystemCapacity',
    'beginModelRemoval', 'closeModelRemoval', 'confirmModelRemoval', 'handleModelRemoved', 'bindModelRemoval',
    'resourceCheck', 'renderResourceStatus', 'constrainResourceSettings', 'gpuReservation', 'refreshVramBudget',
    'setResourceState', 'applyDesktopTelemetry', 'applyTelemetryEvent', 'memorySettingsLocked', 'setMemoryInput',
    'applyModelInfo', 'handleSettingsApplied', 'renderModelSettings', 'configureExpertRouting',
    'renderRoutingDots', 'syncVisualControls', 'updateRuntimeButtons', 'setPaused', 'setModelRunning',
    'throughputBounds', 'seriesPoints', 'updateThroughputGraphs',
    'cpuExecutionSelected', 'cpuRamReservation', 'renderExecutionControls', 'changeExecutionMode',
    'collectSettings',
  ].filter(name => html.includes(`  function ${name}(`)).map(productionFunction).join('\n');
  const constants = ['fmt', 'contextLabel', 'memoryLabel', 'isNumber', 'setValue', 'clampPct']
    .map(productionConst).join('\n');
  vm.runInContext(`
    const $ = id => document.getElementById(id);
    const models = ${JSON.stringify(models)};
    let key = 'test';
    let catalogueKeys = null;
    let modelRemovalKey = '', modelRemovalId = '', modelRemovalPending = false, modelRemovalSequence = 0;
    let recommendedSettings = null;
    let resourceState = { ramAvailable: 6, vramAvailable: 8, ramSafe: ${ramSafe}, vramSafe: 4096, ramHeadroom: 2048, vramHeadroom: 512, gdsMax: 32 };
    let resourceNeedsSave = false;
    let resourceBlocked = false;
    let modelRunning = false;
    let modelTransition = null;
    let paused = false;
    let activeContextLimit = 0;
    let history = [], prefillHistory = [], lastDecodeSample = null, lastPrefillSample = null;
    const CSS = globalThis.CSS;
    const document = globalThis.document;
    const showToast = () => {};
    const markSettingsDirty = () => {};
    const notifyDesktop = (action, payload) => notifications.push({ action, ...payload });
    const desktopHost = true;
    ${constants}
    ${functions}
    globalThis.harness = {
      applyModelInfo,
      applyCatalogue,
      beginRemoval: () => typeof beginModelRemoval === 'function' && beginModelRemoval(),
      confirmRemoval: () => typeof confirmModelRemoval === 'function' && confirmModelRemoval(),
      cancelRemoval: () => typeof closeModelRemoval === 'function' && closeModelRemoval(),
      removed: event => typeof handleModelRemoved === 'function' && handleModelRemoved(event),
      get removalId() { return modelRemovalId; },
      get dialogOpen() { return $('modelRemoveDialog').open; },
      get removalPending() { return modelRemovalPending; },
      get selected() { return key; },
      get options() { return $('modelSelect').options.map(option => option.value).join(','); },
      get items() { return [...modelItems.keys()].join(','); },
      get notices() { return notifications; },
      get hasTestModel() { return Boolean(models.test); },
      get hidden() { return id => $(id).hidden; },
      handleSettingsApplied,
      renderModelSettings,
      syncVisualControls,
      changeExecutionMode: typeof changeExecutionMode === 'function' ? changeExecutionMode : null,
      checkResources: resourceCheck,
      selectModel: setModel,
      applyTelemetryEvent: typeof applyTelemetryEvent === 'function' ? applyTelemetryEvent : null,
      applyDesktopTelemetry,
      setModelRunning,
      setMax(id, value) { $(id).max = String(value); },
      setInput(id, value) { $(id).value = String(value); },
      setChecked(id, value) { $(id).checked = value; },
      checked(id) { return $(id).checked; },
      constrain: constrainResourceSettings,
      setTransition(value) { modelTransition = value; updateRuntimeButtons(); },
      get model() { return models[key]; },
      get resourceNeedsSave() { return resourceNeedsSave; },
      get resourceBlocked() { return resourceBlocked; },
      get value() { return id => $(id).value; },
      get routingOptions() { return $('settingExpertK').options.map(option => Number(option.value)).join(','); },
      get routingSegments() { return $('routingDots').children.length; },
      get activeRoutingSegments() { return [...$('routingDots').children].filter(dot => dot.classList.contains('active')).length; },
      get routingNodes() { return [...$('routingDots').children]; },
      get max() { return id => Number($(id).max); },
      get disabled() { return id => $(id).disabled; },
      get running() { return modelRunning; },
      get startDisabled() { return $('startModel').disabled; },
      get applyDisabled() { return $('applySettings').disabled; },
      get recommendDisabled() { return $('recommendSettings').disabled; },
      get statusText() { return $('resourceStatus').textContent; },
      get headroomText() { return $('resourceStatus').textContent; },
      get text() { return id => $(id).textContent; },
      get graph() { return id => $(id).getAttribute('points'); },
      get histories() { return [history.join(','), prefillHistory.join(',')].join('|'); },
      get itemStatus() { return modelItems.get('test').state.textContent; },
      get itemSummary() { return modelItems.get('test').querySelector('small').textContent; },
      get itemCaps() { return modelItems.get('test').capText(); },
      get available() { return models[key].available; },
    };
  `, context);
  return context.harness;
}

test('an empty catalogue clears selection and telemetry and disables model actions', () => {
  const app = createHarness();
  app.applyCatalogue([]);
  assert.equal(app.selected, '');
  assert.equal(app.items, '');
  assert.equal(app.text('catalogStorage'), '0.00 GiB used');
  assert.equal(app.text('chatModel'), '—');
  assert.equal(app.startDisabled, true);
  assert.equal(app.applyDisabled, true);
  assert.equal(app.hidden('catalogueEmpty'), false);
  app.applyModelInfo(metadata({}));
  assert.equal(app.selected, '');
  assert.equal(app.hasTestModel, false);
  assert.equal(app.text('catalogCount'), '0 available');
});

test('CPU settings ignore GPU caches and reserve CPU KV against RAM headroom', () => {
  const app = createHarness({ ramSafe: 4096, baseOverrides: {
    config: { execution_mode: 'cpu', cpu_threads: 2 },
    memoryLayout: { kv_bytes_by_type: { f16: [16384] }, draft_kv_bytes_by_type: { f16: [0] }, recurrent_bytes_by_layer: [0] }
  } });
  app.renderModelSettings();
  app.setInput('settingKv', 'f16');
  app.setInput('settingContextLength', '4096');
  app.setInput('settingSpecType', 'none');
  app.setInput('settingCacheRam', '512');
  app.setInput('settingVram', '16384');
  app.setInput('settingPinned', '16384');
  app.constrain('', false);
  assert.equal(app.checkResources().ok, true);
  assert.equal(app.disabled('settingVram'), true);
  assert.equal(app.disabled('settingPinned'), true);
  assert.equal(app.disabled('settingGpuLayers'), true);
  app.setMax('settingCacheRam', '4096');
  app.setInput('settingCacheRam', '4096');
  app.constrain('', false);
  assert.equal(app.checkResources().ok, false);
});

test('CPU and GPU mode switching restores independent settings and is locked during execution', () => {
  const app = createHarness({baseOverrides:{config:{execution_mode:'gpu',gpu_available:true,cache_type:'tq3_0',context:4096,l1_cache:2816,l2_cache:4096,cache_ram:1024,spec_type:'draft-mtp',mode_settings:{cpu:{execution_mode:'cpu',cache_type:'q8_0',context:4096,l1_cache:0,l2_cache:0,cache_ram:512,cpu_threads:2,spec_type:'none',gpu_layers:0}}},l1Cache:2816,cacheRam:1024}});
  app.renderModelSettings();
  app.setChecked('settingUseGpu',false);app.changeExecutionMode();
  assert.equal(app.value('settingKv'),'q8_0');assert.equal(app.value('settingCacheRam'),'512');assert.equal(app.value('settingSpecType'),'none');
  assert.equal(app.disabled('settingVram'),true);assert.equal(app.resourceNeedsSave,true);
  app.setInput('settingCacheRam','1024');
  app.setChecked('settingUseGpu',true);app.changeExecutionMode();
  assert.equal(app.value('settingVram'),'2816');assert.equal(app.value('settingKv'),'tq3_0');assert.equal(app.value('settingSpecType'),'draft-mtp');
  app.setChecked('settingUseGpu',false);app.changeExecutionMode();assert.equal(app.value('settingCacheRam'),'1024');
  app.setModelRunning(true);assert.equal(app.disabled('settingUseGpu'),true);
  app.setChecked('settingUseGpu',true);app.changeExecutionMode();assert.equal(app.checked('settingUseGpu'),false);
});

test('catalogue removal selects the surviving entry and adding after empty restores selection', () => {
  const app = createHarness();
  app.applyCatalogue([{ model_key: 'other', name: 'Other model', available: false }]);
  assert.equal(app.selected, 'other');
  assert.equal(app.items, 'other');
  app.applyCatalogue([]);
  app.applyCatalogue([{ model_key: 'test', name: 'Test MoE', available: true }]);
  assert.equal(app.selected, 'test');
  assert.equal(app.hidden('catalogueEmpty'), true);
  assert.equal(app.hidden('settingsPanel'), false);
  app.applyModelInfo(metadata({}));
  assert.equal(app.text('settingsModelName'), 'Test MoE');
});

test('removal requires confirmation and sends a pinned key only once', () => {
  const app = createHarness();
  app.beginRemoval();
  assert.equal(app.dialogOpen, true);
  assert.equal(app.notices.length, 0);
  assert.equal(app.text('modelRemoveName'), 'Test MoE');
  app.cancelRemoval();
  assert.equal(app.dialogOpen, false);
  assert.equal(app.notices.length, 0);
  app.beginRemoval();
  app.confirmRemoval();
  const id = app.removalId;
  app.confirmRemoval();
  app.cancelRemoval();
  assert.equal(app.dialogOpen, true);
  assert.equal(app.notices.length, 1);
  assert.equal(app.notices[0].action, 'remove_model');
  assert.equal(app.notices[0].model, 'test');
  assert.equal(app.notices[0].removal_id, id);
  assert.equal(app.disabled('removeModel'), true);
});

test('running or transitioning models cannot be removed through the UI', () => {
  const app = createHarness();
  app.setModelRunning(true);
  app.beginRemoval();
  assert.equal(app.dialogOpen, false);
  assert.equal(app.disabled('removeModel'), true);
  app.setModelRunning(false);
  app.setTransition({ kind: 'starting' });
  app.beginRemoval();
  assert.equal(app.dialogOpen, false);
  app.setTransition(null);
  app.beginRemoval();
  assert.equal(app.dialogOpen, true);
});

test('failed removal keeps the entry and allows retry while stale results are ignored', () => {
  const app = createHarness();
  app.beginRemoval();
  app.confirmRemoval();
  app.removed({ model_key: 'test', removal_id: 'old', ok: true });
  assert.equal(app.removalPending, true);
  app.removed({ model_key: 'test', removal_id: app.removalId, ok: false, message: 'Could not save catalogue' });
  assert.equal(app.dialogOpen, true);
  assert.equal(app.removalPending, false);
  assert.equal(app.items, 'test');
  assert.equal(app.text('modelRemoveStatus'), 'Could not save catalogue');
  const oldId = app.removalId;
  app.confirmRemoval();
  assert.notEqual(app.removalId, oldId);
  app.removed({ model_key: 'test', removal_id: oldId, ok: true });
  assert.equal(app.removalPending, true);
  app.removed({ model_key: 'test', removal_id: app.removalId, ok: true });
  assert.equal(app.dialogOpen, false);
  app.applyCatalogue([]);
  assert.equal(app.items, '');
});

function profile(config) {
  return { model_key: 'test', phase: 'profile', available: true, config };
}

function metadata(config) {
  return {
    model_key: 'test', phase: 'metadata', available: true, config,
    name: 'Test MoE', expert_count: 8, expert_used_count: 4,
    layers: 40, native_context: 8192, artifact_gib: 3,
    dense_gib: 1, expert_gib: 2,
  };
}

test('stored auxiliary lookup weights are not shown as dense core weights', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({}), artifact_gib: 87.24870553612709, dense_gib: 31.80630922317505,
    core_gib: 4.984090328216553, lookup_gib: 26.822218894958496,
    expert_gib: 55.43212890625, tensor_gib: 87.23843812942505 });

  assert.equal(app.text('settingsDenseWeights'), '4.98 GiB');
  assert.equal(app.text('settingsLookupWeights'), '26.82 GiB');
  assert.equal(app.text('settingsExpertWeights'), '55.43 GiB');
  assert.equal(app.text('settingsTensorWeights'), '87.24 GiB');
  assert.match(app.text('settingsModelMeta'), /87\.25 GiB local artifact/);
});

test('known absent lookup tables display zero and incomplete breakdowns do not reuse previous values', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({}), core_gib: 1, lookup_gib: 4, tensor_gib: 7 });
  app.applyModelInfo({ ...metadata({}), core_gib: 1, lookup_gib: 0, tensor_gib: 3 });
  assert.equal(app.text('settingsLookupWeights'), '0.00 GiB');

  app.applyModelInfo(metadata({}));
  assert.equal(app.text('settingsDenseWeights'), '—');
  assert.equal(app.text('settingsLookupWeights'), '—');
  assert.equal(app.text('settingsTensorWeights'), '—');
});

test('saved zero prompt cache survives profile and metadata hydration under the safe RAM limit', () => {
  const app = createHarness();

  app.handleSettingsApplied({ type: 'settings_applied', ok: true });
  app.applyModelInfo(profile({ cache_ram: 0, l2_cache: 4096 }));
  app.applyModelInfo(metadata({ cache_ram: 0, l2_cache: 4096 }));

  assert.equal(app.value('settingCacheRam'), '0');
  assert.equal(app.value('settingPinned'), '4096');
  assert.equal(app.resourceNeedsSave, false);
  assert.equal(app.resourceBlocked, false);
  assert.equal(app.startDisabled, false);
});

test('saved 5 GiB config hydrates from phase-less model info even when its GGUF artifact is missing', () => {
  const app = createHarness({ baseOverrides: { cacheRam: 8192, l2Cache: 8192 } });
  const savedProfile = {
    model_key: 'test', available: false,
    config: { cache_ram: 5120, l2_cache: 4096, l1_cache: 2816, context: 32768 },
  };

  app.applyModelInfo(savedProfile);
  app.handleSettingsApplied({ type: 'settings_applied', ok: true });
  app.applyModelInfo(savedProfile);

  assert.equal(app.value('settingCacheRam'), '5120');
  assert.equal(app.value('settingPinned'), '4096');
  assert.equal(app.available, false);
});

test('missing-artifact profile hydration preserves explicit zero allocations', () => {
  const app = createHarness({ baseOverrides: { cacheRam: 8192, l2Cache: 8192, l1Cache: 2816 } });

  app.applyModelInfo({
    model_key: 'test', phase: 'profile', available: false,
    config: { cache_ram: 0, l2_cache: 0, l1_cache: 0 },
  });

  assert.equal(app.value('settingCacheRam'), '0');
  assert.equal(app.value('settingPinned'), '0');
});

test('catalogue entries marked unavailable stay unavailable after the engine is set stopped', () => {
  const app = createHarness();
  app.applyCatalogue([{ model_key: 'test', name: 'Test MoE', available: false }]);
  app.setModelRunning(false);
  assert.equal(app.itemStatus, 'unavailable');
});

test('metadata errors stay unavailable after the engine is set stopped', () => {
  const app = createHarness();
  app.applyModelInfo({
    model_key: 'test', phase: 'metadata', available: true,
    metadata_error: 'GGUF metadata could not be read',
  });
  app.setModelRunning(false);
  assert.equal(app.itemStatus, 'unavailable');
});

test('phase-less metadata errors retain supplied config and remain unavailable', () => {
  const app = createHarness({ baseOverrides: { cacheRam: 8192, l2Cache: 8192 } });
  app.applyModelInfo({
    model_key: 'test', available: true,
    metadata_error: 'GGUF metadata could not be read',
    config: { cache_ram: 5120, l2_cache: 4096 },
  });

  app.setModelRunning(false);

  assert.equal(app.value('settingCacheRam'), '5120');
  assert.equal(app.value('settingPinned'), '4096');
  assert.equal(app.itemStatus, 'unavailable');
});

test('catalogue entries without availability metadata do not become unavailable', () => {
  const app = createHarness();

  app.applyCatalogue([{ model_key: 'test', name: 'Test MoE' }]);

  assert.notEqual(app.itemStatus, 'unavailable');
});

test('idle startup telemetry hides readiness TPS and leaves throughput graphs empty', () => {
  const app = createHarness();
  app.applyDesktopTelemetry({
    running: true, request_state: 'idle', decode_tps: 48000, prefill_tps: 450000,
    decode_sample: 1, prefill_sample: 1, n_prompt: 16384,
    draft_accept_pct: 88, active_experts: 4,
  });

  assert.equal(app.text('decodeValue'), '—');
  assert.equal(app.text('prefillValue'), '—');
  assert.equal(app.text('chatDecode'), '—');
  assert.equal(app.text('chatPrefill'), '—');
  assert.equal(app.text('contextValue'), '—');
  assert.equal(app.text('chatContext'), '—');
  assert.equal(app.text('prefillPromptValue'), '—');
  assert.equal(app.text('chatContextStrip'), '—');
  assert.equal(app.text('acceptValue'), '—');
  assert.equal(app.text('expertsValue'), '—');
  assert.equal(app.graph('decodeGraphLine'), '');
  assert.equal(app.graph('prefillGraphLine'), '');
  assert.equal(app.histories, '|');
});

test('stopped engine rejects completed request metrics but retains hardware telemetry', () => {
  const app = createHarness();
  app.applyDesktopTelemetry({
    running: false, request_state: 'completed', decode_tps: 88000, prefill_tps: 640000,
    decode_sample: 1, prefill_sample: 1, n_prompt: 32768, draft_accept_pct: 91,
    active_experts: 8, gpu_util: 64,
  });

  assert.equal(app.text('decodeValue'), '—');
  assert.equal(app.text('prefillValue'), '—');
  assert.equal(app.text('contextValue'), '—');
  assert.equal(app.text('chatContext'), '—');
  assert.equal(app.text('prefillPromptValue'), '—');
  assert.equal(app.text('chatContextStrip'), '—');
  assert.equal(app.text('acceptValue'), '—');
  assert.equal(app.text('expertsValue'), '—');
  assert.equal(app.graph('decodeGraphLine'), '');
  assert.equal(app.graph('prefillGraphLine'), '');
  assert.equal(app.histories, '|');
  assert.equal(app.text('gpuValue'), '64%');
});

test('active and completed requests retain high measured TPS without an arbitrary cap', () => {
  const app = createHarness();

  app.applyDesktopTelemetry({
    running: true, request_state: 'active', decode_tps: 250000, prefill_tps: 900000,
    decode_sample: 1, prefill_sample: 1,
  });
  assert.match(app.text('decodeValue'), /250,000/);
  assert.equal(app.text('prefillValue'), '900,000');
  assert.notEqual(app.graph('decodeGraphLine'), '');
  assert.notEqual(app.graph('prefillGraphLine'), '');

  app.applyDesktopTelemetry({
    running: true, request_state: 'completed', decode_tps: 350000, prefill_tps: 1200000,
    decode_sample: 2, prefill_sample: 2,
  });
  assert.match(app.text('decodeValue'), /350,000/);
  assert.equal(app.text('prefillValue'), '1,200,000');
});

test('explicit zero L2 and prompt cache values remain valid settings', () => {
  const app = createHarness();

  app.handleSettingsApplied({ type: 'settings_applied', ok: true });
  app.applyModelInfo(profile({ cache_ram: 0, l2_cache: 0 }));
  app.applyModelInfo(metadata({ cache_ram: 0, l2_cache: 0 }));

  assert.equal(app.value('settingCacheRam'), '0');
  assert.equal(app.value('settingPinned'), '0');
  assert.equal(app.resourceNeedsSave, false);
  assert.equal(app.startDisabled, false);
});

test('missing cache values retain model values and missing base values use slider defaults', () => {
  const configured = createHarness({ ramSafe: 8192, baseOverrides: { cacheRam: 3072, l2Cache: 2048 } });
  configured.applyModelInfo(profile({}));
  configured.applyModelInfo(metadata({}));
  assert.equal(configured.value('settingCacheRam'), '3072');
  assert.equal(configured.value('settingPinned'), '2048');

  const absent = createHarness({ baseOverrides: { cacheRam: undefined, l2Cache: undefined } });
  absent.renderModelSettings();
  assert.equal(absent.value('settingCacheRam'), '8192');
  assert.equal(absent.value('settingPinned'), '2048');
});

test('offline over-budget cache allocations remain visible and block Start and Apply', () => {
  const app = createHarness({ ramSafe: 2048 });

  app.handleSettingsApplied({ type: 'settings_applied', ok: true });
  app.applyModelInfo(profile({ cache_ram: 0, l2_cache: 4096 }));
  app.applyModelInfo(metadata({ cache_ram: 0, l2_cache: 4096 }));

  assert.equal(app.value('settingCacheRam'), '0');
  assert.equal(app.value('settingPinned'), '4096');
  assert.equal(app.resourceNeedsSave, false);
  assert.equal(app.resourceBlocked, true);
  assert.equal(app.startDisabled, true);
  assert.equal(app.applyDisabled, true);
});

test('profile hydration expands a stale native range maximum before assigning 2816 MiB', () => {
  const app = createHarness({ baseOverrides: { l1Cache: 2816 } });
  // A previous profile or capacity poll left this native range at 512 MiB.
  app.setMax('settingVram', 512);
  app.setMax('settingPinned', 0);
  app.setMax('settingCacheRam', 0);
  app.applyModelInfo(profile({l1_cache:2816,l2_cache:3072,cache_ram:1024}));
  assert.equal(app.value('settingVram'), '2816');
  assert.equal(app.value('settingPinned'), '3072');
  assert.equal(app.value('settingCacheRam'), '1024');
});

test('passive low-capacity polls preserve all allocations and keep maxima above values', () => {
  const app = createHarness({ baseOverrides: { l1Cache: 2816, l2Cache: 4096, cacheRam: 2048 } });
  app.applyModelInfo(profile({ vram_cache: 2816, l2_cache: 4096, cache_ram: 2048 }));
  app.applyDesktopTelemetry({
    running: false, request_state: 'idle', vram_total_gb: 1, ram_total_gb: 2,
    vram_available_gb: 0.5, ram_available_gb: 1,
    vram_safe_mib: 0, ram_safe_mib: 0,
  });
  app.applyDesktopTelemetry({
    running: false, request_state: 'idle', vram_total_gb: 1, ram_total_gb: 2,
    vram_available_gb: 0.5, ram_available_gb: 1,
    vram_safe_mib: 0, ram_safe_mib: 0,
  });

  assert.deepEqual(
    ['settingVram', 'settingPinned', 'settingCacheRam'].map(id => app.value(id)),
    ['2816', '4096', '2048'],
  );
  assert.ok(app.max('settingVram') >= 2816);
  assert.ok(app.max('settingPinned') >= 4096);
  assert.ok(app.max('settingCacheRam') >= 2048);
  assert.equal(app.resourceBlocked, true);
});

test('first running poll locks cache controls before applying low availability', () => {
  const app = createHarness({ baseOverrides: { l1Cache: 2816, l2Cache: 4096, cacheRam: 2048 } });
  app.applyModelInfo(profile({ vram_cache: 2816, l2_cache: 4096, cache_ram: 2048 }));
  assert.equal(typeof app.applyTelemetryEvent, 'function');
  app.applyTelemetryEvent({
    type: 'telemetry', running: true, request_state: 'idle',
    vram_total_gb: 1, ram_total_gb: 2, vram_available_gb: 0.5,
    ram_available_gb: 1, vram_safe_mib: 0, ram_safe_mib: 0,
  });

  assert.equal(app.running, true);
  for (const id of ['settingVram', 'settingPinned', 'settingCacheRam']) assert.equal(app.disabled(id), true);
  assert.equal(app.recommendDisabled, true);
  assert.deepEqual(
    ['settingVram', 'settingPinned', 'settingCacheRam'].map(id => app.value(id)),
    ['2816', '4096', '2048'],
  );
  assert.ok(app.max('settingVram') >= 2816);
  assert.match(app.headroomText, /RAM|VRAM/);
});

test('loading and stopping transitions ignore stale offline telemetry until transition completion', () => {
  const app = createHarness();
  assert.equal(typeof app.applyTelemetryEvent, 'function');
  app.setTransition({ kind: 'starting' });
  app.applyTelemetryEvent({ type: 'telemetry', running: false, request_state: 'idle' });
  assert.equal(app.running, false);
  for (const id of ['settingVram', 'settingPinned', 'settingCacheRam']) assert.equal(app.disabled(id), true);
  assert.equal(app.recommendDisabled, true);

  app.setTransition(null);
  app.setModelRunning(true);
  app.setTransition({ kind: 'stopping' });
  app.applyTelemetryEvent({ type: 'telemetry', running: false, request_state: 'idle' });
  assert.equal(app.running, true);
  for (const id of ['settingVram', 'settingPinned', 'settingCacheRam']) assert.equal(app.disabled(id), true);

  app.setTransition(null);
  app.applyTelemetryEvent({ type: 'telemetry', running: false, request_state: 'idle' });
  assert.equal(app.running, false);
  assert.equal(app.disabled('settingCacheRam'), false);
});

test('explicit RAM cache edits clamp only the changed slider to shared safe budget', () => {
  const app = createHarness({ ramSafe: 4096, baseOverrides: { l2Cache: 4096, cacheRam: 2048 } });
  app.applyModelInfo(profile({ l2_cache: 4096, cache_ram: 2048 }));
  // Simulate an explicit input event on the prompt-cache control.
  app.setInput('settingCacheRam', '4096');
  app.constrain('settingCacheRam');
  assert.equal(app.value('settingPinned'), '4096');
  assert.equal(app.value('settingCacheRam'), '0');
});

const budgetLayout = {
  layer_weights_bytes: [1073741824, 1073741824], output_weights_bytes: 536870912,
  kv_bytes_by_type: { tq3_0: [224, 224], f16: [1024, 1024] },
  draft_kv_bytes_by_type: { f16: [0, 0] }, recurrent_bytes_by_layer: [0, 0], compute_reserve_mib: 512,
};

test('unmeasured graph workspace never invents a 512 MiB hard limit', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ gpu_layers: 999, context: 65536, cache_type: 'tq3_0' }), native_context: 262144, memory_layout: budgetLayout });
  app.applyTelemetryEvent({ running: false, ram_available_gb: 16, vram_available_gb: 6, vram_free_safe_mib: 5632, memory_layout: budgetLayout });
  assert.equal(app.max('settingVram'), 2944); // 5632 - 2560 - 28, no fictitious workspace
  assert.match(app.text('resourceStatus'), /provisional|unmeasured/i);
});

test('remaining budget below one slider step is shown in MiB, not 0 GB', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ gpu_layers: 999, context: 65536, cache_type: 'tq3_0' }), native_context: 262144, memory_layout: budgetLayout });
  app.applyTelemetryEvent({ running: false, ram_available_gb: 16, vram_available_gb: 3, vram_free_safe_mib: 2618.8618, memory_layout: budgetLayout });
  assert.match(app.text('vramMaxLabel'), /30.9 MiB.*below.*step/i);
});

test('real engine allocation unlocks 2816 MiB and microbatch edits invalidate it', () => {
  const app = createHarness({ baseOverrides: { l1Cache: 2816 } });
  const config = { gpu_layers: 999, context: 65536, batch: 512, microbatch: 128,
    parallel: 1, draft_tokens: 3, expert_k: 4, l1_cache: 2816, cache_type: 'tq3_0',
    draft_cache_type: 'f16', spec_type: 'draft-mtp', draft_model: '', flash_attention: true, rope_scaling: 'auto' };
  const signature = { ...config, vram_cache: 2816 }; delete signature.l1_cache;
  const layout = { ...budgetLayout,
    kv_bytes_by_type: { tq3_0: [4480, 0] }, draft_kv_bytes_by_type: { f16: [0, 2048] },
    runtime_memory: { config: signature, model_bytes: 2135327232, context_bytes: 691273728,
      compute_bytes: 146001152, expert_pool_bytes: 2755490304 } };
  app.applyModelInfo({ ...metadata(config), native_context: 262144, memory_layout: layout });
  // This lightweight DOM fixture starts selects at '0'; real HTML starts at
  // 128 / Auto. Supply the actual selected values for the graph signature.
  app.setInput('settingMicro', '128');
  app.setInput('settingRope', 'auto');
  app.applyTelemetryEvent({ running: false, ram_available_gb: 16, vram_available_gb: 5759/1024, vram_free_safe_mib: 5631, memory_layout: layout });
  assert.equal(app.value('settingVram'), '2816');
  assert.equal(app.max('settingVram'), 2944);
  assert.match(app.text('resourceStatus'), /engine workspace 139.2 MiB/i);
  app.setInput('settingMicro', '256');
  app.constrain();
  assert.doesNotMatch(app.text('resourceStatus'), /engine workspace/);
  assert.equal(app.value('settingVram'), '2816');
});

test('expert slider subtracts model weights and KV buffers from live free VRAM', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ gpu_layers: 999, context: 65536, cache_type: 'tq3_0' }), native_context: 262144, memory_layout: budgetLayout });
  app.applyTelemetryEvent({ running: false, ram_available_gb: 16, vram_available_gb: 6, vram_free_safe_mib: 5632, memory_layout: budgetLayout });
  // 5632 safe free - 2560 weights - 28 KV; unknown compute is not fabricated.
  assert.equal(app.max('settingVram'), 2944);
  assert.match(app.text('vramMaxLabel'), /expert/);
  assert.match(app.text('resourceStatus'), /weights.*KV/i);
});

test('unsaved context and KV precision edits immediately change expert ceiling', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ gpu_layers: 999, context: 65536, cache_type: 'tq3_0' }), native_context: 262144, memory_layout: budgetLayout });
  app.applyTelemetryEvent({ running: false, ram_available_gb: 16, vram_available_gb: 6, vram_free_safe_mib: 5632, memory_layout: budgetLayout });
  app.setInput('settingKv', 'f16');
  app.constrain();
  assert.equal(app.max('settingVram'), 2944);  // 5632 - 2560 - 128
  app.setInput('settingContextLength', '131072');
  app.constrain();
  assert.equal(app.max('settingVram'), 2816);
});

test('running memory is locked and not reserved twice or silently shrunk', () => {
  const app = createHarness({ baseOverrides: { l1Cache: 2816 } });
  app.applyModelInfo({ ...metadata({ l1_cache: 2816, gpu_layers: 999, cache_type: 'tq3_0' }), memory_layout: budgetLayout });
  app.applyTelemetryEvent({ running: true, ram_available_gb: 1, vram_available_gb: 0.5, vram_free_safe_mib: 0, memory_layout: budgetLayout });
  assert.equal(app.value('settingVram'), '2816');
  assert.equal(app.disabled('settingVram'), true);
  assert.match(app.text('vramMaxLabel'), /locked/i);
});

test('selecting an uninspected model never reuses the previous model memory layout', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ gpu_layers: 999, cache_type: 'tq3_0' }), memory_layout: budgetLayout });
  app.applyTelemetryEvent({ running: false, ram_available_gb: 16, vram_available_gb: 6, vram_free_safe_mib: 5632, memory_layout: budgetLayout });
  app.applyCatalogue([{ model_key: 'test', available: true }, { model_key: 'other', available: true }]);
  app.selectModel('other');
  assert.equal(app.text('kvMemorySummary'), 'KV size unavailable');
  assert.match(app.statusText, /memory sizing unavailable/i);
  assert.equal(app.startDisabled, true);
});

test('KV label includes exact main and separate native MTP per-token payloads', () => {
  const app = createHarness();
  const layout = { ...budgetLayout, kv_bytes_by_type: { tq3_0: [4480, 0] }, draft_kv_bytes_by_type: { f16: [0, 2048] } };
  app.applyModelInfo({ ...metadata({ gpu_layers: 999, cache_type: 'tq3_0', spec_type: 'draft-mtp' }), memory_layout: layout });
  app.constrain();
  assert.equal(app.text('kvMemorySummary'), '4.375 KiB/token + 2.000 MTP');
});

test('model buffers exceeding free VRAM block launch even without an enabled expert slider', () => {
  const app = createHarness({ baseOverrides: { arch: 'Dense', nativeK: null } });
  app.applyModelInfo({ ...metadata({ gpu_layers: 999, cache_type: 'tq3_0' }), architecture: 'dense', expert_count: 0, memory_layout: budgetLayout });
  app.applyTelemetryEvent({ running: false, ram_available_gb: 16, vram_available_gb: 2, vram_free_safe_mib: 1536, memory_layout: budgetLayout });
  assert.match(app.statusText, /model.*buffers.*exceed/i);
  assert.equal(app.startDisabled, true);
});

test('dense architecture keeps expert cache controls disabled while offline', () => {
  const app = createHarness({ baseOverrides: { arch: 'Dense', nativeK: null } });
  app.renderModelSettings();
  app.setModelRunning(false);
  assert.equal(app.disabled('settingVram'), true);
  assert.equal(app.disabled('settingPinned'), true);
  assert.equal(app.disabled('settingCacheRam'), false);
});

function makeImportElement(tagName = 'div', setActiveElement = () => {}) {
  let ownText = '';
  const listeners = new Map();
  const element = {
    tagName: tagName.toUpperCase(),
    children: [],
    dataset: {},
    style: {},
    attributes: new Map(),
    disabled: false,
    hidden: false,
    checked: false,
    value: '',
    open: false,
    addEventListener(type, listener) {
      const callbacks = listeners.get(type) || [];
      callbacks.push(listener);
      listeners.set(type, callbacks);
    },
    dispatch(type, event = {}) {
      const dispatched = {
        target: this,
        preventDefault() { this.defaultPrevented = true; },
        ...event,
      };
      for (const callback of listeners.get(type) || []) callback(dispatched);
      return dispatched;
    },
    click() { return this.dispatch('click'); },
    append(...nodes) { this.children.push(...nodes); },
    appendChild(node) { this.children.push(node); return node; },
    replaceChildren(...nodes) { this.children = nodes; ownText = ''; },
    set textContent(value) { ownText = String(value); this.children = []; },
    get textContent() { return ownText + this.children.map(child => child.textContent || '').join(''); },
    set innerHTML(_value) { throw new Error('untrusted import data must not use innerHTML'); },
    setAttribute(name, value) { this.attributes.set(name, String(value)); },
    focus() { setActiveElement(this); },
    showModal() { this.open = true; },
    close() { this.open = false; },
    querySelectorAll(selector) {
      const matches = [];
      const visit = node => {
        if (selector === 'input[type="checkbox"]' && node.tagName === 'INPUT' && node.type === 'checkbox') matches.push(node);
        if (selector === 'input[type="checkbox"]:checked' && node.tagName === 'INPUT' && node.type === 'checkbox' && node.checked) matches.push(node);
        for (const child of node.children || []) visit(child);
      };
      visit(this);
      // Native querySelectorAll returns an iterable NodeList, not an Array.
      return Object.assign({ length: matches.length, forEach: callback => matches.forEach(callback), [Symbol.iterator]: () => matches[Symbol.iterator]() }, matches);
    },
  };
  return element;
}

function createFolderImportHarness() {
  const elements = new Map();
  let activeElement = null;
  const notifications = [];
  const document = {
    get activeElement() { return activeElement; },
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, makeImportElement(id === 'modelFolderDialog' ? 'dialog' : 'div', element => { activeElement = element; }));
      }
      return elements.get(id);
    },
    createElement(tagName) { return makeImportElement(tagName, element => { activeElement = element; }); },
  };
  const names = [
    'setModelFolderStatus', 'updateModelFolderConfirmState', 'renderModelFolderCandidates',
    'beginModelFolderImport', 'closeModelFolderDialog', 'handleModelFolderScan',
    'confirmModelFolderImport', 'handleModelFolderImported', 'bindModelFolderDialog',
  ];
  const extracted = names.map(productionFunction).join('\n');
  const state = html.match(/^  let modelFolderScanId=.*;$/m);
  assert.ok(state, 'could not find production model-folder state declaration');
  const eventStart = html.indexOf('  window.siphonDesktopEvent = event => {');
  const eventEnd = html.indexOf('\n  };', eventStart);
  assert.ok(eventStart !== -1 && eventEnd !== -1, 'could not find desktop event router');
  const router = html.slice(eventStart, eventEnd + '\n  };'.length);
  const context = vm.createContext({ document, window: {}, Date, String, Array, Set, notifyDesktop: (action, payload) => notifications.push(JSON.parse(JSON.stringify({ action, ...payload }))), desktopHost: true });
  vm.runInContext(`
    const $ = id => document.getElementById(id);
    ${state[0]}
    ${extracted}
    ${router}
    bindModelFolderDialog();
    globalThis.folderHarness = {
      begin: () => $('addModel').click(),
      scan: event => window.siphonDesktopEvent({ ...event, type: 'model_folder_scan' }),
      imported: event => window.siphonDesktopEvent({ ...event, type: 'model_folder_imported' }),
      confirm: () => $('modelFolderConfirm').click(),
      close: () => $('modelFolderCancel').click(),
      escape() { return $('modelFolderDialog').dispatch('cancel'); },
      backdrop() { return $('modelFolderDialog').dispatch('click', { target: $('modelFolderDialog') }); },
      get dialogOpen() { return $('modelFolderDialog').open; },
      get scanId() { return modelFolderScanId; },
      get status() { return $('modelFolderStatus').textContent; },
      get candidateText() { return $('modelFolderCandidates').textContent; },
      get resultText() { return $('modelFolderResults').textContent; },
      get checkboxes() { return $('modelFolderCandidates').querySelectorAll('input[type="checkbox"]'); },
      get confirmDisabled() { return $('modelFolderConfirm').disabled; },
      get cancelDisabled() { return $('modelFolderCancel').disabled; },
      get importing() { return modelFolderImporting; },
      get notifications() { return globalThis.notifications; },
      get activeElement() { return document.activeElement; },
      get addButton() { return $('addModel'); },
    };
  `, context);
  context.notifications = notifications;
  return context.folderHarness;
}

test('Add model folder opens the native dialog with a unique scan id and cancellation restores focus', () => {
  const app = createFolderImportHarness();
  app.begin();
  const firstScanId = app.scanId;
  assert.equal(app.dialogOpen, true);
  assert.match(firstScanId, /^model-folder-/);
  assert.deepEqual(app.notifications[0], { action: 'add_model', scan_id: firstScanId });

  app.escape();
  assert.equal(app.dialogOpen, false);
  assert.equal(app.notifications[1].action, 'cancel_model_import');
  assert.equal(app.notifications[1].scan_id, firstScanId);
  assert.equal(app.activeElement, app.addButton);

  app.begin();
  assert.notEqual(app.scanId, firstScanId);
});

test('folder scan displays untrusted filenames as text and never preselects a multi-candidate folder', () => {
  const app = createFolderImportHarness();
  app.begin();
  const scanId = app.scanId;
  const hostileName = '<img src=x onerror=alert(1)> model';
  app.scan({
    scan_id: scanId, phase: 'ready', folder: '/models', models: [
      { id: 'valid', name: hostileName, architecture: 'MoE', path: '/models/a-00001-of-00002.gguf', parts: ['/models/a-00001-of-00002.gguf', '/models/a-00002-of-00002.gguf'], part_count: 2, artifact_gib: 3.25, available: true, error: null, metadata: {} },
      { id: 'invalid', name: 'Broken model', architecture: 'Dense', path: '/models/b.gguf', parts: ['/models/b.gguf'], part_count: 1, artifact_gib: 1, available: false, error: 'missing shard 2', metadata: {} },
    ],
  });

  assert.equal(app.checkboxes.length, 2);
  assert.equal(app.checkboxes[0].checked, false);
  assert.equal(app.checkboxes[1].disabled, true);
  assert.match(app.candidateText, /<img src=x onerror=alert\(1\)> model/);
  assert.match(app.candidateText, /a-00001-of-00002\.gguf/);
  assert.match(app.candidateText, /a-00002-of-00002\.gguf/);
  assert.match(app.candidateText, /missing shard 2/);
  assert.equal(app.confirmDisabled, true);

  app.close();
  app.begin();
  app.scan({
    scan_id: app.scanId, phase: 'ready', folder: '/models',
    models: [{ id: 'only', name: 'One model', architecture: 'Dense', path: '/models/one.gguf', parts: ['/models/one.gguf'], part_count: 1, artifact_gib: 1, available: true, error: null, metadata: {} }],
  });
  assert.equal(app.checkboxes.length, 1);
  assert.equal(app.checkboxes[0].checked, true);
  assert.equal(app.confirmDisabled, false);
});

test('folder scan shows progress, empty, and error states without accepting stale events', () => {
  const app = createFolderImportHarness();
  app.begin();
  const oldScanId = app.scanId;
  app.scan({ scan_id: oldScanId, phase: 'scanning', folder: '/models', models: [], message: 'Reading model files…' });
  assert.match(app.status, /Reading model files/);
  app.scan({ scan_id: oldScanId, phase: 'ready', folder: '/models', models: [], message: 'No runnable models found.' });
  assert.match(app.status, /No runnable models found/);

  app.close();
  app.begin();
  app.scan({ scan_id: app.scanId, phase: 'error', folder: '/private', models: [], message: 'Permission denied' });
  assert.match(app.status, /Permission denied/);
  app.scan({ scan_id: oldScanId, phase: 'ready', models: [{ id: 'stale', available: true }] });
  assert.match(app.status, /Permission denied/);
  assert.equal(app.checkboxes.length, 0);
});

test('selected models submit once, lock cancellation while importing, and show every result until closed', () => {
  const app = createFolderImportHarness();
  app.begin();
  const scanId = app.scanId;
  app.scan({
    scan_id: scanId, phase: 'ready', folder: '/models', models: [
      { id: 'first', name: 'First', architecture: 'MoE', parts: ['/models/first.gguf'], part_count: 1, artifact_gib: 1, available: true, error: null, metadata: {} },
      { id: 'second', name: 'Second', architecture: 'Dense', parts: ['/models/second.gguf'], part_count: 1, artifact_gib: 2, available: true, error: null, metadata: {} },
    ],
  });
  for (const checkbox of app.checkboxes) checkbox.checked = true;
  app.confirm();

  assert.deepEqual(app.notifications.at(-1), { action: 'import_models', scan_id: scanId, model_ids: ['first', 'second'] });
  assert.equal(app.importing, true);
  assert.equal(app.confirmDisabled, true);
  assert.equal(app.cancelDisabled, true);
  app.confirm();
  app.escape();
  app.backdrop();
  assert.equal(app.notifications.filter(event => event.action === 'import_models').length, 1);
  assert.equal(app.dialogOpen, true);

  app.imported({
    scan_id: scanId, message: 'Batch finished', results: [
      { id: 'first', ok: true, model_key: 'first-model', message: 'Added' },
      { id: 'second', ok: false, model_key: null, message: 'Profile could not be saved' },
    ],
  });
  assert.equal(app.importing, false);
  assert.match(app.resultText, /Added/);
  assert.match(app.resultText, /Profile could not be saved/);
  assert.match(app.status, /Batch finished/);
  assert.equal(app.confirmDisabled, true);
  app.confirm();
  assert.equal(app.notifications.filter(event => event.action === 'import_models').length, 1);
  app.close();
  assert.equal(app.dialogOpen, false);
  assert.equal(app.notifications.filter(event => event.action === 'cancel_model_import').length, 1);
});

test('model metadata includes available part counts in its meta and catalogue caps', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({}), part_count: 7 });
  assert.match(app.model.meta, /7 parts/);
  assert.match(app.itemCaps[1], /7 parts/);
});

test('imported native routing counts replace fixed choices without exceeding total experts', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ expert_k: 0 }), expert_count: 4, expert_used_count: 2 });
  assert.equal(app.value('settingExpertK'), '2');
  assert.equal(app.text('routingValue'), 2);
  assert.equal(app.routingOptions, '2,4');
  app.applyModelInfo({ ...metadata({ expert_k: 0 }), expert_count: 512, expert_used_count: 10 });
  assert.equal(app.value('settingExpertK'), '10');
  assert.equal(app.routingOptions, '4,6,10');
  app.applyModelInfo({ ...metadata({ expert_k: 7 }), expert_count: 512, expert_used_count: 10 });
  assert.equal(app.value('settingExpertK'), '7');
  assert.equal(app.routingOptions, '4,6,7,10');
});

test('routing meter uses native active experts, not eight dashes or the total expert pool', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ expert_k: 6 }), expert_count: 512, expert_used_count: 10 });
  assert.equal(app.routingSegments, 10);
  assert.equal(app.activeRoutingSegments, 6);
  const nodes = app.routingNodes;
  app.setInput('settingExpertK', 10);
  app.syncVisualControls();
  assert.equal(app.routingSegments, 10);
  assert.equal(app.activeRoutingSegments, 10);
  assert.deepEqual(app.routingNodes, nodes, 'ordinary updates should reuse the existing segments');
});

test('routing meter resizes on model switching and clears for dense or unknown metadata', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ expert_k: 6 }), expert_count: 512, expert_used_count: 10 });
  app.applyCatalogue([{ model_key: 'test', available: true }, { model_key: 'other', available: true }]);
  app.applyModelInfo({ ...metadata({ expert_k: 4 }), model_key: 'other', expert_count: 256, expert_used_count: 8 });
  app.selectModel('other');
  app.syncVisualControls();
  assert.equal(app.routingSegments, 8);
  assert.equal(app.activeRoutingSegments, 4);
  app.selectModel('test');
  app.syncVisualControls();
  assert.equal(app.routingSegments, 10);
  assert.equal(app.activeRoutingSegments, 6);
  app.applyModelInfo({ ...metadata({}), expert_count: 0, expert_used_count: 0 });
  assert.equal(app.routingSegments, 0);
  assert.equal(app.hidden('routingDots'), true);
  app.applyModelInfo({ ...metadata({}), expert_count: 256, expert_used_count: null });
  assert.equal(app.routingSegments, 0);
});

test('routing meter supports larger native counts and custom routing above native K', () => {
  const app = createHarness();
  app.applyModelInfo({ ...metadata({ expert_k: 6 }), expert_count: 256, expert_used_count: 16 });
  assert.equal(app.routingSegments, 16);
  assert.equal(app.activeRoutingSegments, 6);
  app.applyModelInfo({ ...metadata({ expert_k: 12 }), expert_count: 512, expert_used_count: 10 });
  assert.equal(app.routingSegments, 12);
  assert.equal(app.activeRoutingSegments, 12);
});

test('folder preview treats null sizes as unavailable and rounds known artifact sizes', () => {
  const app = createFolderImportHarness();
  app.begin();
  app.scan({
    scan_id: app.scanId, phase: 'ready', folder: '/models', models: [
      { id: 'unknown', name: 'Unknown size', architecture: 'Dense', parts: ['/models/unknown.gguf'], part_count: 1, artifact_gib: null, available: true, error: null, metadata: {} },
      { id: 'known', name: 'Known size', architecture: 'MoE', parts: ['/models/known.gguf'], part_count: 1, artifact_gib: 3.256789, available: true, error: null, metadata: {} },
    ],
  });
  assert.match(app.candidateText, /size unavailable/);
  assert.match(app.candidateText, /3\.26 GiB/);
  assert.doesNotMatch(app.candidateText, /3\.256789|0 GiB/);
});

test('late scan and import events cannot overwrite an importing or completed batch', () => {
  const app = createFolderImportHarness();
  app.begin();
  const scanId = app.scanId;
  app.scan({ scan_id: scanId, phase: 'ready', models: [{ id: 'one', name: 'One', architecture: 'Dense', parts: ['/models/one.gguf'], part_count: 1, artifact_gib: 2, available: true, error: null, metadata: {} }] });
  app.confirm();
  app.scan({ scan_id: scanId, phase: 'scanning', models: [], message: 'Late scan' });
  assert.equal(app.checkboxes.length, 1);
  assert.doesNotMatch(app.status, /Late scan/);
  app.imported({ scan_id: 'old-scan', results: [{ id: 'wrong', ok: true, message: 'Stale import' }] });
  assert.equal(app.importing, true);

  app.imported({ scan_id: scanId, results: [{ id: 'one', ok: true, model_key: 'one-model', message: 'Added one' }] });
  app.scan({ scan_id: scanId, phase: 'ready', models: [], message: 'Late ready' });
  app.imported({ scan_id: scanId, results: [{ id: 'wrong', ok: false, message: 'Duplicate result' }] });
  assert.match(app.resultText, /Added one/);
  assert.doesNotMatch(app.resultText, /Duplicate result/);
  assert.doesNotMatch(app.status, /Late ready/);
});

test('folder preview renders all eight parts and cancellation creates no import', () => {
  const app = createFolderImportHarness();
  app.begin();
  const scanId = app.scanId;
  const parts = Array.from({ length: 8 }, (_, index) => `/models/model-${index + 1}.gguf`);
  app.scan({ scan_id: scanId, phase: 'ready', models: [{ id: 'eight', name: 'Eight parts', architecture: 'MoE', parts, part_count: 8, artifact_gib: 8, available: true, error: null, metadata: {} }] });
  assert.match(app.candidateText, /8 parts/);
  for (let index = 1; index <= 8; index++) assert.ok(app.candidateText.includes(`model-${index}.gguf`));
  app.backdrop();
  assert.equal(app.dialogOpen, false);
  assert.equal(app.notifications.at(-1).action, 'cancel_model_import');
  assert.equal(app.notifications.filter(event => event.action === 'import_models').length, 0);
  app.imported({ scan_id: scanId, results: [{ id: 'eight', ok: true, message: 'Stale result' }] });
  assert.doesNotMatch(app.resultText, /Stale result/);
});
