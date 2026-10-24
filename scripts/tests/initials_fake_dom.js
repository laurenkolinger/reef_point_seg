/* Node driver for the initials-seed suites: runs one browser script against a
   fake DOM and reports what the page's initials fields hold.

   Usage: node initials_fake_dom.js '<json>' where json is
     { file, search, store, throws, ids, type: [[id, text, caret]] }
   file    absolute path of the script to run (an .html file runs its inline
           <script> blocks that carry no src attribute)
   search  the page's location.search
   store   initial localStorage contents; throws makes getItem throw
   ids     element ids whose value is reported after load (and after typing)
   type    for each [id, text, caret]: set the field, fire its input listeners
   Prints one JSON line: { values, writes, carets }. No emoji, no em dashes. */
'use strict';
const fs = require('fs');
const vm = require('vm');

function makeStub(id) {
  const base = {
    id, value: '', selectionStart: 0, selectionEnd: 0, _l: {}, hidden: false,
    style: {}, dataset: {}, checked: false, textContent: '', innerHTML: '',
    className: '',
    addEventListener(t, f) { (this._l[t] = this._l[t] || []).push(f); },
    setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; },
  };
  const fn = function () { return proxy; };
  const proxy = new Proxy(fn, {
    get(_, k) {
      if (k in base) return base[k];
      if (k === Symbol.toPrimitive) return () => '';
      if (k === 'then') return undefined;
      return proxy;
    },
    set(_, k, v) { base[k] = v; return true; },
    apply() { return proxy; },
    has() { return true; },
  });
  proxy.__base = base;
  return proxy;
}

function buildContext(spec, els, writes) {
  const store = Object.assign({}, spec.store || {});
  const document = {
    getElementById(id) { return els[id] || (els[id] = makeStub(id)); },
    querySelector() { return makeStub('q'); },
    querySelectorAll() { return []; },
    createElement() { return makeStub('c'); },
    addEventListener() {},
    body: makeStub('body'),
    documentElement: makeStub('html'),
  };
  const localStorage = {
    getItem(k) {
      if (spec.throws) throw new Error('storage blocked');
      return k in store ? store[k] : null;
    },
    setItem(k, v) { writes.push([k, v]); store[k] = v; },
  };
  const ctx = vm.createContext({
    document, localStorage, URLSearchParams, console, Promise, String, RegExp,
    Math, Number, Object, Array, JSON, Date, Error, isFinite,
    setTimeout() { return 0; }, clearTimeout() {}, setInterval() { return 0; },
    location: { search: spec.search || '', hostname: 'h', href: '' },
    fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) }),
    open() {}, confirm() { return true; }, addEventListener() {},
    matchMedia: () => ({ matches: false, addEventListener() {} }),
  });
  ctx.window = ctx;
  return ctx;
}

function sources(file) {
  const text = fs.readFileSync(file, 'utf8');
  if (!file.endsWith('.html')) return [text];
  const out = [];
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g;
  let m;
  while ((m = re.exec(text))) out.push(m[1]);
  return out;
}

function main() {
  const spec = JSON.parse(process.argv[2]);
  const els = {};
  const writes = [];
  const ctx = buildContext(spec, els, writes);
  sources(spec.file).forEach((src) => vm.runInContext(src, ctx, { filename: spec.file }));
  const carets = {};
  (spec.type || []).forEach(([id, text, caret]) => {
    const el = ctx.document.getElementById(id);
    el.value = text;
    el.selectionStart = caret;
    ((el.__base._l.input) || []).forEach((f) => f.call(el, { target: el }));
    carets[id] = el.selectionStart;
  });
  const values = {};
  (spec.ids || []).forEach((id) => { values[id] = ctx.document.getElementById(id).value; });
  process.stdout.write(JSON.stringify({ values, writes, carets }) + '\n');
}
main();
