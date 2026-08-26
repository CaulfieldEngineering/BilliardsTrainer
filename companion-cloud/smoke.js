// Runtime smoke gate for the phone page - run by deploy.py BEFORE deploying.
//
// Born from a three-day outage: a temporal-dead-zone ReferenceError at the
// top level of app.js killed everything after it (the sessions list showed
// "Loading" forever), while `node --check` passed happily - syntax checks
// never execute anything. This harness EXECUTES app.js under stubs and
// fails the deploy unless the script (a) evaluates to its final statement
// and (b) actually issues the /api/sessions fetch.
//
//   node companion-cloud/smoke.js
//
// Exit 0 = safe to deploy. Anything else = the page is broken for everyone.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function el() {
  const e = {
    style: {}, dataset: {}, children: [],
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    innerHTML: "", textContent: "", value: "",
    setAttribute() {}, getAttribute: () => null, removeAttribute() {},
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { cs.forEach(c => this.children.push(c)); },
    addEventListener() {}, removeEventListener() {},
    querySelector: () => null, querySelectorAll: () => [],
    getContext: () => new Proxy({}, { get: () => () => {} }),
    requestVideoFrameCallback() {}, play: () => Promise.resolve(), pause() {},
    focus() {}, blur() {}, click() {}, remove() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
  };
  return new Proxy(e, {
    get: (t, k) => (k in t ? t[k] : (typeof k === "string" ? undefined : t[k])),
    set: (t, k, v) => { t[k] = v; return true; },
  });
}

const calls = { fetches: [], errors: [] };
const storage = new Map();

const sandbox = {
  console, setTimeout, clearTimeout, setInterval, clearInterval,
  performance: { now: () => Date.now() },
  fetch: (url) => {
    calls.fetches.push(String(url));
    // resolve statics with plausible bodies; APIs with a tiny valid shape
    const body = String(url).includes("whatsnew") ? { entries: [] }
      : String(url).includes("version") ? { build: "smoke" }
      : String(url).includes("sessions") ? { sessions: [] }
      : {};
    return Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(""),
    });
  },
  localStorage: {
    getItem: k => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
    removeItem: k => storage.delete(k),
  },
  navigator: { serviceWorker: undefined, userAgent: "smoke" },
  caches: undefined,
  history: { replaceState() {} },
  screen: { width: 400, height: 800 },
  requestAnimationFrame: () => 0,
  AbortController,
  HTMLVideoElement: { prototype: { requestVideoFrameCallback() {} } },
  URL, URLSearchParams, Date, Math, JSON, Promise, Error, Number, String,
  Array, Object, RegExp, Map, Set, parseInt, parseFloat, isNaN, isFinite,
  encodeURIComponent, decodeURIComponent, alert() {}, prompt: () => null,
  confirm: () => false,
  addEventListener() {}, removeEventListener() {}, dispatchEvent: () => true,
};
sandbox.window = sandbox;
sandbox.self = sandbox;
sandbox.document = {
  getElementById: () => el(),
  createElement: () => el(),
  addEventListener() {}, removeEventListener() {},
  body: el(), documentElement: el(),
  querySelector: () => null, querySelectorAll: () => [],
  hidden: false,
};
sandbox.location = new URL("https://billiards-review.vercel.app/");

const src = fs.readFileSync(
  path.join(__dirname, "public", "app.js"), "utf-8");

let evalOk = false;
try {
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox, { filename: "app.js", timeout: 15000 });
  evalOk = true;
} catch (e) {
  console.error("SMOKE FAIL: top-level evaluation threw:\n  " + e.stack.split("\n").slice(0, 4).join("\n  "));
  process.exit(1);
}

// let the async tail (loadSessions -> fetch) run
setTimeout(() => {
  const hitSessions = calls.fetches.some(u => u.includes("/api/sessions"));
  if (!evalOk || !hitSessions) {
    console.error("SMOKE FAIL: evaluated=" + evalOk +
      ", /api/sessions fetched=" + hitSessions +
      "\n  fetches seen: " + JSON.stringify(calls.fetches));
    process.exit(1);
  }
  console.log("SMOKE OK: app.js evaluates end-to-end and requests /api/sessions"
    + " (" + calls.fetches.length + " fetches observed)");
  process.exit(0);
}, 300);
