// Shared: Dropbox short-lived token (from the refresh token held in env)
// and the page-key gate. The refresh token is READ-ONLY scoped
// (files.metadata.read + files.content.read) and never leaves the server.
let cached = { token: null, exp: 0 };

async function dbxToken() {
  if (cached.token && Date.now() < cached.exp - 60000) return cached.token;
  const r = await fetch("https://api.dropbox.com/oauth2/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: process.env.DROPBOX_REFRESH_TOKEN,
      client_id: process.env.DROPBOX_APP_KEY,
      client_secret: process.env.DROPBOX_APP_SECRET,
    }),
  });
  if (!r.ok) throw new Error("dropbox token exchange failed: " + r.status);
  const d = await r.json();
  cached = { token: d.access_token, exp: Date.now() + d.expires_in * 1000 };
  return cached.token;
}

const crypto = require("crypto");

function checkKey(req, res) {
  // HEADER ONLY: query-string keys land in edge logs and referrers, and
  // the page never sends them anyway (security review). Constant-time
  // compare; fail closed when PAGE_KEY is unset.
  const k = req.headers["x-key"];
  const expect = process.env.PAGE_KEY;
  const ok = Boolean(expect) && typeof k === "string"
    && k.length === expect.length
    && crypto.timingSafeEqual(Buffer.from(k), Buffer.from(expect));
  if (!ok) {
    res.status(401).json({ error: "unauthorized" });
    return false;
  }
  return true;
}

// One session filename, nothing else: no slashes, no traversal, .mp4 only.
function safeName(name) {
  return (typeof name === "string"
          && /^[A-Za-z0-9][A-Za-z0-9._ -]{0,120}\.(mp4|mov|m4v)$/.test(name))
    ? name : null;
}

const FOLDER = "/Billiards/BilliardsTrainer";

module.exports = { dbxToken, checkKey, safeName, FOLDER };
