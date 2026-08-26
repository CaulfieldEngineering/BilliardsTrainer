const { dbxToken, checkKey, safeName, FOLDER } = require("./_lib.js");

module.exports = async (req, res) => {
  if (!checkKey(req, res)) return;
  let name = safeName(req.query && req.query.name);
  // slow-mo renders live in a subfolder the sessions list never shows
  const raw = String((req.query && req.query.name) || "");
  if (!name && /^slowmo\/[A-Za-z0-9._-]{1,80}\.mp4$/.test(raw)) name = raw;
  if (!name) return res.status(400).json({ error: "bad name" });
  try {
    const token = await dbxToken();
    const grab = async (p) => {
      const r = await fetch(
        "https://api.dropboxapi.com/2/files/get_temporary_link", {
          method: "POST",
          headers: { Authorization: "Bearer " + token,
                     "Content-Type": "application/json" },
          body: JSON.stringify({ path: p }),
        });
      return r.ok ? (await r.json()).link : null;
    };
    // PHONE PROXY (2026-08-26): sessions record at 16-24 Mbps — heavy to
    // decode on the phone and heavier on cellular. A 720p ~3.5 Mbps proxy
    // with the IDENTICAL timeline is rendered at session close into
    // proxies/; prefer it when it exists. Same times => every overlay,
    // seek and correction works unchanged.
    let link = null, proxied = false;
    if (/^session-[A-Za-z0-9._-]+\.mp4$/.test(name)) {
      link = await grab(`${FOLDER}/proxies/${name}`);
      proxied = !!link;
    }
    if (!link) link = await grab(`${FOLDER}/${name}`);
    if (!link) throw new Error("get_temporary_link failed");
    // valid ~4h, served by Dropbox with Range support (seekable on iOS)
    res.setHeader("Cache-Control", "no-store");
    res.status(200).json({ link, proxied });
  } catch (err) {
    res.status(502).json({ error: String(err.message || err) });
  }
};
