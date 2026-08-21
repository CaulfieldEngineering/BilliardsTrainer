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
    const r = await fetch(
      "https://api.dropboxapi.com/2/files/get_temporary_link", {
        method: "POST",
        headers: { Authorization: "Bearer " + token,
                   "Content-Type": "application/json" },
        body: JSON.stringify({ path: `${FOLDER}/${name}` }),
      });
    if (!r.ok) throw new Error("get_temporary_link " + r.status);
    const d = await r.json();
    // valid ~4h, served by Dropbox with Range support (seekable on iOS)
    res.setHeader("Cache-Control", "no-store");
    res.status(200).json({ link: d.link });
  } catch (err) {
    res.status(502).json({ error: String(err.message || err) });
  }
};
