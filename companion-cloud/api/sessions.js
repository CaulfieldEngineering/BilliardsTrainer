const { dbxToken, checkKey, FOLDER } = require("./_lib.js");

module.exports = async (req, res) => {
  if (!checkKey(req, res)) return;
  try {
    const token = await dbxToken();
    let entries = [];
    let body = { path: FOLDER, limit: 500 };
    let url = "https://api.dropboxapi.com/2/files/list_folder";
    for (;;) {
      const r = await fetch(url, {
        method: "POST",
        headers: { Authorization: "Bearer " + token,
                   "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error("list_folder " + r.status);
      const d = await r.json();
      entries = entries.concat(d.entries);
      if (!d.has_more) break;
      url = "https://api.dropboxapi.com/2/files/list_folder/continue";
      body = { cursor: d.cursor };
    }
    const summaries = new Set(
      entries.filter(e => e.name.endsWith(".shots.json")).map(e => e.name));
    const sessions = entries
      .filter(e => e[".tag"] === "file" && /\.(mp4|mov|m4v)$/.test(e.name))
      .map(e => ({
        name: e.name,
        size_mb: Math.round(e.size / 1e6),
        modified: e.server_modified,
        analyzed: summaries.has(e.name + ".shots.json"),
      }))
      .sort((a, b) => (a.modified < b.modified ? 1 : -1))
      .slice(0, 100);
    res.setHeader("Cache-Control", "no-store");
    res.status(200).json({ sessions });
  } catch (err) {
    res.status(502).json({ error: String(err.message || err) });
  }
};
