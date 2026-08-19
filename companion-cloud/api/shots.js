const { dbxToken, checkKey, safeName, FOLDER } = require("./_lib.js");

module.exports = async (req, res) => {
  if (!checkKey(req, res)) return;
  const name = safeName(req.query && req.query.name);
  if (!name) return res.status(400).json({ error: "bad name" });
  try {
    const token = await dbxToken();
    const r = await fetch("https://content.dropboxapi.com/2/files/download", {
      method: "POST",
      headers: {
        Authorization: "Bearer " + token,
        "Dropbox-API-Arg": JSON.stringify(
          { path: `${FOLDER}/${name}.shots.json` }),
      },
    });
    if (r.status === 409) return res.status(404).json({ error: "no summary" });
    if (!r.ok) throw new Error("download " + r.status);
    res.setHeader("Cache-Control", "no-store");
    res.status(200).json(await r.json());
  } catch (err) {
    res.status(502).json({ error: String(err.message || err) });
  }
};
