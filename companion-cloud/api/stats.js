const { dbxToken, checkKey, FOLDER } = require("./_lib.js");

// Lifetime Stats (Joe): the PC writes lifetime_stats.json into the
// recordings folder on every export; this just serves it to the app.
module.exports = async (req, res) => {
  if (!checkKey(req, res)) return;
  try {
    const token = await dbxToken();
    const r = await fetch("https://content.dropboxapi.com/2/files/download", {
      method: "POST",
      headers: {
        Authorization: "Bearer " + token,
        "Dropbox-API-Arg": JSON.stringify(
          { path: `${FOLDER}/lifetime_stats.json` }),
      },
    });
    if (r.status === 409) return res.status(404).json({ error: "no stats yet" });
    if (!r.ok) throw new Error("download " + r.status);
    res.setHeader("Cache-Control", "no-store");
    res.status(200).json(await r.json());
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
};
