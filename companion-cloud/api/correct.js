const { dbxToken, checkKey, safeName, FOLDER } = require("./_lib.js");

// "Correct this clip": upload a small verdict JSON into the corrections
// queue folder. The mini PC's Dropbox client syncs it down and the
// companion watcher applies it REVIEW-ranked to the session's sidecar.
// The write path is server-enforced to exactly one folder.
const OUTCOMES = new Set(["make", "miss", "scratch"]);
const CUTS = new Set(["left", "right", "straight"]);
const SIDES = new Set(["left", "right"]);
const ACTIONS = new Set(["stroke", "break", "ball_in_hand", "rearrange",
                         "nothing"]);

module.exports = async (req, res) => {
  if (!checkKey(req, res)) return;
  if (req.method !== "POST") return res.status(405).json({ error: "POST" });
  const b = req.body || {};
  const name = safeName(b.session);
  const start = Number(b.start);
  const outcome = OUTCOMES.has(b.outcome) ? b.outcome : undefined;
  const action = ACTIONS.has(b.action) ? b.action : undefined;
  const note = (typeof b.note === "string" ? b.note : "").trim().slice(0, 500);
  const cut = CUTS.has(b.cut) ? b.cut : undefined;
  const missSide = SIDES.has(b.miss_side) ? b.miss_side : undefined;
  const clear = b.clear === true;
  const confirm = b.confirm === true;
  const rife = b.rife === true;
  const split = isFinite(Number(b.split)) ? Number(b.split) : null;
  if (!name || !isFinite(start)
      || (!outcome && !action && !note && !clear && !confirm && !rife
          && !cut && !missSide && split == null)) {
    return res.status(400).json({ error: "bad verdict" });
  }
  const doc = { session: name, start: Math.round(start * 100) / 100,
                ts: new Date().toISOString() };
  if (split != null) doc.split = Math.round(split * 100) / 100;
  if (rife) {
    doc.rife = true;
    if (isFinite(Number(b.end))) doc.end = Math.round(Number(b.end) * 100) / 100;
  }
  if (clear) doc.clear = true;
  if (confirm) doc.confirm = true;
  if (cut) doc.cut = cut;
  if (missSide) doc.miss_side = missSide;
  if (outcome) doc.outcome = outcome;
  if (action) doc.action = action;
  if (note) doc.note = note;
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  try {
    const token = await dbxToken();
    const r = await fetch("https://content.dropboxapi.com/2/files/upload", {
      method: "POST",
      headers: {
        Authorization: "Bearer " + token,
        "Content-Type": "application/octet-stream",
        "Dropbox-API-Arg": JSON.stringify({
          path: `${FOLDER}/corrections/${id}.json`,
          mode: "add", autorename: true, mute: true,
        }),
      },
      body: JSON.stringify(doc),
    });
    if (!r.ok) throw new Error("upload " + r.status);
    res.status(200).json({ queued: true });
  } catch (err) {
    res.status(502).json({ error: String(err.message || err) });
  }
};
