const { dbxToken, checkKey, FOLDER } = require("./_lib.js");

// Playlist sync: the phone's playlists were device-local (localStorage) —
// one lost phone would erase all of Joe's curation. GET returns the
// server copy; POST overwrites it (whole-state, last-write-wins by the
// doc's `mod` stamp — the client merges before posting). The file also
// lands in the PC's Dropbox folder, so playlists are durable and a
// future desktop surface can read the same document.
const PATH = `${FOLDER}/playlists.json`;
const MAX_BYTES = 200 * 1024;

module.exports = async (req, res) => {
  if (!checkKey(req, res)) return;
  try {
    const token = await dbxToken();
    if (req.method === "GET") {
      const r = await fetch("https://content.dropboxapi.com/2/files/download", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + token,
          "Dropbox-API-Arg": JSON.stringify({ path: PATH }),
        },
      });
      if (r.status === 409)
        return res.status(200).json({ mod: 0, playlists: [] });
      if (!r.ok) throw new Error("download " + r.status);
      res.setHeader("Cache-Control", "no-store");
      return res.status(200).json(await r.json());
    }
    if (req.method === "POST") {
      const b = req.body || {};
      if (!Array.isArray(b.playlists))
        return res.status(400).json({ error: "bad doc" });
      const doc = {
        mod: Number(b.mod) || Date.now(),
        playlists: b.playlists.slice(0, 200).map(p => ({
          id: String(p.id || "").slice(0, 32),
          name: String(p.name || "").slice(0, 80),
          mod: Number(p.mod) || 0,
          clips: (Array.isArray(p.clips) ? p.clips : []).slice(0, 500)
            .map(c => {
              const o = { session: String(c.session || "").slice(0, 80),
                          start: Math.round(Number(c.start) * 100) / 100 };
              if (c.slowmo) o.slowmo = String(c.slowmo).slice(0, 120);
              if (c.label) o.label = String(c.label).slice(0, 120);
              return o;
            })
            .filter(c => c.slowmo || (c.session && isFinite(c.start))),
        })).filter(p => p.id && p.name),
      };
      // The "Slow-mo" playlist is PC-OWNED: the render watcher appends
      // to it. A phone push carries the phone's whole document, so a
      // stale/empty copy there would ERASE finished renders — which is
      // exactly what happened (three renders on disk, zero clips in the
      // list). Union the server's Slow-mo clips into whatever arrives.
      try {
        const cur = await fetch(
          "https://content.dropboxapi.com/2/files/download", {
            method: "POST",
            headers: { Authorization: "Bearer " + token,
                       "Dropbox-API-Arg": JSON.stringify({ path: PATH }) },
          });
        if (cur.ok) {
          const prev = await cur.json();
          const srv = (prev.playlists || []).find(q => q.name === "Slow-mo");
          if (srv && srv.clips && srv.clips.length) {
            let mine = doc.playlists.find(q => q.name === "Slow-mo");
            if (!mine) {
              mine = { id: srv.id || "slowmo", name: "Slow-mo",
                       mod: srv.mod || 0, clips: [] };
              doc.playlists.push(mine);
            }
            const have = new Set(mine.clips.map(c => c.slowmo || ""));
            for (const c of srv.clips) {
              if (c.slowmo && !have.has(c.slowmo)) mine.clips.push(c);
            }
          }
        }
      } catch (e) { /* merge is best-effort; never block the save */ }
      const body = JSON.stringify(doc);
      if (body.length > MAX_BYTES)
        return res.status(413).json({ error: "too large" });
      const r = await fetch("https://content.dropboxapi.com/2/files/upload", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + token,
          "Content-Type": "application/octet-stream",
          "Dropbox-API-Arg": JSON.stringify(
            { path: PATH, mode: "overwrite", mute: true }),
        },
        body,
      });
      if (!r.ok) throw new Error("upload " + r.status);
      return res.status(200).json({ saved: true, mod: doc.mod });
    }
    return res.status(405).json({ error: "GET or POST" });
  } catch (err) {
    res.status(502).json({ error: String(err.message || err) });
  }
};
