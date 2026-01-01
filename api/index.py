from __future__ import annotations

import os
from pathlib import Path

from chat_recycler.dashboard.app import create_app

OUT_DIR = Path(os.environ.get("CHAT_RECYCLER_OUT", "out"))
app = create_app(OUT_DIR)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
