# Synthetic Face Demo

> **Responsible AI demonstration** — educational use only.

This project demonstrates synthetic face generation and face-swapping techniques to raise awareness about deepfake technology, its capabilities, and how to detect it. Every output is watermarked and the pipeline requires explicit consent at each step.

![Synthetic Face Demo UI](assets/UI.png)

---

## What this is

- Face detection and analysis using InsightFace
- Face swap on static images
- Watermarked outputs (visible + invisible)
- Built-in deepfake detection to show the other side of the coin
- Simple Gradio web UI — no API keys required

## What this is NOT

- A tool for non-consensual image manipulation
- Production-ready or hardened software
- A guide for malicious use

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/islas104/synthetic-face-demo.git
cd synthetic-face-demo

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
python app.py
```

Open http://localhost:7860 in your browser.

---

## Project structure

```
synthetic-face-demo/
├── app.py              # Gradio UI entry point
├── core/
│   ├── swap.py         # Face swap pipeline
│   ├── watermark.py    # Visible + invisible watermarking
│   └── detect.py       # Deepfake detection
├── models/             # Downloaded model weights (git-ignored)
├── samples/            # Example images for demo
├── requirements.txt
└── ETHICS.md           # Responsible use policy
```

---

## Responsible use

Read [ETHICS.md](ETHICS.md) before using this software.

Key principles:
1. **Consent first** — only swap faces you have explicit permission to use
2. **Label outputs** — all outputs are watermarked; do not strip the watermark
3. **No impersonation** — do not use to deceive, defraud, or harass
4. **Detection included** — the repo ships a detector so viewers can verify synthetic media

---

## License

MIT — see [LICENSE](LICENSE). By using this software you agree to the terms in [ETHICS.md](ETHICS.md).
