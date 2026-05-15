# Magic Movie

A web app that recommends movies from MovieLens data and shows the statistics behind each pick (final project for MATH1220: Mathematical Statistics).

![MovieMagic Homepage](web/assets/images/homepage.png)

## How to run

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m magicmovie.fetch_movielens
python -m magicmovie.app --port 8001
```

Then open:

```text
http://127.0.0.1:8001/
```

Do not open `web/index.html` directly. The app needs the local Python server for the CSS, JavaScript, images, and recommendation API.

## Notes

- The app downloads the official MovieLens Latest Small dataset.
- If the download is skipped, it falls back to demo data.


## Tests

```bash
pytest
```
