#!/usr/bin/env python3
"""Generate media derivatives, or validate a built site. Requires ffmpeg/ffprobe and Pillow."""
import argparse
import json
import io
from PIL import Image
import struct
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def probe(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))
    return data['streams']


def generate():
    manifest = {}
    for path in sorted(list((ROOT / 'videos').glob('*.mp4')) + list((ROOT / 'images').glob('*'))):
        if not path.is_file() or path.suffix.lower() not in ('.mp4', '.jpg', '.jpeg', '.png', '.webp'):
            continue
        stream = next(s for s in probe(path) if s['codec_type'] == 'video')
        width, height = stream['width'], stream['height']
        item = {'width': width, 'height': height}
        if path.suffix == '.mp4':
            poster = ROOT / 'videos/posters' / (path.stem + '.webp')
            poster.parent.mkdir(exist_ok=True)
            if not poster.exists():
                frame = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'])
                Image.open(io.BytesIO(frame)).save(poster, quality=82)
            item['poster'] = '/' + str(poster.relative_to(ROOT))
        elif not path.name.startswith(('favicon', '404')):
            variants = []
            for size in sorted(set(min(width, n) for n in (320, 640, 960, width))):
                output = ROOT / 'images/responsive' / f'{path.stem}-{size}.webp'
                output.parent.mkdir(exist_ok=True)
                if not output.exists() or output.stat().st_mtime < path.stat().st_mtime:
                    with Image.open(path) as image:
                        image.resize((size, round(height * size / width)), Image.Resampling.LANCZOS).save(output, quality=88, method=6)
                variants.append({'url': '/' + str(output.relative_to(ROOT)), 'width': size})
            item['variants'] = variants
        manifest['/' + str(path.relative_to(ROOT))] = item
    (ROOT / '_data/media.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Generated metadata for {len(manifest)} assets')


def faststart(path):
    # Parse top-level atoms, rather than matching incidental bytes in encoded frames.
    with path.open('rb') as f:
        while header := f.read(8):
            size, kind = struct.unpack('>I4s', header)
            consumed = 8
            if size == 1:
                size = struct.unpack('>Q', f.read(8))[0]
                consumed = 16
            if kind == b'moov':
                return True
            if kind == b'mdat' or size < consumed:
                return False
            f.seek(size - consumed, 1)
    return False


class MediaParser(HTMLParser):
    def __init__(self, root, page, errors):
        super().__init__()
        self.root, self.page, self.errors = root, page, errors

    def check(self, value):
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path:
            return
        path = self.root / unquote(url.path).lstrip('/') if url.path.startswith('/') else self.page.parent / unquote(url.path)
        if not path.is_file():
            self.errors.append(f'{self.page.relative_to(self.root)}: missing {value}')

    def handle_starttag(self, tag, attributes):
        if tag not in ('video', 'img', 'source'):
            return
        attrs = dict(attributes)
        for key in ('src', 'data-src', 'poster', 'data-original'):
            if attrs.get(key):
                self.check(attrs[key])
        for key in ('srcset', 'data-srcset'):
            for part in attrs.get(key, '').split(','):
                if part.strip():
                    self.check(part.strip().split()[0])
        if tag in ('video', 'img') and (attrs.get('src', attrs.get('data-src', '')).startswith('/')):
            if not all(attrs.get(k, '').isdigit() for k in ('width', 'height')):
                self.errors.append(f'{self.page.relative_to(self.root)}: {tag} missing dimensions')
        if tag == 'video' and not attrs.get('poster'):
            self.errors.append(f'{self.page.relative_to(self.root)}: video missing poster')


def validate(build):
    errors = []
    manifest = json.loads((ROOT / '_data/media.json').read_text())
    for url, item in manifest.items():
        path = ROOT / url.lstrip('/')
        streams = probe(path)
        stream = next(s for s in streams if s['codec_type'] == 'video')
        if (stream['width'], stream['height']) != (item['width'], item['height']):
            errors.append(f'{url}: stale dimensions')
        if path.suffix == '.mp4':
            if stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p' or any(s['codec_type'] == 'audio' for s in streams):
                errors.append(f'{url}: expected silent H.264 yuv420p')
            if not faststart(path):
                errors.append(f'{url}: moov atom must precede mdat')
            poster = ROOT / item['poster'].lstrip('/')
            if not poster.is_file():
                errors.append(f'{url}: missing poster')
            else:
                probe(poster)
    pages = list(build.rglob('*.html'))
    if not pages:
        errors.append(f'No HTML found in {build}')
    for page in pages:
        MediaParser(build, page, errors).feed(page.read_text())
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Validated {len(manifest)} assets and media references in {len(pages)} pages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'validate'])
    parser.add_argument('--site', type=Path, default=ROOT / '_site')
    args = parser.parse_args()
    generate() if args.command == 'generate' else validate(args.site.resolve())
