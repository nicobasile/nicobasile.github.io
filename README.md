# nicobasile.github.io

Personal site and research notes on embodied AI, robot policy training, and accelerator infrastructure. Live at [nicobasile.github.io](https://nicobasile.github.io).

Built with [Jekyll](https://jekyllrb.com/) and hosted on GitHub Pages.

## Running locally

```bash
bundle install
bundle exec jekyll serve
```

The site is then at `http://localhost:4000`.

## Layout

| Path | Contents |
| --- | --- |
| `_posts/` | Published articles, `YYYY-M-D-slug.md` |
| `_drafts/` | Work in progress, never built |
| `_pages/` | Standalone pages such as About |
| `_layouts/`, `_includes/` | Templates |
| `_sass/`, `assets/style.scss` | Styles |
| `assets/particles.js` | Background canvas effect |
| `images/`, `videos/` | Post media |

## Posts

Front matter drives both the article page and its homepage card:

```yaml
---
layout: post
title: Post Title
categories: [embodied-ai, robotics]
author: Nicolas Basile
description: "Plain-text summary used for SEO and social previews."
hook: "Rich-text summary shown on the homepage card; may contain <strong> tags."
media_type: video          # or image
media_url: /videos/clip.mp4
media_alt: "Description of the clip"
---
```

`media_url_2` / `media_type_2` / `media_alt_2` add a second card thumbnail.

## Credits

Originally forked from [Reverie](https://github.com/amitmerchant1990/reverie) by Amit Merchant, itself a fork of [jekyll-now](https://github.com/barryclark/jekyll-now). MIT licensed.

## Media performance

Render media with `{% include media.html url="/videos/clip.mp4" alt="Description" controls=true %}` (omit `controls` for card previews). Existing post front matter still works. The include uses `_data/media.json` for intrinsic dimensions, posters, and responsive image sources. Optional parameters include `type`, `poster`, `class`, `sizes`, `eager`, and `deferred` (for images in initially hidden panels).

Videos load within 300px of the viewport and play only onscreen in the active browser tab. Filters, comparison tabs, and expanded media share `window.SiteMedia` (`refresh`, `setModal`, `release`). Reduced motion and blocked autoplay show a Play button. Network failures retain the player and poster, retry twice with backoff, then allow explicit Retry. Offscreen pauses resume automatically; deliberate user pauses persist.

After adding or replacing media, generate derivatives and metadata with Python 3, Pillow, and ffmpeg/ffprobe:

```bash
python3 scripts/media_assets.py generate
bundle exec jekyll build --destination /tmp/nico-media-site
python3 scripts/media_assets.py validate --site /tmp/nico-media-site
```

This keeps original images and MP4s intact, creates WebP image variants, and supplies missing video posters. GIF originals are retained but are not used by the published articles. Use silent H.264/yuv420p MP4s with the `moov` atom before `mdat` (`-movflags +faststart`) for new animations.

The browser regression script requires Playwright and its Chromium, Firefox, and WebKit runtimes. Install those in your development tooling, then run (set `NODE_PATH` if Playwright is installed outside this repository):

```bash
node scripts/test_media.cjs /tmp/nico-media-site
# Optionally compare initial video requests against a previous build:
node scripts/test_media.cjs /tmp/nico-media-site /tmp/nico-media-before
```

Tests cover playback visibility, filters, comparisons, lightbox cleanup, manual pauses, reduced motion, autoplay rejection, HTTP failure retry limits, offline recovery, and responsive images on mobile. Browser-tab visibility is simulated deterministically; real iPhone/Safari testing remains useful for device-specific power-saving policies.

Set `BROWSERS=chromium,firefox` or `BROWSERS=webkit` to run a subset. On macOS 14, WebKit was verified with Playwright 1.51.1 / WebKit 18.4; the newer bundled driver's `PushAPIEnabled` setting is incompatible with its macOS 14 WebKit runtime.
