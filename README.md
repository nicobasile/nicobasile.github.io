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
