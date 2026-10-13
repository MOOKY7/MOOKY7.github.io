# Mookyong Son website

A research portfolio with two blog tracks, Markdown articles, search, RSS, and a static site build. The site works on GitHub Pages. It has no database or paid service dependency.

## Start here

1. Open `dist/index.html` through a local web server to view the included build. Opening it directly as a file will not resolve links that start with `/`.
2. Edit your profile in `content/profile.json`.
3. Edit the project summaries in `content/projects.json` and papers in `content/publications.json`.
4. Add notes in `content/posts/`.

The first articles are editable launch copy. Review the new JEPA note, missing data note, and weekly plan in your own voice before putting them on your public GitHub website. The weekly entry describes planned work and makes no claim of completion. The old JAX post has been repaired and updated. The original upload is retained in `archive/original/` and is not included in the deployed site.

Your role is shown as Research Associate from October 2026. No email address or CV file was supplied, so the site uses your existing LinkedIn link for contact and includes a printable research profile. There is no broken CV link. Add a verified email or a current CV when ready.

## Publish to your existing GitHub website

The supplied archive does not change the live GitHub repository by itself.

1. Make a backup or a new branch in `MOOKY7.github.io`.
2. Replace the old website source with the files in this folder. Include `.github/workflows/pages.yml`. Do not keep an old `index.html`, `_config.yml`, `_layouts/`, or `_includes/` at the repository root. Their originals are already preserved under `archive/original/`.
3. In GitHub, open the repository Settings, then Pages. Set the build source to GitHub Actions.
4. Commit the new files to the `main` or `master` branch that serves your site.
5. Wait for the Publish portfolio workflow to complete. The workflow builds and validates `dist/` before publishing it.
6. Visit `https://mooky7.github.io` and check the homepage, a note, and the mobile menu.

The generated `dist/` folder is included for convenience. GitHub Actions rebuilds it from the editable source. Do not edit generated HTML as your main workflow.

If you copy the site to a different domain, change `siteUrl` in `content/profile.json`. This version expects a site at the root of its domain, as your existing personal GitHub website is. It does not support a repository subpath without changing the URL logic.

## Local editing

Install Node.js 22 or later. In this folder run the following commands.

```sh
npm install
npm run build
npm run check
python3 -m http.server 8000 --directory dist
```

Visit `http://localhost:8000`. Rebuild after editing content. The only build dependency is Marked, pinned to version 17.0.5. Google Fonts is optional at runtime, and the layout includes local font fallbacks.

## Publish a new note

You can use a terminal or create the file directly in GitHub.

```sh
npm run new -- architecture attention-masks "What an attention mask changes"
npm run new -- practice week-two "Week 02 · Check the implementation"
```

The command creates a Markdown draft with a JSON metadata block. JSON is used so the build needs no YAML parser. Keep the opening and closing `---` lines. Use double quotes and valid JSON inside them.

```md
---
{
  "title": "What an attention mask changes",
  "date": "2026-10-19",
  "slug": "attention-masks",
  "track": "architecture",
  "kind": "Technical note",
  "summary": "A short question and the evidence used to answer it.",
  "tags": ["Attention", "Python"],
  "published": false
}
---

## The question

Write your note here.
```

1. Choose `architecture` or `practice` as the track.
2. Keep each slug unique. Once a note is public, keep its slug stable.
3. Add the content, code, and source links.
4. Set `published` to `true` when the note is ready.
5. Commit the file. GitHub Actions builds the site, search index attributes, article page, and feeds.

Set `updated` to a date such as `2026-10-20` when revising a published note. Set `published` to `false` to remove a note from a new build. A new build clears the old output first. Future dates are descriptive only and do not schedule publication. A note with `published` set to `true` is included immediately.

Draft files are omitted from the generated site. They remain visible in a public source repository. Keep private plans outside a public repository.

## A sustainable weekly rhythm

Publish one useful technical note when you have a clear explanation or a small experiment. Publish one practice review at the end of a week. A short review with working code and one honest failure is more useful than a long list of unsupported accomplishments.

For Architecture Notes, include the question, mechanism, small example, checks, limitations, and original sources. For Research Practice, separate the plan from completed work, link the code, and state what changes next week.

Suggested first topics are attention shape checks, masking, representation collapse, calibration under shift, and fair model comparisons. These are ideas for future writing, not completed work.

## Add your CV or email

Put the CV at `assets/Mookyong_Son_CV.pdf` and set `cv` in `content/profile.json` to `/assets/Mookyong_Son_CV.pdf`. Set `email` to your preferred public contact address. Rebuild. The relevant links will appear automatically.

## Feeds and old links

The combined RSS feed is `/feed.xml`. Individual feeds are `/feeds/architecture.xml` and `/feeds/practice.xml`. The old blog index and both original post URLs redirect to the new notebook.

## Verification

`npm run check` checks internal pages, assets, anchors, duplicate IDs, required metadata, and placeholder profile text. The site includes responsive styles, keyboard controls, reduced motion support, a skip link, and readable content without JavaScript. Browser layout checks were not available in the creation environment, so review the final rendering on a phone and a desktop.

The representation animation is an explicitly labeled schematic with synthetic points. It does not show GENIE data, a fitted embedding, or empirical performance.

## Attribution

The original website used the [Plain Academic template](https://github.com/mavroudisv/plain-academic). Its source and license remain in `archive/original/`. The original BSD license notice is retained in `LICENSE.txt` and the generated site. The new design and content structure replace the old layout.
