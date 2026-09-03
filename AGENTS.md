# Repository Guidelines

## Project Structure & Module Organization

This is an Astro static blog deployed to `koktn.github.io`. Page routes live in `src/pages/`; reusable UI is in `src/components/`, and shared page shells are in `src/layouts/`. Markdown posts belong in `src/content/blog/` and are validated by `src/content.config.ts`. Keep global styling in `src/styles/global.css`, shared helpers in `src/utils/`, and directly served images or metadata in `public/`. Never commit generated `dist/`, `.astro/`, or `node_modules/` content.

## Build, Test, and Development Commands

Use Node.js 24 and npm:

```sh
npm ci             # install the locked dependency set
npm run dev        # start Astro at http://localhost:4321
npm run check      # validate Astro, TypeScript, and post data
npm run build      # generate the production site in dist/
npm test           # validate generated links, metadata, and feeds
npm run preview    # serve the production build locally
```

Run `npm run check && npm run build && npm test` before opening a pull request.

## Coding Style & Naming Conventions

Use two-space indentation in Astro, TypeScript, JSON, and YAML. Prefer semantic HTML, small Astro components, strict TypeScript, and CSS custom properties over new UI dependencies. Component files use PascalCase (`PostCard.astro`); utilities use lowercase names. Post filenames and image names use descriptive English kebab-case. Use root-relative public URLs such as `/img/posts/example.png`.

## Content & Testing Guidelines

Every post must provide `title`, `description`, `publishedAt`, `category`, `tags`, and `draft` frontmatter. Begin new posts with `draft: true`; production routes, feeds, and indexes exclude drafts. Check affected pages at desktop and mobile widths, both color schemes, and with keyboard navigation. Confirm images, heading links, taxonomy links, `/rss.xml`, and generated sitemap files. Use `git diff --check` to catch whitespace errors.

## Commit & Pull Request Guidelines

History uses the terse subject `Updates`, but new changes should use short imperative subjects such as `Add Astro publishing workflow`. Keep code, migrated content, and required assets in one coherent commit. Pull requests should summarize user-visible changes, identify tested routes, link relevant issues, and include before/after screenshots for layout work. Merging to `main` deploys production, so require the `CI / validate` check first.
