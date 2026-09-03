import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const output = resolve('dist');
const files = [];

function walk(directory) {
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) walk(path);
    else files.push(path);
  }
}

function localTarget(page, value) {
  if (value.startsWith('#') || /^(?:https?:|mailto:|data:)/.test(value)) return null;
  const url = new URL(value, 'https://koktn.github.io/');
  let target = value.startsWith('/')
    ? join(output, decodeURIComponent(url.pathname))
    : resolve(dirname(page), decodeURIComponent(url.pathname));
  if (url.pathname.endsWith('/') || !target.split('/').at(-1).includes('.')) {
    target = join(target, 'index.html');
  }
  return target;
}

if (!existsSync(output)) throw new Error('dist/ is missing; run npm run build first.');
walk(output);

const missing = [];
for (const page of files.filter((file) => file.endsWith('.html'))) {
  const html = readFileSync(page, 'utf8');
  if (!html.includes('<link rel="canonical"') || !html.includes('<meta property="og:title"')) {
    missing.push(`${page}: required SEO metadata`);
  }
  if (html.includes('下書きプレビュー')) missing.push(`${page}: draft content was published`);
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const target = localTarget(page, match[1]);
    if (target && !existsSync(target)) missing.push(`${page}: ${match[1]}`);
  }
}

for (const required of ['rss.xml', 'sitemap-index.xml', 'sitemap-0.xml', 'robots.txt']) {
  if (!existsSync(join(output, required))) missing.push(`dist/${required}: missing`);
}

if (missing.length) {
  throw new Error(`Built-site validation failed:\n${missing.join('\n')}`);
}

console.log(`Validated ${files.filter((file) => file.endsWith('.html')).length} HTML pages and all local links.`);
