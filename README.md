# koki's blog

Astroで生成し、GitHub Pagesで公開する個人ブログです。

## ローカル開発

Node.js 24を使用します。

```sh
npm ci
npm run dev
```

`http://localhost:4321/`を開いて確認してください。

## 記事を書く

`src/content/blog/`にMarkdownファイルを追加します。新しい記事は、確認が終わるまで`draft: true`にしてください。

```yaml
---
title: 記事タイトル
description: 記事の概要
publishedAt: 2026-09-03
category: 開発
tags:
  - Astro
draft: true
---
```

`draft: false`にしてPRを`main`へマージすると、GitHub Actionsが自動的に公開します。

## 検証

```sh
npm run check
npm run build
npm test
npm run preview
```

`check`はAstroとコンテンツの型を検査し、`build`は本番用の静的サイトを`dist/`へ生成します。`test`は生成後の内部リンク、SEOメタデータ、フィードを検証します。

## 初回のGitHub設定

リポジトリの **Settings → Pages → Build and deployment → Source** で **GitHub Actions** を選択してください。`main`には、PRと`CI / validate`の成功を必須にするブランチ保護を推奨します。
