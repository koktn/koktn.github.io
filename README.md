# koki's blog

Astroで生成し、GitHub Pagesで公開する個人ブログです。

## 必要な環境

Node.js 24とnpmを使用します。Node.jsのバージョンは`.nvmrc`に固定しています。

```sh
nvm use
npm ci
npm run dev
```

`http://localhost:4321/`を開くと、下書きを含むサイトを確認できます。

## ディレクトリ構成

- `src/content/blog/`: Markdown形式の記事
- `src/pages/`: ページとURLの定義
- `src/components/`, `src/layouts/`: 共通UI
- `src/styles/global.css`: サイト全体のデザイン
- `public/img/`: そのまま配信する画像
- `.github/workflows/`: PR検証とGitHub Pagesへのデプロイ

`dist/`、`.astro/`、`node_modules/`は生成物なので編集・コミットしません。

## 新しい記事を追加する

1. `main`を最新にして作業ブランチを作ります。

   ```sh
   git switch main
   git pull
   git switch -c post/my-new-post
   ```

2. `src/content/blog/`に英小文字のkebab-caseでMarkdownファイルを作ります。例: `my-new-post.md`

3. 次のfrontmatterと本文を記述します。新規記事は必ず`draft: true`から始めます。

```yaml
---
title: 記事タイトル
description: 一覧と検索結果に表示する短い概要
publishedAt: 2026-09-03
# updatedAt: 2026-09-04
category: 開発
tags:
  - Astro
draft: true
# image: /img/posts/my-new-post-cover.jpg
---

ここからMarkdownで本文を書きます。
```

frontmatterの意味:

| 項目 | 必須 | 説明 |
| --- | --- | --- |
| `title` | 必須 | 記事タイトル |
| `description` | 必須 | 記事一覧、OGP、RSSで使用する概要 |
| `publishedAt` | 必須 | 公開日（`YYYY-MM-DD`） |
| `updatedAt` | 任意 | 内容を更新した日 |
| `category` | 必須 | 1つのカテゴリー |
| `tags` | 必須 | タグの配列。不要なら`tags: []` |
| `draft` | 必須 | `true`は下書き、`false`は公開 |
| `image` | 任意 | OGP画像のルート相対パス |

記事URLは公開日とファイル名から、`/posts/YYYY/MM/DD/my-new-post/`の形式で生成されます。公開後に`publishedAt`またはファイル名を変えるとURLも変わるため、通常は変更しません。

## 画像を追加する

画像は`public/img/posts/`へ、記事名を含むkebab-caseで保存します。

```text
public/img/posts/my-new-post-diagram.png
```

Markdownではルート相対パスと具体的な代替テキストを指定します。

```md
![構成図の説明](/img/posts/my-new-post-diagram.png)
```

ファイルサイズを抑え、写真はJPEG、透過が必要な図はPNG、対応可能ならWebPを使用してください。

## 確認して公開する

1. `npm run dev`で本文、画像、見出しリンク、カテゴリー、タグ、ライト／ダーク表示を確認します。開発環境では下書きにバッジが表示されます。
2. 公開準備ができたら`draft: false`へ変更します。
3. 本番と同じ検査を実行します。

   ```sh
   npm run check
   npm run build
   npm test
   npm run preview
   ```

4. 変更をコミットしてpushし、`main`向けのPull Requestを作成します。
5. `CI / validate`の成功を確認してマージします。`main`へのマージ後、`Deploy to GitHub Pages`が自動的に公開します。
6. [Actions](https://github.com/koktn/koktn.github.io/actions)が成功し、[公開サイト](https://koktn.github.io/)に反映されたことを確認します。

## 検証

```sh
npm run check
npm run build
npm test
npm run preview
```

各コマンドの役割:

- `check`: Astro、TypeScript、frontmatterを検査
- `build`: 本番用の静的サイトを`dist/`へ生成
- `test`: 生成後の内部リンク、画像、SEOメタデータ、RSS、サイトマップを検査
- `preview`: `dist/`をローカル配信

RSSは`/rss.xml`、サイトマップは`/sitemap-index.xml`に生成されます。

## 既存記事の更新・非公開化

- 内容を修正した場合は`updatedAt`を更新します。
- URLを維持するため、既存記事のファイル名と`publishedAt`は変更しません。
- 一時的に公開から外す場合は`draft: true`へ変更してデプロイします。記事はサイト、RSS、サイトマップから除外されます。
- このリポジトリとGit履歴は公開されています。下書きであっても機密情報、個人情報、APIキーは絶対にコミットしないでください。

## 依存関係の保守

DependabotがnpmパッケージとGitHub Actionsの更新PRを毎月作成します。更新PRではリリースノートを確認し、次を通してからマージします。

```sh
npm ci
npm run check
npm run build
npm test
```

Node.jsのメジャーバージョンを変更する場合は、`.nvmrc`、`package.json`の`engines.node`、両GitHub Actionsのバージョン指定を同時に更新します。

## 障害対応と切り戻し

- CI失敗: Actionsの失敗ステップを確認し、同じコマンドをローカルで再現します。
- ローカル依存の不整合: `node_modules/`を削除して`npm ci`をやり直します。
- デプロイ失敗: **Settings → Pages → Source**が**GitHub Actions**であることと、`github-pages`環境が`main`を許可していることを確認し、失敗したworkflowを再実行します。
- 公開内容に問題がある: 問題のコミットを`git revert <commit>`で打ち消すPRを作り、マージして再デプロイします。共有履歴を書き換える`git reset --hard`やforce pushは使用しません。

## GitHub設定

- Default branch: `main`
- Pages source: **GitHub Actions**
- Production environment: `github-pages`、許可ブランチ`main`
- Branch protection: `main`へのPRと`CI / validate`成功を必須にすることを推奨
