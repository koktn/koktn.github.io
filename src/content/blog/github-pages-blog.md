---
title: GitHub pagesでBlog作成
description: GitHub Pagesでブログを作成したときに参考にした情報をまとめます。
publishedAt: 2020-09-25
category: Vue.js
tags:
  - GitHubPages
  - Vue.js
draft: false
---

ふと、GitHub PagesでBlogを作ろうと思い（結果できあがったのがココ）、参考にしたページをまとめます。

## GitHub Pagesでのブログのデプロイ

blogのデプロイ手順については、こちらを参考にしました。

[VuePress を使って GitHub Pages でブログを公開する](https://qiita.com/kumapo0313/items/a59df3d74a7eaaaf3137)

サイト用リポジトリ名は `[アカウント名].github.io` にする必要があります。
（作られるページは `https://アカウント名.github.io`）

## VuePressのテーマ

今回テンプレートには、vuepress-theme-meteorlxy を使いました。

インストール方法は[vuepress-theme-meteorlxy](https://vuepress-theme-meteorlxy.meteorlxy.cn/posts/2019/02/27/theme-guide-en.html)を参考にしました。

また、日本語の対応に関しては以下のサイトが詳しいです。

[VuePressで作ったblogに配布されているテーマを設定する](https://qiita.com/tomopict/items/9da7cf28c9bcd5f933cb)

1点、メニューの日本語化については、`en-JP`ではなく、`ja-JP`に修正されているようです。

```js
// 変更前
themeConfig: {
  lang: 'en-US',
}

// 変更後
themeConfig: {
  lang: 'ja-JP',
  // OR
  lang: require('vuepress-theme-meteorlxy/lib/langs/en-JP'),
}
```

## おわりに

とりあえずこの情報で今のこのページの状態までは作れます。あとは中身を見ながらボチボチいじっていこうかなと思います。
