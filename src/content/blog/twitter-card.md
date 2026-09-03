---
title: twitterにリンクを張ったら出てくるアレ（twitterカード）を設定する
description: ブログのURLをtwitterに貼り付けたときに表示される、twitterカードを設定した備忘録です。
publishedAt: 2020-09-29
updatedAt: 2020-09-30
category: Vue.js
tags:
  - GitHubPages
  - Vue.js
  - twitter
  - Google Search Console
draft: false
---

ブログをURLをtwitterに貼り付けたとき表示される、twitterカードの設定をしました。
その備忘録です。

## プラグイン vuepress-plugin-seo のインストールと設定

[vuepress-plugin-seo](https://github.com/lorisleiva/vuepress-plugin-seo)

インストール：

```bash
npm i vuepress-plugin-seo -D
```

`config.js`の設定はこちらを参考にしました。

[VuePressでSEOに強いOGPを自動生成する](https://dorasu-tech.dorasu.com/posts/2020/01/24/vuepress-ogp-autogen.html)

```js
plugins: {
  'seo': {
    description: ($page, $site) =>
      $page.frontmatter.description ||
      ($page.excerpt && $page.excerpt.replace(/<("[^"]*"|'[^']*'|[^'">])*>/g, '')) ||
      $site.description || '',
    title: ($page, $site) => $page.title || $site.title,
    author: (_, $site) => $site.themeConfig.author,
    twitterCard: _ => 'summary_large_image',
  },
},
```

## OGP Checkerでの確認

Chromeプラグインの[Localhost OGP チェッカー](https://chrome.google.com/webstore/detail/localhost-open-graph-chec/gcbnmkhkglonipggglncobhklaegphgn)で設定が正しくされているかチェックします。

手順は[ここを参考にしました](https://bellbellbell.info/posts/vuepress-custom-twitter-ogp.html#%E8%A8%AD%E5%AE%9A%E3%81%8C%E3%81%86%E3%81%BE%E3%81%8F%E3%81%A3%E3%81%A6%E3%82%8B%E3%81%8B%E7%A2%BA%E8%AA%8D)。

2020/09/30 追記：Chromeプラグインを使用せずに[Card validator](https://cards-dev.twitter.com/validator)で確認できました。

## サイトマップvuepress-plugin-sitemap のインストールと設定

と、ここまでやってtwitter上で確認しても表示されず……sitemapの設定が必要そうだということで、こちらをやりました。

[vuepress-plugin-sitemap](https://github.com/ekoeryanto/vuepress-plugin-sitemap)

インストール：

```bash
npm install vuepress-plugin-sitemap
```

`config.js`の設定は以下の通り。

```js
plugins: {
  'sitemap': {
    hostname: 'https://koktn.github.io/',
    changefreq: 'weekly',
  },
},
```

## robots.txtの設定

robots.txtも必要とのことで設定します。

[robots.txt の仕様](https://developers.google.com/search/reference/robots_txt?hl=ja)

```text
User-agetnt: Twitterbot
Disallow:
User-agent: *
Disallow: /log/
Sitemap: https://koktn.github.io/sitemap.xml
```

`User-agetnt: Twitterbot`は結果不要な気もしますが、試行錯誤の流れで入れています。
また、`robots.txt`は、`src/.vuepress/public`に配置しています。

## Google Search Consoleへの登録

[Google Search Console](https://search.google.com/search-console)にsitemapを登録します。

まず、プロパティの追加をします。方法は今回、URLプレフィックスを選択しました。

![プロパティタイプ](/img/posts/search-console-property.png)

タグを追加して確認をします。

![HTMLタグ](/img/posts/search-console-html-tag.png)

追加先は`config.js`です。

```js
head: [
  ['meta', { name: 'google-site-verification', content: 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' }],
],
```

確認ができたら、sitemap.xmlを登録します。

左メニューのサイトマップから、「新しいサイトマップの追加」を行います。

![サイトマップ](/img/posts/search-console-sitemap.png)

うまく行けば、ステータスが「成功しました」となります。

※プロパティ登録直後だと、私の場合サイトマップの登録に失敗しました。翌日試したらできたので時間がかかるものだと思っておいたほうが良いです。

## twitterでの確認

twitterにブログポストのURLを入力して、twitterカードが表示されるかどうか確認します。

![twitterカードの表示結果](/img/posts/twitter-card-preview.png)

## それでも表示されないときは（キャッシュクリア）

私の場合、上記手順を試してもうまく表示されなかったのですが、twitterに記載するURLを

```text
https://koktn.github.io/?aaa
```

とするとうまくいきました。（`?`を入力して以降の文字列`aaa`は適当）

[こちら](https://hikikomorineet.com/wp-twitter-card)を参考にするとtwitterに残っているキャッシュを`?`でクリアしているそうです。

とりあえず、これで解決しました！
