import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { SITE } from '../config';
import { isPublished, postPath, sortPosts } from '../utils/posts';

export async function GET(context: { site: URL }) {
  const posts = sortPosts((await getCollection('blog')).filter(isPublished));
  return rss({
    title: SITE.title,
    description: SITE.description,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.publishedAt,
      link: postPath(post),
      categories: [post.data.category, ...post.data.tags],
    })),
    customData: '<language>ja</language>',
  });
}
