import { MetadataRoute } from 'next'
import { allBlogs } from 'contentlayer/generated'
import siteMetadata from '@/data/siteMetadata'

function joinUrl(base: string, path: string): string {
  const baseWithoutTrailingSlash = base.replace(/\/$/, '')
  const pathWithLeadingSlash = path.startsWith('/') ? path : `/${path}`
  return `${baseWithoutTrailingSlash}${pathWithLeadingSlash}`
}

export default function sitemap(): MetadataRoute.Sitemap {
  const siteUrl = siteMetadata.siteUrl

  const blogRoutes = allBlogs
    .filter((post) => !post.draft)
    .map((post) => ({
      url: joinUrl(siteUrl, post.path),
      lastModified: post.lastmod || post.date,
    }))

  const routes = ['', 'blog', 'projects', 'tags'].map((route) => ({
    url: joinUrl(siteUrl, route),
    lastModified: new Date().toISOString().split('T')[0],
  }))

  return [...routes, ...blogRoutes]
}
