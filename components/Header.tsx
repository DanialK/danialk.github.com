'use client'

import siteMetadata from '@/data/siteMetadata'
import headerNavLinks from '@/data/headerNavLinks'
import Link from './Link'
import MobileNav from './MobileNav'
import ThemeSwitch from './ThemeSwitch'
import SearchButton from './SearchButton'
import { useEffect, useState } from 'react'

const wait = (timeout: number) => new Promise((resolve) => setTimeout(resolve, timeout))

const Header = () => {
  const [description, setDescription] = useState(siteMetadata.description)
  useEffect(() => {
    const texts = [
      'Artificial Intelligence',
      'Machine Learning',
      'Data Science',
      'Statistics',
      'Web Development',
      'Functional Programming',
      'Data Engineering',
      siteMetadata.description,
    ]

    async function simulateType(textIndex) {
      const text = texts[textIndex]
      let progress = ''

      const time = 100
      for (let i = 0; i < text.length; i++) {
        progress += text[i]
        await wait(time)
        setDescription(progress)
      }

      await wait(1500)
      const nextIndex = textIndex + 1
      await simulateBackslash(textIndex, nextIndex === texts.length ? 0 : nextIndex)
    }

    async function simulateBackslash(lastIndex, nextIndex) {
      const text = texts[lastIndex]
      const time = 50
      let progress = text

      for (let i = text.length - 1; i >= 0; i--) {
        progress = text.substring(0, i)
        await wait(time)
        setDescription(progress)
      }
      await wait(2000)
      simulateType(nextIndex)
    }

    wait(2000).then(() => simulateBackslash(texts.length - 1, 0))
  }, [])

  return (
    <header className="flex items-center justify-between py-10">
      <div>
        <Link href="/" aria-label={siteMetadata.headerTitle}>
          <div className="flex items-center justify-between">
            {typeof siteMetadata.headerTitle === 'string' ? (
              <div className="hidden h-6 text-2xl font-semibold sm:block">
                {siteMetadata.headerTitle}
              </div>
            ) : (
              siteMetadata.headerTitle
            )}
          </div>
        </Link>
        <p className="pt-4 text-lg leading-7 text-gray-500 dark:text-gray-400">{`> ${description}`}</p>
      </div>
      <div className="flex items-center space-x-4 leading-5 sm:space-x-6">
        {headerNavLinks
          .filter((link) => link.href !== '/')
          .map((link) => (
            <Link
              key={link.title}
              href={link.href}
              className="hidden font-medium text-gray-900 hover:text-primary-500 dark:text-gray-100 dark:hover:text-primary-400
              sm:block"
            >
              {link.title}
            </Link>
          ))}
        <SearchButton />
        <ThemeSwitch />
        <MobileNav />
      </div>
    </header>
  )
}

export default Header
