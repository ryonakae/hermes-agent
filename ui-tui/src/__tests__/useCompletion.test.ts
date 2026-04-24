import { describe, expect, it } from 'vitest'

import { getLocalSlashCompletion } from '../hooks/useCompletion.js'
import type { SlashCatalog } from '../types.js'

const catalog: SlashCatalog = {
  canon: {
    '/bg': '/background',
    '/background': '/background',
    '/help': '/help',
    '/model': '/model',
    '/new': '/new',
    '/reasoning': '/reasoning',
    '/reset': '/new'
  },
  categories: [],
  pairs: [
    ['/new', 'Start a new session'],
    ['/background', 'Run a prompt in the background'],
    ['/reasoning', 'Manage reasoning effort and display'],
    ['/help', 'Show available commands'],
    ['/model', 'Switch model for this session'],
    ['/compact', 'Toggle compact display mode'],
    ['/gif-search', 'Search for GIFs across providers']
  ],
  skillCount: 1,
  sub: {
    '/reasoning': ['none', 'low', 'high', 'show', 'hide']
  }
}

describe('getLocalSlashCompletion', () => {
  it('completes cached top-level slash commands and aliases locally', () => {
    expect(getLocalSlashCompletion('/he', catalog)).toEqual({
      items: [{ display: '/help', meta: 'Show available commands', text: 'help' }],
      replace_from: 1
    })

    expect(getLocalSlashCompletion('/re', catalog)).toEqual({
      items: [
        {
          display: '/reset',
          meta: 'Start a new session (alias for /new)',
          text: 'reset'
        },
        {
          display: '/reasoning',
          meta: 'Manage reasoning effort and display',
          text: 'reasoning'
        }
      ],
      replace_from: 1
    })
  })

  it('adds a trailing space for exact local command matches', () => {
    expect(getLocalSlashCompletion('/help', catalog)).toEqual({
      items: [{ display: '/help', meta: 'Show available commands', text: 'help ' }],
      replace_from: 1
    })

    expect(getLocalSlashCompletion('/gif-search', catalog)).toEqual({
      items: [{ display: '/gif-search', meta: 'Search for GIFs across providers', text: 'gif-search ' }],
      replace_from: 1
    })
  })

  it('completes static subcommands locally', () => {
    expect(getLocalSlashCompletion('/reasoning sh', catalog)).toEqual({
      items: [{ display: 'show', text: 'show' }],
      replace_from: 11
    })

    expect(getLocalSlashCompletion('/reasoning show', catalog)).toEqual({
      items: [],
      replace_from: 11
    })
  })

  it('falls back to the gateway for runtime subcommand sources and unknown slash prefixes', () => {
    expect(getLocalSlashCompletion('/model so', catalog)).toBeNull()
    expect(getLocalSlashCompletion('/plugin', catalog)).toBeNull()
  })
})
