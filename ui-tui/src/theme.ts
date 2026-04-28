export interface ThemeColors {
  gold: string
  amber: string
  bronze: string
  cornsilk: string
  dim: string
  completionBg: string
  completionCurrentBg: string

  label: string
  ok: string
  error: string
  warn: string

  prompt: string
  sessionLabel: string
  sessionBorder: string

  statusBg: string
  statusFg: string
  statusGood: string
  statusWarn: string
  statusBad: string
  statusCritical: string
  selectionBg: string

  diffAdded: string
  diffRemoved: string
  diffAddedWord: string
  diffRemovedWord: string

  shellDollar: string
}

export interface ThemeBrand {
  name: string
  icon: string
  prompt: string
  welcome: string
  goodbye: string
  tool: string
  helpHeader: string
}

export interface Theme {
  color: ThemeColors
  brand: ThemeBrand
  bannerLogo: string
  bannerHero: string
}

export interface FromSkinOptions {
  env?: NodeJS.ProcessEnv
  lightMode?: boolean
}

// ── Color math ───────────────────────────────────────────────────────

function parseHex(h: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(h)

  if (!m) {
    return null
  }

  const n = parseInt(m[1]!, 16)

  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
}

function mix(a: string, b: string, t: number) {
  const pa = parseHex(a)
  const pb = parseHex(b)

  if (!pa || !pb) {
    return a
  }

  const lerp = (i: 0 | 1 | 2) => Math.round(pa[i] + (pb[i] - pa[i]) * t)

  return '#' + ((1 << 24) | (lerp(0) << 16) | (lerp(1) << 8) | lerp(2)).toString(16).slice(1)
}

function relativeLuminance(h: string): number | null {
  const rgb = parseHex(h)

  if (!rgb) {
    return null
  }

  const linear = rgb.map((v) => {
    const c = v / 255

    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }) as [number, number, number]

  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
}

export function contrastRatio(foreground: string, background: string): number {
  const fg = relativeLuminance(foreground)
  const bg = relativeLuminance(background)

  // Invalid colors are treated as worst-case contrast. This keeps typos in a
  // skin from masquerading as readable simply because parsing failed.
  if (fg === null || bg === null) {
    return 1
  }

  const light = Math.max(fg, bg)
  const dark = Math.min(fg, bg)

  return (light + 0.05) / (dark + 0.05)
}

function ensureContrast(foreground: string, background: string, minimum = 4.5): string {
  if (contrastRatio(foreground, background) >= minimum) {
    return foreground
  }

  if (relativeLuminance(background) === null) {
    return foreground
  }

  const blackRatio = contrastRatio('#000000', background)
  const whiteRatio = contrastRatio('#FFFFFF', background)
  const target = blackRatio >= whiteRatio ? '#000000' : '#FFFFFF'

  if (relativeLuminance(foreground) === null) {
    return target
  }

  for (let i = 1; i <= 20; i += 1) {
    const candidate = mix(foreground, target, i / 20)

    if (contrastRatio(candidate, background) >= minimum) {
      return candidate
    }
  }

  return target
}

function minContrast(background: string, foregrounds: string[]): number {
  return Math.min(...foregrounds.map((foreground) => contrastRatio(foreground, background)))
}

function ensureBackgroundContrast(background: string, foregrounds: string[], minimum = 4.5): string {
  if (relativeLuminance(background) === null) {
    return background
  }

  if (minContrast(background, foregrounds) >= minimum) {
    return background
  }

  const blackRatio = minContrast('#000000', foregrounds)
  const whiteRatio = minContrast('#FFFFFF', foregrounds)
  const target = blackRatio >= whiteRatio ? '#000000' : '#FFFFFF'

  for (let i = 1; i <= 20; i += 1) {
    const candidate = mix(background, target, i / 20)

    if (minContrast(candidate, foregrounds) >= minimum) {
      return candidate
    }
  }

  return target
}

// ── Defaults ─────────────────────────────────────────────────────────

const BRAND: ThemeBrand = {
  name: 'Hermes Agent',
  icon: '⚕',
  prompt: '❯',
  welcome: 'Type your message or /help for commands.',
  goodbye: 'Goodbye! ⚕',
  tool: '┊',
  helpHeader: '(^_^)? Commands'
}

export const DARK_THEME: Theme = {
  color: {
    gold: '#FFD700',
    amber: '#FFBF00',
    bronze: '#CD7F32',
    cornsilk: '#FFF8DC',
    // Bumped from the old `#B8860B` darkgoldenrod (~53% luminance) which
    // read as barely-visible on dark terminals for long body text.  The
    // new value sits ~60% luminance — readable without losing the "muted /
    // secondary" semantic.  Field labels still use `label` (65%) which
    // stays brighter so hierarchy holds.
    dim: '#CC9B1F',
    completionBg: '#10161D',
    completionCurrentBg: '#322A16',

    label: '#DAA520',
    ok: '#4caf50',
    error: '#ef5350',
    warn: '#ffa726',

    prompt: '#FFF8DC',
    // sessionLabel/sessionBorder intentionally track the `dim` value — they
    // are "same role, same colour" by design.  fromSkin's banner_dim fallback
    // relies on this pairing (#11300).
    sessionLabel: '#CC9B1F',
    sessionBorder: '#CC9B1F',

    statusBg: '#1a1a2e',
    statusFg: '#C0C0C0',
    statusGood: '#8FBC8F',
    statusWarn: '#FFD700',
    statusBad: '#FF8C00',
    statusCritical: '#FF6B6B',
    selectionBg: '#3a3a55',

    diffAdded: 'rgb(220,255,220)',
    diffRemoved: 'rgb(255,220,220)',
    diffAddedWord: 'rgb(36,138,61)',
    diffRemovedWord: 'rgb(207,34,46)',
    shellDollar: '#4dabf7'
  },

  brand: BRAND,

  bannerLogo: '',
  bannerHero: ''
}

// Light-terminal palette: darker golds/ambers that stay legible on white
// backgrounds. Same shape as DARK_THEME so `fromSkin` still layers on top
// cleanly (#11300).
export const LIGHT_THEME: Theme = {
  color: {
    gold: '#8B6914',
    amber: '#A0651C',
    bronze: '#7A4F1F',
    cornsilk: '#3D2F13',
    dim: '#7A5A0F',
    completionBg: '#F5F5F5',
    completionCurrentBg: '#FFF8DC',

    label: '#7A5A0F',
    ok: '#2E7D32',
    error: '#C62828',
    warn: '#E65100',

    prompt: '#2B2014',
    sessionLabel: '#7A5A0F',
    sessionBorder: '#7A5A0F',

    statusBg: '#F5F5F5',
    statusFg: '#333333',
    statusGood: '#2E7D32',
    statusWarn: '#8B6914',
    statusBad: '#D84315',
    statusCritical: '#B71C1C',
    selectionBg: '#D4E4F7',

    diffAdded: 'rgb(200,240,200)',
    diffRemoved: 'rgb(240,200,200)',
    diffAddedWord: 'rgb(27,94,32)',
    diffRemovedWord: 'rgb(183,28,28)',
    shellDollar: '#1565C0'
  },

  brand: BRAND,

  bannerLogo: '',
  bannerHero: ''
}

// Pick light vs dark. Explicit `HERMES_TUI_LIGHT` wins; otherwise sniff
// `COLORFGBG` (set by XFCE Terminal, rxvt, Terminal.app, etc.) — last field is the
// background ANSI index; 7/15 are the "white" slots most light themes emit (#11300).
export function detectLightMode(env: NodeJS.ProcessEnv = process.env): boolean {
  const explicit = (env.HERMES_TUI_LIGHT ?? '').trim().toLowerCase()

  if (/^(?:1|true|yes|on)$/.test(explicit)) {
    return true
  }

  if (/^(?:0|false|no|off)$/.test(explicit)) {
    return false
  }

  const parts = (env.COLORFGBG ?? '').trim().split(';')
  const bg = Number(parts[parts.length - 1])

  return bg === 7 || bg === 15
}

export const DEFAULT_THEME: Theme = detectLightMode() ? LIGHT_THEME : DARK_THEME

function themeFor(lightMode: boolean): Theme {
  return lightMode ? LIGHT_THEME : DARK_THEME
}

// ── Skin → Theme ─────────────────────────────────────────────────────

// Skins are user-customizable, but the TUI must remain readable when macOS flips
// the terminal between light and dark appearances. fromSkin therefore treats
// invalid or low-contrast skin colors as requests to stay visually close while
// meeting the relevant foreground/background contrast threshold.
export function fromSkin(
  colors: Record<string, string>,
  branding: Record<string, string>,
  bannerLogo = '',
  bannerHero = '',
  toolPrefix = '',
  helpHeader = '',
  options: FromSkinOptions = {}
): Theme {
  const lightMode = options.lightMode ?? detectLightMode(options.env)
  const d = themeFor(lightMode)
  // Terminals do not expose an exact background color to Node. Use the Hermes
  // dark surface and plain white as conservative contrast targets for the two
  // appearance modes; custom skin surfaces are checked separately below.
  const terminalBg = lightMode ? '#FFFFFF' : '#10161D'
  const c = (k: string) => colors[k]
  const onTerminal = (value: string, minimum = 4.5) => ensureContrast(value, terminalBg, minimum)
  const onBg = (value: string, bg: string, minimum = 4.5) => ensureContrast(value, bg, minimum)

  const terminalColor = (value: string | undefined, fallback: string, minimum = 4.5) =>
    value ? onTerminal(value, minimum) : fallback

  const backgroundColor = (value: string | undefined, fallback: string, foregrounds: string[], minimum = 4.5) => {
    const background = value && relativeLuminance(value) !== null ? value : fallback

    return ensureBackgroundContrast(background, foregrounds, minimum)
  }

  const surfaceColor = (value: string | undefined, fallback: string, bg: string, fallbackBg: string, minimum = 4.5) =>
    // Empty strings are treated as absent skin values; only real values or an
    // explicit surface change trigger contrast correction against the surface.
    (value || bg !== fallbackBg) ? onBg(value ?? fallback, bg, minimum) : fallback

  // Text-like values use WCAG AA 4.5:1 by default; borders and deliberately
  // muted secondary text can sit lower because they are hierarchy/decoration,
  // not the only readable content.
  const amber = c('ui_accent') ?? c('banner_accent')
  const dim = c('banner_dim')
  const label = terminalColor(c('ui_label'), d.color.label)
  const readableDim = terminalColor(dim, d.color.dim, 3.8)

  const completionBg = backgroundColor(c('completion_menu_bg'), d.color.completionBg, [label, readableDim])

  const completionCurrentBg = backgroundColor(
    c('completion_menu_current_bg'),
    d.color.completionCurrentBg,
    [label, readableDim]
  )

  const statusBgValue = c('status_bar_bg')
  const statusBg = statusBgValue && relativeLuminance(statusBgValue) !== null ? statusBgValue : d.color.statusBg

  return {
    color: {
      gold: terminalColor(c('banner_title'), d.color.gold),
      amber: terminalColor(amber, d.color.amber),
      bronze: terminalColor(c('banner_border'), d.color.bronze, 3),
      cornsilk: terminalColor(c('banner_text'), d.color.cornsilk),
      dim: readableDim,
      completionBg,
      completionCurrentBg,

      label,
      ok: terminalColor(c('ui_ok'), d.color.ok),
      error: terminalColor(c('ui_error'), d.color.error),
      warn: terminalColor(c('ui_warn'), d.color.warn),

      prompt: terminalColor(c('prompt') ?? c('banner_text'), d.color.prompt),
      sessionLabel: terminalColor(c('session_label') ?? dim, d.color.sessionLabel, 3.8),
      sessionBorder: terminalColor(c('session_border') ?? dim, d.color.sessionBorder, 3),

      statusBg,
      statusFg: surfaceColor(c('status_bar_text') ?? c('banner_text'), d.color.statusFg, statusBg, d.color.statusBg),
      statusGood: surfaceColor(c('status_bar_good') ?? c('ui_ok'), d.color.statusGood, statusBg, d.color.statusBg),
      statusWarn: surfaceColor(c('status_bar_warn') ?? c('ui_warn'), d.color.statusWarn, statusBg, d.color.statusBg),
      statusBad: surfaceColor(c('status_bar_bad'), d.color.statusBad, statusBg, d.color.statusBg),
      statusCritical: surfaceColor(c('status_bar_critical'), d.color.statusCritical, statusBg, d.color.statusBg),
      selectionBg: c('selection_bg') ?? d.color.selectionBg,

      diffAdded: d.color.diffAdded,
      diffRemoved: d.color.diffRemoved,
      diffAddedWord: d.color.diffAddedWord,
      diffRemovedWord: d.color.diffRemovedWord,
      shellDollar: terminalColor(c('shell_dollar'), d.color.shellDollar)
    },

    brand: {
      name: branding.agent_name ?? d.brand.name,
      icon: d.brand.icon,
      prompt: branding.prompt_symbol ?? d.brand.prompt,
      welcome: branding.welcome ?? d.brand.welcome,
      goodbye: branding.goodbye ?? d.brand.goodbye,
      tool: toolPrefix || d.brand.tool,
      helpHeader: branding.help_header ?? (helpHeader || d.brand.helpHeader)
    },

    bannerLogo,
    bannerHero
  }
}
