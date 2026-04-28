import { describe, expect, it } from 'vitest'

import { contrastRatio, DARK_THEME, DEFAULT_THEME, detectLightMode, fromSkin, LIGHT_THEME } from '../theme.js'

describe('DEFAULT_THEME', () => {
  it('has brand defaults', () => {
    expect(DEFAULT_THEME.brand.name).toBe('Hermes Agent')
    expect(DEFAULT_THEME.brand.prompt).toBe('❯')
    expect(DEFAULT_THEME.brand.tool).toBe('┊')
  })

  it('has dark palette anchors and tracks the detected mode', () => {
    expect(DARK_THEME.color.gold).toBe('#FFD700')
    expect(DARK_THEME.color.error).toBe('#ef5350')
    expect(DARK_THEME.color.completionBg).toBe('#10161D')
    expect(DARK_THEME.color.completionCurrentBg).toBe('#322A16')
    expect(DEFAULT_THEME.color.error).toBe(detectLightMode() ? LIGHT_THEME.color.error : DARK_THEME.color.error)
  })
})

describe('LIGHT_THEME', () => {
  it('avoids bright-yellow accents unreadable on white backgrounds (#11300)', () => {
    expect(LIGHT_THEME.color.gold).not.toBe('#FFD700')
    expect(LIGHT_THEME.color.amber).not.toBe('#FFBF00')
    expect(LIGHT_THEME.color.dim).not.toBe('#B8860B')
    expect(LIGHT_THEME.color.statusWarn).not.toBe('#FFD700')
  })

  it('keeps the same shape as DARK_THEME', () => {
    expect(Object.keys(LIGHT_THEME.color).sort()).toEqual(Object.keys(DARK_THEME.color).sort())
    expect(LIGHT_THEME.brand).toEqual(DARK_THEME.brand)
  })
})

describe('DEFAULT_THEME aliasing', () => {
  it('tracks the detected terminal mode', () => {
    expect(DEFAULT_THEME).toBe(detectLightMode() ? LIGHT_THEME : DARK_THEME)
  })
})

describe('detectLightMode', () => {
  it('returns false on empty env', () => {
    expect(detectLightMode({})).toBe(false)
  })

  it('honors HERMES_TUI_LIGHT on/off', () => {
    expect(detectLightMode({ HERMES_TUI_LIGHT: '1' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'true' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'on' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: '0' })).toBe(false)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'off' })).toBe(false)
  })

  it('sniffs COLORFGBG bg slots 7 and 15 as light (#11300)', () => {
    expect(detectLightMode({ COLORFGBG: '0;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;default;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;7' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '15;0' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '7;default;0' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '15;' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '15' })).toBe(true)
  })

  it('lets HERMES_TUI_LIGHT=0 override a light COLORFGBG', () => {
    expect(detectLightMode({ COLORFGBG: '0;15', HERMES_TUI_LIGHT: '0' })).toBe(false)
  })
})

describe('fromSkin', () => {
  it('treats invalid hex as worst-case contrast input', () => {
    expect(contrastRatio('not-a-color', '#FFFFFF')).toBe(1)
  })

  it('overrides banner colors', () => {
    expect(fromSkin({ banner_title: '#FF0000' }, {}, '', '', '', '', { lightMode: false }).color.gold).toBe('#FF0000')
  })

  it('preserves unset colors', () => {
    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.amber).toBe(DEFAULT_THEME.color.amber)
  })

  it('overrides branding', () => {
    const { brand } = fromSkin({}, { agent_name: 'TestBot', prompt_symbol: '$' })
    expect(brand.name).toBe('TestBot')
    expect(brand.prompt).toBe('$')
  })

  it('defaults for empty skin', () => {
    expect(fromSkin({}, {}).color).toEqual(DEFAULT_THEME.color)
    expect(fromSkin({}, {}).brand.icon).toBe(DEFAULT_THEME.brand.icon)
  })

  it('preserves diff and selection colors unless explicitly skinned', () => {
    const { color } = fromSkin({ selection_bg: '#123456' }, {}, '', '', '', '', { lightMode: false })

    expect(color.selectionBg).toBe('#123456')
    expect(color.diffAdded).toBe(DARK_THEME.color.diffAdded)
    expect(color.diffRemoved).toBe(DARK_THEME.color.diffRemoved)
    expect(color.diffAddedWord).toBe(DARK_THEME.color.diffAddedWord)
    expect(color.diffRemovedWord).toBe(DARK_THEME.color.diffRemovedWord)
  })

  it('uses the detected default theme when the options argument is omitted', () => {
    expect(fromSkin({}, {}).color.statusBg).toBe(DEFAULT_THEME.color.statusBg)
    expect(fromSkin({}, {}).color.completionBg).toBe(DEFAULT_THEME.color.completionBg)
  })

  it('passes banner logo/hero', () => {
    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerLogo).toBe('LOGO')
    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerHero).toBe('HERO')
  })

  it('maps ui_ color keys + cascades to status', () => {
    const { color } = fromSkin({ ui_ok: '#008000' }, {}, '', '', '', '', { lightMode: false })
    expect(contrastRatio(color.ok, '#10161D')).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.statusGood, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('keeps status text readable when a skin only overrides status background', () => {
    const { color } = fromSkin({ status_bar_bg: '#FFFFFF' }, {}, '', '', '', '', { lightMode: false })

    expect(color.statusBg).toBe('#FFFFFF')
    expect(contrastRatio(color.statusFg, color.statusBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.statusGood, color.statusBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.statusWarn, color.statusBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.statusBad, color.statusBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.statusCritical, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('keeps explicit status text when it already contrasts against explicit status background', () => {
    const { color } = fromSkin(
      { status_bar_bg: '#FFFFFF', status_bar_text: '#333333' },
      {},
      '',
      '',
      '',
      '',
      { lightMode: false }
    )

    expect(color.statusBg).toBe('#FFFFFF')
    expect(color.statusFg).toBe('#333333')
    expect(contrastRatio(color.statusFg, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('keeps explicit status text readable against explicit status background', () => {
    const { color } = fromSkin(
      { status_bar_bg: '#FFFFFF', status_bar_text: '#FFF8DC' },
      {},
      '',
      '',
      '',
      '',
      { lightMode: false }
    )

    expect(color.statusBg).toBe('#FFFFFF')
    expect(color.statusFg).not.toBe('#FFF8DC')
    expect(contrastRatio(color.statusFg, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('keeps explicit status text readable against the default status background', () => {
    const { color } = fromSkin({ status_bar_text: '#10161D' }, {}, '', '', '', '', { lightMode: false })

    expect(color.statusBg).toBe(DARK_THEME.color.statusBg)
    expect(color.statusFg).not.toBe('#10161D')
    expect(contrastRatio(color.statusFg, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('keeps completion surfaces readable in both built-in modes', () => {
    for (const lightMode of [false, true]) {
      const { color } = fromSkin({}, {}, '', '', '', '', { lightMode })

      expect(contrastRatio(color.label, color.completionBg)).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(color.dim, color.completionBg)).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(color.label, color.completionCurrentBg)).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(color.dim, color.completionCurrentBg)).toBeGreaterThanOrEqual(4.5)
    }
  })

  it('keeps completion menu text readable when only the completion background is overridden', () => {
    const { color } = fromSkin({ completion_menu_bg: '#DAA520' }, {}, '', '', '', '', { lightMode: false })

    expect(color.completionBg).not.toBe('#DAA520')
    expect(contrastRatio(color.label, color.completionBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.dim, color.completionBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('keeps current completion row readable when only its background is overridden', () => {
    const { color } = fromSkin({ completion_menu_current_bg: '#DAA520' }, {}, '', '', '', '', { lightMode: false })

    expect(color.completionCurrentBg).not.toBe('#DAA520')
    expect(contrastRatio(color.label, color.completionCurrentBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.dim, color.completionCurrentBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('falls back for invalid completion background colors', () => {
    const { color } = fromSkin(
      { completion_menu_bg: 'not-a-color', completion_menu_current_bg: 'also-not-a-color' },
      {},
      '',
      '',
      '',
      '',
      { lightMode: false }
    )

    expect(color.completionBg).toBe(DARK_THEME.color.completionBg)
    expect(color.completionCurrentBg).toBe(DARK_THEME.color.completionCurrentBg)
  })

  it('uses env-driven light mode detection when no explicit option is provided', () => {
    const { color } = fromSkin({ banner_title: '#FFD700' }, {}, '', '', '', '', { env: { COLORFGBG: '0;15' } })

    expect(color.gold).not.toBe('#FFD700')
    expect(contrastRatio(color.gold, '#FFFFFF')).toBeGreaterThanOrEqual(4.5)
  })

  it('honors HERMES_TUI_LIGHT=0 in fromSkin env options', () => {
    const { color } = fromSkin(
      { banner_title: '#FFD700' },
      {},
      '',
      '',
      '',
      '',
      { env: { COLORFGBG: '0;15', HERMES_TUI_LIGHT: '0' } }
    )

    expect(color.gold).toBe('#FFD700')
    expect(contrastRatio(color.gold, '#10161D')).toBeGreaterThanOrEqual(4.5)
  })

  it('lets explicit lightMode override env-driven detection', () => {
    const { color } = fromSkin(
      { banner_title: '#FFD700' },
      {},
      '',
      '',
      '',
      '',
      { env: { COLORFGBG: '0;15' }, lightMode: false }
    )

    expect(color.gold).toBe('#FFD700')
    expect(contrastRatio(color.gold, '#10161D')).toBeGreaterThanOrEqual(4.5)
  })

  it('lowers dark skin foregrounds when the terminal is light', () => {
    const darkSkinColors = {
      banner_title: '#1A1A1A',
      banner_text: '#101010',
      prompt: '#101010',
      ui_label: '#1A1A1A'
    }

    const { color } = fromSkin(darkSkinColors, {}, '', '', '', '', { lightMode: true })
    const lightTerminalBg = '#FFFFFF'

    expect(color.gold).toBe('#1A1A1A')
    expect(color.cornsilk).toBe('#101010')
    expect(color.prompt).toBe('#101010')
    expect(contrastRatio(color.gold, lightTerminalBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.cornsilk, lightTerminalBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.prompt, lightTerminalBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.label, lightTerminalBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('moves low-contrast status text toward a readable foreground', () => {
    const { color } = fromSkin(
      { status_bar_bg: '#777777', status_bar_text: '#777777' },
      {},
      '',
      '',
      '',
      '',
      { lightMode: true }
    )

    expect(color.statusFg).not.toBe('#777777')
    expect(contrastRatio(color.statusFg, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('treats an empty explicit status text as absent when status background changes', () => {
    const { color } = fromSkin({ status_bar_bg: '#FFFFFF', status_bar_text: '' }, {}, '', '', '', '', { lightMode: false })

    expect(color.statusFg).not.toBe('')
    expect(contrastRatio(color.statusFg, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('replaces invalid skin foreground colors with readable fallbacks', () => {
    const { color } = fromSkin({ banner_title: 'not-a-color' }, {}, '', '', '', '', { lightMode: true })

    expect(color.gold).toBe('#000000')
    expect(contrastRatio(color.gold, '#FFFFFF')).toBeGreaterThanOrEqual(4.5)
  })

  it('falls back to the base status background when a skin status background is invalid', () => {
    const { color } = fromSkin(
      { status_bar_bg: 'not-a-color', status_bar_text: '#FFF8DC' },
      {},
      '',
      '',
      '',
      '',
      { lightMode: false }
    )

    expect(color.statusBg).toBe(DARK_THEME.color.statusBg)
    expect(contrastRatio(color.statusFg, color.statusBg)).toBeGreaterThanOrEqual(4.5)
  })

  it('applies lower contrast thresholds for decorative border and dim values', () => {
    const { color } = fromSkin(
      { banner_border: '#936723', banner_dim: '#8B7355', session_border: '#936723' },
      {},
      '',
      '',
      '',
      '',
      { lightMode: false }
    )

    const darkTerminalBg = '#10161D'

    expect(color.bronze).toBe('#936723')
    expect(color.dim).toBe('#8B7355')
    expect(color.sessionBorder).toBe('#936723')
    expect(contrastRatio(color.bronze, darkTerminalBg)).toBeGreaterThanOrEqual(3)
    expect(contrastRatio(color.dim, darkTerminalBg)).toBeGreaterThanOrEqual(3.8)
    expect(contrastRatio(color.sessionBorder, darkTerminalBg)).toBeGreaterThanOrEqual(3)
  })

  it('keeps warm light skins intact on light terminals', () => {
    const warmLightColors = {
      banner_border: '#8B6914',
      banner_title: '#5C3D11',
      banner_accent: '#8B4513',
      banner_dim: '#8B7355',
      banner_text: '#2C1810',
      ui_accent: '#8B4513',
      ui_label: '#5C3D11',
      prompt: '#2C1810',
      completion_menu_bg: '#F5EFE0',
      completion_menu_current_bg: '#E8DCC8',
      status_bar_bg: '#F5F0E8'
    }

    const { color } = fromSkin(warmLightColors, {}, '', '', '', '', { lightMode: true })

    expect(color.gold).toBe('#5C3D11')
    expect(color.cornsilk).toBe('#2C1810')
    expect(color.prompt).toBe('#2C1810')
    expect(color.statusBg).toBe('#F5F0E8')
  })

  it('lifts warm light skin foregrounds when the terminal is dark', () => {
    const warmLightColors = {
      banner_title: '#5C3D11',
      banner_accent: '#8B4513',
      banner_dim: '#8B7355',
      banner_text: '#2C1810',
      ui_label: '#5C3D11',
      prompt: '#2C1810'
    }

    const { color } = fromSkin(warmLightColors, {}, '', '', '', '', { lightMode: false })
    const darkTerminalBg = '#10161D'

    expect(color.gold).not.toBe('#5C3D11')
    expect(color.cornsilk).not.toBe('#2C1810')
    expect(color.prompt).not.toBe('#2C1810')
    expect(contrastRatio(color.gold, darkTerminalBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.cornsilk, darkTerminalBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.prompt, darkTerminalBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.label, darkTerminalBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(color.dim, darkTerminalBg)).toBeGreaterThanOrEqual(3.8)
  })
})
