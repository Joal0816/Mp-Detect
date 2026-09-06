# components/theme.py - Design system constants
"""MP Detect Design System — Dark Scientific Instrument UI"""
import flet as ft


# ── Color Palette ──────────────────────────────────────────────────────

class Colors:
    # Backgrounds
    BG_PRIMARY = "#08080D"
    BG_SURFACE = "#0F1018"
    BG_CARD = "#16171F"
    BG_ELEVATED = "#1C1D27"

    # Glass
    GLASS_BG = "rgba(255,255,255,0.03)"
    GLASS_BORDER = "rgba(255,255,255,0.06)"
    GLASS_BORDER_ACTIVE = "rgba(0,229,255,0.3)"

    # Accents
    ACCENT_CYAN = "#00E5FF"
    ACCENT_PURPLE = "#7C4DFF"
    ACCENT_GRADIENT_START = "#00E5FF"
    ACCENT_GRADIENT_END = "#7C4DFF"

    # Status
    SUCCESS = "#00E676"
    WARNING = "#FFD740"
    ERROR = "#FF5252"

    # Text
    TEXT_PRIMARY = "#F0F0F5"
    TEXT_SECONDARY = "#6B6B7B"
    TEXT_MUTED = "#3A3A4A"


# ── Gradients ──────────────────────────────────────────────────────────

GRADIENT_CYAN = ft.LinearGradient(
    begin=ft.Alignment(-1, -1),
    end=ft.Alignment(1, 1),
    colors=[Colors.ACCENT_CYAN, Colors.ACCENT_PURPLE],
)

GRADIENT_SUBTLE = ft.LinearGradient(
    begin=ft.Alignment(-1, -1),
    end=ft.Alignment(1, 1),
    colors=[Colors.BG_CARD, Colors.BG_ELEVATED],
)

GRADIENT_CYAN_HORIZONTAL = ft.LinearGradient(
    begin=ft.Alignment(-1, 0),
    end=ft.Alignment(1, 0),
    colors=[Colors.ACCENT_CYAN, Colors.ACCENT_PURPLE],
)


# ── Text Styles ────────────────────────────────────────────────────────

TEXT_HEADING1 = ft.TextStyle(size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY)
TEXT_HEADING2 = ft.TextStyle(size=18, weight=ft.FontWeight.W_600, color=Colors.TEXT_PRIMARY)
TEXT_BODY = ft.TextStyle(size=14, color=Colors.TEXT_PRIMARY)
TEXT_CAPTION = ft.TextStyle(size=12, color=Colors.TEXT_SECONDARY)
TEXT_MONO = ft.TextStyle(size=12, font_family="monospace", color=Colors.ACCENT_CYAN)


# ── Reusable Helpers ───────────────────────────────────────────────────

def glass_container(content, **kwargs):
    """Create a glassmorphism container."""
    return ft.Container(
        content=content,
        bgcolor=Colors.GLASS_BG,
        border=ft.Border.all(1, Colors.GLASS_BORDER),
        border_radius=kwargs.get("border_radius", 12),
        padding=kwargs.get("padding", 16),
        **{k: v for k, v in kwargs.items() if k not in ("border_radius", "padding")},
    )


def card(content, **kwargs):
    """Create a styled card."""
    return ft.Container(
        content=content,
        bgcolor=Colors.BG_CARD,
        border=ft.Border.all(1, Colors.GLASS_BORDER),
        border_radius=kwargs.get("border_radius", 12),
        padding=kwargs.get("padding", 16),
        **{k: v for k, v in kwargs.items() if k not in ("border_radius", "padding")},
    )


def gradient_button(text, on_click=None, icon=None, width=None):
    """Create a gradient-accented button."""
    return ft.Button(
        text,
        icon=icon,
        on_click=on_click,
        bgcolor=Colors.ACCENT_CYAN,
        color=Colors.BG_PRIMARY,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
        width=width,
    )


def outline_button(text, on_click=None, icon=None, width=None):
    """Create an outline button."""
    return ft.Button(
        text,
        icon=icon,
        on_click=on_click,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            side=ft.BorderSide(1, Colors.GLASS_BORDER),
            bgcolor=Colors.GLASS_BG,
            color=Colors.TEXT_PRIMARY,
        ),
        width=width,
    )


def icon_button(icon, on_click=None, tooltip=None, color=None):
    """Create a styled icon button."""
    return ft.IconButton(
        icon=icon,
        on_click=on_click,
        tooltip=tooltip,
        icon_color=color or Colors.TEXT_SECONDARY,
        icon_size=20,
    )


def status_chip(text, color=Colors.ACCENT_CYAN):
    """Create a status chip."""
    return ft.Container(
        content=ft.Text(text, size=11, color=color, weight=ft.FontWeight.W_500),
        bgcolor=ft.Colors.with_opacity(0.15, color),
        border_radius=16,
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
    )


def metric_card(label, value, color=Colors.ACCENT_CYAN):
    """Create a metric display card."""
    return ft.Container(
        content=ft.Column([
            ft.Text(value, size=20, weight=ft.FontWeight.BOLD, color=color),
            ft.Text(label, size=11, color=Colors.TEXT_SECONDARY),
        ], spacing=2),
        bgcolor=Colors.BG_CARD,
        border=ft.Border.all(1, Colors.GLASS_BORDER),
        border_radius=10,
        padding=12,
        width=120,
    )
