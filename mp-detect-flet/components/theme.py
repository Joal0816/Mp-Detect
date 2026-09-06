# components/theme.py - Modern glassmorphic design system
"""MP Detect Design System — Dark Scientific Instrument UI with Glassmorphism."""
import flet as ft


# ── Color Palette ──────────────────────────────────────────────────────

class Colors:
    # Backgrounds
    BG_PRIMARY = "#0A0A0F"
    BG_SURFACE = "#12121A"
    BG_CARD = "#1A1A25"
    BG_ELEVATED = "#222230"

    # Glass
    GLASS_BG = "rgba(26,26,37,0.75)"
    GLASS_BORDER = "rgba(255,255,255,0.08)"
    GLASS_BORDER_ACTIVE = "rgba(0,242,254,0.3)"

    # Accents (Electric Cyan + Neon Green)
    ACCENT_CYAN = "#00F2FE"
    ACCENT_GREEN = "#00E676"
    ACCENT_PURPLE = "#7C4DFF"
    ACCENT_GRADIENT_START = "#00F2FE"
    ACCENT_GRADIENT_END = "#7C4DFF"

    # Status
    SUCCESS = "#00E676"
    WARNING = "#FFD740"
    ERROR = "#FF5252"

    # Text
    TEXT_PRIMARY = "#F5F5F7"
    TEXT_SECONDARY = "#8E8E93"
    TEXT_MUTED = "#48484A"


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


# ── Reusable Components ────────────────────────────────────────────────

def glass_container(content, blur: int = 20, opacity: float = 0.75, **kwargs):
    """Create a glassmorphism container with backdrop blur effect."""
    return ft.Container(
        content=content,
        bgcolor=ft.Colors.with_opacity(opacity, Colors.BG_CARD),
        border=ft.Border.all(1, Colors.GLASS_BORDER),
        border_radius=kwargs.get("border_radius", 16),
        padding=kwargs.get("padding", 16),
        **{k: v for k, v in kwargs.items() if k not in ("border_radius", "padding")},
    )


def floating_hud(text: str, icon: ft.Icons = None, color: str = None):
    """Create a floating HUD pill with glassmorphism."""
    color = color or Colors.ACCENT_CYAN
    controls = []
    if icon:
        controls.append(ft.Icon(icon, size=14, color=color))
    controls.append(ft.Text(text, size=11, color=color, font_family="monospace",
                           weight=ft.FontWeight.W_500))
    
    return ft.Container(
        content=ft.Row(controls, spacing=4),
        bgcolor=ft.Colors.with_opacity(0.6, Colors.BG_CARD),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.2, color)),
        border_radius=20,
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
    )


def pill_button(text: str, on_click=None, icon: ft.Icons = None, 
                color: str = None, width: int = None):
    """Create a pill-shaped modern button."""
    color = color or Colors.ACCENT_CYAN
    return ft.Button(
        text, icon=icon, on_click=on_click,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=24),
            bgcolor=ft.Colors.with_opacity(0.15, color),
            color=color,
        ),
        height=44, width=width,
    )


def card(content, blur: int = 20, **kwargs):
    """Create a styled card with glassmorphism."""
    return ft.Container(
        content=content,
        bgcolor=ft.Colors.with_opacity(0.75, Colors.BG_CARD),
        border=ft.Border.all(1, Colors.GLASS_BORDER),
        border_radius=kwargs.get("border_radius", 16),
        padding=kwargs.get("padding", 16),
        **{k: v for k, v in kwargs.items() if k not in ("border_radius", "padding")},
    )


def metric_card(label: str, value: str, color: str = None, icon: ft.Icons = None):
    """Create a modern metric display card."""
    color = color or Colors.ACCENT_CYAN
    controls = []
    if icon:
        controls.append(ft.Icon(icon, size=20, color=color))
    controls.append(ft.Column([
        ft.Text(value, size=24, weight=ft.FontWeight.BOLD, color=color),
        ft.Text(label, size=11, color=Colors.TEXT_SECONDARY),
    ], spacing=2))
    
    return ft.Container(
        content=ft.Row(controls, spacing=8) if icon else ft.Column([
            ft.Text(value, size=24, weight=ft.FontWeight.BOLD, color=color),
            ft.Text(label, size=11, color=Colors.TEXT_SECONDARY),
        ], spacing=2),
        bgcolor=ft.Colors.with_opacity(0.75, Colors.BG_CARD),
        border=ft.Border.all(1, Colors.GLASS_BORDER),
        border_radius=12,
        padding=16,
    )


def confidence_badge(confidence: float):
    """Create a color-coded confidence badge."""
    if confidence >= 0.85:
        color = Colors.SUCCESS
        label = "High"
    elif confidence >= 0.60:
        color = Colors.WARNING
        label = "Medium"
    else:
        color = Colors.ERROR
        label = "Low"
    
    return ft.Container(
        content=ft.Text(f"{confidence:.0%} {label}", size=10, 
                       color=color, weight=ft.FontWeight.W_500),
        bgcolor=ft.Colors.with_opacity(0.15, color),
        border_radius=12,
        padding=ft.Padding.symmetric(horizontal=8, vertical=4),
    )


def morphology_tag(morphology: str):
    """Create a morphology tag badge."""
    colors = {
        "Fiber": Colors.ACCENT_CYAN,
        "Fragment": Colors.ACCENT_GREEN,
        "Film": Colors.ACCENT_PURPLE,
        "Foam": Colors.WARNING,
        "Pellet": Colors.ACCENT_CYAN,
        "Bead": Colors.ACCENT_GREEN,
    }
    color = colors.get(morphology, Colors.TEXT_SECONDARY)
    
    return ft.Container(
        content=ft.Text(morphology, size=10, color=color, weight=ft.FontWeight.W_500),
        bgcolor=ft.Colors.with_opacity(0.15, color),
        border_radius=12,
        padding=ft.Padding.symmetric(horizontal=8, vertical=4),
    )


# ── Text Styles ────────────────────────────────────────────────────────

TEXT_HEADING1 = ft.TextStyle(size=24, weight=ft.FontWeight.BOLD, color=Colors.TEXT_PRIMARY)
TEXT_HEADING2 = ft.TextStyle(size=18, weight=ft.FontWeight.W_600, color=Colors.TEXT_PRIMARY)
TEXT_BODY = ft.TextStyle(size=14, color=Colors.TEXT_PRIMARY)
TEXT_CAPTION = ft.TextStyle(size=12, color=Colors.TEXT_SECONDARY)
TEXT_MONO = ft.TextStyle(size=12, font_family="monospace", color=Colors.ACCENT_CYAN)
