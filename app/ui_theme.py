"""Meeting Scribe look and feel: light-blue pill buttons that turn dark blue on hover.

Icons are drawn with QPainter at runtime, so the app ships no image files and loads nothing
from the network.
"""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QPushButton, QWidget

APP_NAME = "Meeting Scribe"

BACKGROUND_TOP = "#F4F8FC"
BACKGROUND_BOTTOM = "#DCE8F5"
BUTTON_LIGHT = "#DCEBFA"
BUTTON_BORDER = "#B9D3EE"
BUTTON_TEXT = "#1E3A5F"
BUTTON_HOVER = "#24508F"
BUTTON_PRESSED = "#1B3E70"
BUTTON_HOVER_TEXT = "#FFFFFF"
DISABLED_BACKGROUND = "#EEF2F6"
DISABLED_TEXT = "#9AA8B8"
RECORD_RED = "#D93025"
RECORD_LIGHT = "#D4F5D4"
RECORD_BORDER = "#A8DDA8"
RECORD_TEXT = "#1E5631"
RECORD_DARK = "#2E7D32"
PAUSE_LIGHT = "#FFF4C2"
PAUSE_BORDER = "#EED98A"
PAUSE_TEXT = "#6B5300"
PAUSE_DARK = "#B98900"
STOP_LIGHT = "#FFD9D9"
STOP_BORDER = "#F0AEAE"
STOP_TEXT = "#7A1C1C"
STOP_DARK = "#C62828"
TITLE_BLUE = "#24508F"
CARD_BORDER = "#C9DAEC"

ICON_SIZE = 30

APP_STYLESHEET = f"""
QWidget {{
    font-family: "Segoe UI";
    font-size: 10pt;
    color: {BUTTON_TEXT};
}}
QMainWindow, QWidget#sessionPage {{
    background: {BACKGROUND_TOP};
}}
QPushButton {{
    background: {BUTTON_LIGHT};
    color: {BUTTON_TEXT};
    border: 1px solid {BUTTON_BORDER};
    border-radius: 15px;
    padding: 6px 16px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {BUTTON_HOVER};
    border-color: {BUTTON_HOVER};
    color: {BUTTON_HOVER_TEXT};
}}
QPushButton:pressed {{
    background: {BUTTON_PRESSED};
    border-color: {BUTTON_PRESSED};
    color: {BUTTON_HOVER_TEXT};
}}
QPushButton:focus {{
    border: 2px solid {BUTTON_HOVER};
}}
QPushButton:disabled {{
    background: {DISABLED_BACKGROUND};
    border-color: {DISABLED_BACKGROUND};
    color: {DISABLED_TEXT};
}}
QPushButton#homeAction {{
    min-height: 62px;
    max-height: 62px;
    border-radius: 32px;
    font-size: 16pt;
    padding: 0 32px;
    background: {BUTTON_LIGHT};
}}
QPushButton#homeAction:hover, QPushButton#homeAction:pressed {{
    background: {BUTTON_HOVER};
    border-color: {BUTTON_HOVER};
    color: {BUTTON_HOVER_TEXT};
}}
QPushButton#recordButton {{
    background: {RECORD_LIGHT};
    border-color: {RECORD_BORDER};
    color: {RECORD_TEXT};
}}
QPushButton#recordButton:hover, QPushButton#recordButton:pressed {{
    background: {RECORD_DARK};
    border-color: {RECORD_DARK};
    color: #FFFFFF;
}}
QPushButton#recordButton:disabled {{
    background: #EEF6EE;
    border-color: #EEF6EE;
    color: #A9BFA9;
}}
QPushButton#pauseButton {{
    background: {PAUSE_LIGHT};
    border-color: {PAUSE_BORDER};
    color: {PAUSE_TEXT};
}}
QPushButton#pauseButton:hover, QPushButton#pauseButton:pressed {{
    background: {PAUSE_DARK};
    border-color: {PAUSE_DARK};
    color: #FFFFFF;
}}
QPushButton#pauseButton:disabled {{
    background: #FBF7E8;
    border-color: #FBF7E8;
    color: #C4B98F;
}}
QPushButton#stopButton {{
    background: {STOP_LIGHT};
    border-color: {STOP_BORDER};
    color: {STOP_TEXT};
}}
QPushButton#stopButton:hover, QPushButton#stopButton:pressed {{
    background: {STOP_DARK};
    border-color: {STOP_DARK};
    color: #FFFFFF;
}}
QPushButton#stopButton:disabled {{
    background: #FBEEEE;
    border-color: #FBEEEE;
    color: #CDA4A4;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BUTTON_BORDER};
    border-radius: 4px;
    background: #FFFFFF;
}}
QCheckBox::indicator:checked {{
    background: {BUTTON_HOVER};
    border-color: {BUTTON_HOVER};
}}
QCheckBox::indicator:disabled {{
    background: {DISABLED_BACKGROUND};
    border-color: {DISABLED_BACKGROUND};
}}
QLabel#modelStatus {{
    color: #5B6F86;
    font-size: 10pt;
}}
QGroupBox {{
    background: #FFFFFF;
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 4px;
    color: {TITLE_BLUE};
}}
QComboBox, QLineEdit {{
    background: #FFFFFF;
    border: 1px solid {BUTTON_BORDER};
    border-radius: 8px;
    padding: 4px 8px;
}}
QComboBox:disabled, QLineEdit:disabled {{
    background: {DISABLED_BACKGROUND};
    color: {DISABLED_TEXT};
}}
QTextEdit#transcript {{
    background: #FFFFFF;
    color: #1F2A37;
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 14px;
    font-size: 11pt;
    selection-background-color: {BUTTON_HOVER};
    selection-color: #FFFFFF;
}}
QTextEdit#log {{
    background: #FFFFFF;
    color: #4A5A6E;
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    font-family: "Cascadia Mono", "Consolas";
    font-size: 9pt;
}}
QToolButton#advancedToggle {{
    border: none;
    color: {TITLE_BLUE};
    font-weight: 600;
    padding: 2px 0;
}}
QLabel#appTitle {{
    color: {TITLE_BLUE};
    font-size: 20pt;
    font-weight: 600;
}}
QSplitter::handle {{
    background: transparent;
}}
"""

IconPainter = Callable[[QPainter, QRectF, QColor], None]


def _pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _paint_mic(painter: QPainter, rect: QRectF, color: QColor) -> None:
    w, h = rect.width(), rect.height()
    x, y = rect.x(), rect.y()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawRoundedRect(QRectF(x + w * 0.36, y + h * 0.08, w * 0.28, h * 0.5), w * 0.14, w * 0.14)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, max(1.5, w * 0.07)))
    painter.drawArc(QRectF(x + w * 0.24, y + h * 0.26, w * 0.52, h * 0.46), 180 * 16, 180 * 16)
    painter.drawLine(QPointF(x + w * 0.5, y + h * 0.72), QPointF(x + w * 0.5, y + h * 0.88))
    painter.drawLine(QPointF(x + w * 0.36, y + h * 0.9), QPointF(x + w * 0.64, y + h * 0.9))


def _paint_record(painter: QPainter, rect: QRectF, color: QColor) -> None:
    w, h = rect.width(), rect.height()
    x, y = rect.x(), rect.y()
    painter.setPen(_pen(color, max(1.5, w * 0.07)))
    painter.setBrush(QColor("#FFFFFF"))
    painter.drawEllipse(QRectF(x + w * 0.18, y + h * 0.06, w * 0.64, h * 0.64))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(RECORD_RED))
    painter.drawEllipse(QRectF(x + w * 0.31, y + h * 0.19, w * 0.38, h * 0.38))
    painter.setPen(_pen(color, max(1.5, w * 0.07)))
    painter.drawLine(QPointF(x + w * 0.5, y + h * 0.72), QPointF(x + w * 0.5, y + h * 0.88))
    painter.drawLine(QPointF(x + w * 0.36, y + h * 0.9), QPointF(x + w * 0.64, y + h * 0.9))


def _paint_folder_mic(painter: QPainter, rect: QRectF, color: QColor) -> None:
    w, h = rect.width(), rect.height()
    x, y = rect.x(), rect.y()
    path = QPainterPath()
    path.moveTo(x + w * 0.06, y + h * 0.84)
    path.lineTo(x + w * 0.06, y + h * 0.2)
    path.lineTo(x + w * 0.38, y + h * 0.2)
    path.lineTo(x + w * 0.46, y + h * 0.3)
    path.lineTo(x + w * 0.94, y + h * 0.3)
    path.lineTo(x + w * 0.94, y + h * 0.84)
    path.closeSubpath()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, max(1.5, w * 0.06)))
    painter.drawPath(path)
    painter.drawLine(QPointF(x + w * 0.06, y + h * 0.42), QPointF(x + w * 0.94, y + h * 0.42))
    inner = QRectF(x + w * 0.32, y + h * 0.44, w * 0.36, h * 0.38)
    _paint_mic(painter, inner, color)


def _paint_close_circle(painter: QPainter, rect: QRectF, color: QColor) -> None:
    w, h = rect.width(), rect.height()
    x, y = rect.x(), rect.y()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, max(1.5, w * 0.07)))
    painter.drawEllipse(QRectF(x + w * 0.08, y + h * 0.08, w * 0.84, h * 0.84))
    painter.drawLine(QPointF(x + w * 0.36, y + h * 0.36), QPointF(x + w * 0.64, y + h * 0.64))
    painter.drawLine(QPointF(x + w * 0.64, y + h * 0.36), QPointF(x + w * 0.36, y + h * 0.64))


ICON_PAINTERS: dict[str, IconPainter] = {
    "mic": _paint_mic,
    "record": _paint_record,
    "folder_mic": _paint_folder_mic,
    "close": _paint_close_circle,
}


def draw_icon(name: str, color: str, size: int = ICON_SIZE) -> QIcon:
    app = QGuiApplication.instance()
    ratio = app.devicePixelRatio() if app is not None else 1.0
    pixmap = QPixmap(int(size * ratio), int(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    ICON_PAINTERS[name](painter, QRectF(0, 0, size, size), QColor(color))
    painter.end()
    return QIcon(pixmap)


class HoverIconButton(QPushButton):
    """Pill button whose icon follows the light/dark hover colours of the text."""

    def __init__(self, text: str, icon_name: str, parent: QWidget | None = None, icon_size: int = ICON_SIZE) -> None:
        super().__init__(text, parent)
        self._normal_icon = draw_icon(icon_name, BUTTON_TEXT, icon_size)
        self._hover_icon = draw_icon(icon_name, BUTTON_HOVER_TEXT, icon_size)
        self._disabled_icon = draw_icon(icon_name, DISABLED_TEXT, icon_size)
        self.setIconSize(QSize(icon_size, icon_size))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        if not self.isEnabled():
            self.setIcon(self._disabled_icon)
        elif self.underMouse() or self.isDown():
            self.setIcon(self._hover_icon)
        else:
            self.setIcon(self._normal_icon)

    def enterEvent(self, event: Any) -> None:  # pragma: no cover - mouse driven
        super().enterEvent(event)
        self.setIcon(self._hover_icon)

    def leaveEvent(self, event: Any) -> None:  # pragma: no cover - mouse driven
        super().leaveEvent(event)
        self._refresh_icon()

    def changeEvent(self, event: Any) -> None:
        super().changeEvent(event)
        if event.type() == event.Type.EnabledChange:
            self._refresh_icon()


def add_soft_shadow(widget: QWidget) -> None:
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(24)
    shadow.setOffset(0, 4)
    shadow.setColor(QColor(36, 80, 143, 45))
    widget.setGraphicsEffect(shadow)


class GradientBackground(QWidget):
    """Soft light-blue gradient with large translucent shapes, as in the Meeting Scribe mock-up."""

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual widget
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor(BACKGROUND_TOP))
        gradient.setColorAt(1.0, QColor(BACKGROUND_BOTTOM))
        painter.fillRect(rect, QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        w, h = rect.width(), rect.height()
        for left, top, width, height, alpha in (
            (0.52, -0.35, 0.75, 1.0, 40),
            (0.70, 0.05, 0.55, 0.85, 30),
            (-0.25, 0.55, 0.70, 0.80, 35),
        ):
            painter.setBrush(QColor(173, 200, 230, alpha))
            painter.drawEllipse(QRectF(w * left, h * top, w * width, h * height))
