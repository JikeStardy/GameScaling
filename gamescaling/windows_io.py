"""Visible Windows desktop capture and foreground-only input.

This module is importable on other platforms; constructing the adapter requires
Windows. Synthetic input is subject to the normal Windows integrity restrictions.
"""
from __future__ import annotations

import contextlib
import ctypes
from ctypes import wintypes
import os

from .action import KEY_VOCAB

# Fixed-width types are intentional: ctypes.wintypes has host-dependent sizes
# when this module is imported on a non-Windows machine.
DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
WORD = ctypes.c_uint16
ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", LONG), ("dy", LONG), ("mouseData", DWORD),
                ("dwFlags", DWORD), ("time", DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", WORD), ("wScan", WORD), ("dwFlags", DWORD),
                ("time", DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", DWORD), ("wParamL", WORD), ("wParamH", WORD)]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", DWORD), ("data", _INPUT_UNION)]


class POINT(ctypes.Structure):
    _fields_ = [("x", LONG), ("y", LONG)]


class RECT(ctypes.Structure):
    _fields_ = [("left", LONG), ("top", LONG), ("right", LONG), ("bottom", LONG)]


# Set 1 scan codes, with the extended flag for E0-prefixed keys.
_SCANS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (
    0x1e, 0x30, 0x2e, 0x20, 0x12, 0x21, 0x22, 0x23, 0x17, 0x24,
    0x25, 0x26, 0x32, 0x31, 0x18, 0x19, 0x10, 0x13, 0x1f, 0x14,
    0x16, 0x2f, 0x11, 0x2d, 0x15, 0x2c)))
_SCANS.update(dict(zip(("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"), range(2, 12))))
_SCANS.update(dict(zip(("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"), range(0x3b, 0x45))))
_SCANS.update({"Eleven": 0x57, "Shift": 0x2a, "Ctrl": 0x1d, "Alt": 0x38,
               "Tab": 0x0f, "Caps": 0x3a, "Esc": 0x01, "Space": 0x39,
               "Enter": 0x1c, "Back": 0x0e, "Equal": 0x0d, "Minus": 0x0c,
               "Period": 0x34, "Slash": 0x35, "Quote": 0x28,
               "Insert": 0x52, "Delete": 0x53, "Home": 0x47, "End": 0x4f,
               "Up": 0x48, "Down": 0x50, "Left": 0x4b, "Right": 0x4d})
_EXTENDED = frozenset(("Insert", "Delete", "Home", "End", "Up", "Down", "Left", "Right"))
# Documented Win32 virtual keys. OEM punctuation follows the US ANSI layout.
_VIRTUAL_KEYS = {c: ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
_VIRTUAL_KEYS.update(dict(zip(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"), range(0x30, 0x3a))))
_VIRTUAL_KEYS.update(dict(zip(("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven"), range(0x70, 0x7b))))
_VIRTUAL_KEYS.update({"Shift": 0xa0, "Ctrl": 0xa2, "Alt": 0xa4, "Tab": 0x09,
                      "Caps": 0x14, "Esc": 0x1b, "Space": 0x20, "Enter": 0x0d,
                      "Back": 0x08, "Delete": 0x2e, "Insert": 0x2d,
                      "Home": 0x24, "End": 0x23, "Pause": 0x13,
                      "Up": 0x26, "Down": 0x28, "Left": 0x25, "Right": 0x27,
                      "Equal": 0xbb, "Minus": 0xbd, "Period": 0xbe,
                      "Slash": 0xbf, "Quote": 0xde})
_MOUSE_BUTTONS = {"LB": (0x0002, 0x0004), "RB": (0x0008, 0x0010), "MB": (0x0020, 0x0040)}


def _keyboard_input(token: str, up: bool, mode: str) -> INPUT:
    """Build one keyboard event; F12 is reserved for the stop hotkey."""
    if mode not in ("scan-code", "virtual-key"):
        raise ValueError(f"Unsupported keyboard mode: {mode!r}")
    if token not in _VIRTUAL_KEYS:
        raise ValueError(f"Unsupported or reserved keyboard token: {token!r}")
    flags = (0x0001 if token in _EXTENDED else 0) | (0x0002 if up else 0)
    if mode == "virtual-key" or token == "Pause":
        return INPUT(type=1, ki=KEYBDINPUT(_VIRTUAL_KEYS[token], 0, flags, 0, 0))
    return INPUT(type=1, ki=KEYBDINPUT(0, _SCANS[token], flags | 0x0008, 0, 0))


def _api():
    if os.name != "nt":
        raise RuntimeError("WindowsDesktop requires Windows")
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    hwnd, boolean = wintypes.HWND, wintypes.BOOL
    signatures = {
        "EnumWindows": ([ctypes.WINFUNCTYPE(boolean, hwnd, wintypes.LPARAM), wintypes.LPARAM], boolean),
        "GetWindowTextLengthW": ([hwnd], ctypes.c_int),
        "GetWindowTextW": ([hwnd, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        "IsWindow": ([hwnd], boolean), "IsWindowVisible": ([hwnd], boolean),
        "IsIconic": ([hwnd], boolean), "GetForegroundWindow": ([], hwnd),
        "GetWindowThreadProcessId": ([hwnd, ctypes.POINTER(DWORD)], DWORD),
        "GetClientRect": ([hwnd, ctypes.POINTER(RECT)], boolean),
        "ClientToScreen": ([hwnd, ctypes.POINTER(POINT)], boolean),
        "GetCursorPos": ([ctypes.POINTER(POINT)], boolean),
        "GetSystemMetrics": ([ctypes.c_int], ctypes.c_int),
        "WindowFromPoint": ([POINT], hwnd), "GetAncestor": ([hwnd, ctypes.c_uint], hwnd),
        "SendInput": ([ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int], ctypes.c_uint),
        "GetAsyncKeyState": ([ctypes.c_int], ctypes.c_short),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(user, name)
        fn.argtypes, fn.restype = args, result
    if hasattr(user, "SetThreadDpiAwarenessContext"):
        user.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [DWORD, boolean, DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, DWORD, wintypes.LPWSTR, ctypes.POINTER(DWORD)]
    kernel.QueryFullProcessImageNameW.restype = boolean
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = boolean
    return user, kernel


def _process_name(kernel, pid):
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        size = DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None
        return os.path.basename(buf.value)
    finally:
        kernel.CloseHandle(handle)


class WindowsDesktop:
    """Bind one visible window; capture its client and send foreground input."""

    def __init__(self, window_title: str, *, process_name: str | None = None,
                 keyboard_mode: str = "scan-code"):
        if not isinstance(window_title, str) or not window_title.strip():
            raise ValueError("window_title must be a non-empty literal substring")
        if process_name is not None and (not isinstance(process_name, str) or not process_name.strip() or any(c in process_name for c in "\\/")):
            raise ValueError("process_name must be an executable basename")
        if keyboard_mode not in ("scan-code", "virtual-key"):
            raise ValueError("keyboard_mode must be scan-code or virtual-key")
        self.keyboard_mode = keyboard_mode
        self._user, self._kernel = _api()
        matches = self._candidates(window_title, process_name)
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one visible window matching {window_title!r}; found {len(matches)}: "
                               + ", ".join(f"{title!r} (pid {pid}, {proc or 'unknown process'})" for _, title, pid, proc in matches))
        self._hwnd, self.title, self._pid, self.process_name = matches[0]
        self._rect()  # Reject invalid geometry at binding time.

    def _candidates(self, title, process_name):
        matches = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def visit(hwnd, _):
            if not self._user.IsWindowVisible(hwnd) or self._user.IsIconic(hwnd):
                return True
            length = self._user.GetWindowTextLengthW(hwnd)
            if not length:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            self._user.GetWindowTextW(hwnd, buf, len(buf))
            if title.casefold() not in buf.value.casefold():
                return True
            pid = DWORD()
            self._user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            proc = _process_name(self._kernel, pid.value)
            if process_name is None or (proc and proc.casefold() == process_name.casefold()):
                matches.append((hwnd, buf.value, pid.value, proc))
            return True

        if not self._user.EnumWindows(visit, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return matches

    @contextlib.contextmanager
    def _dpi(self):
        # Per-monitor-aware coordinates, without changing the process default.
        setter = getattr(self._user, "SetThreadDpiAwarenessContext", None)
        if setter is None:
            raise RuntimeError("Per-monitor DPI awareness requires Windows 10 version 1607 or later")
        previous = setter(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
        if not previous:
            previous = setter(ctypes.c_void_p(-3))  # PER_MONITOR_AWARE
        if not previous:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            setter(previous)

    def _valid(self):
        pid = DWORD()
        if (not self._user.IsWindow(self._hwnd) or
                not self._user.GetWindowThreadProcessId(self._hwnd, ctypes.byref(pid)) or pid.value != self._pid or
                not self._user.IsWindowVisible(self._hwnd) or self._user.IsIconic(self._hwnd)):
            raise RuntimeError("Bound window is closed, replaced, hidden, or minimized")

    def is_foreground(self) -> bool:
        try:
            self._valid()
        except RuntimeError:
            return False
        return self._user.GetForegroundWindow() == self._hwnd

    def _foreground(self):
        if not self.is_foreground():
            raise RuntimeError("Bound window must be visible and foreground")

    def _rect(self):
        self._valid()
        with self._dpi():
            rect = RECT()
            if not self._user.GetClientRect(self._hwnd, ctypes.byref(rect)):
                raise ctypes.WinError(ctypes.get_last_error())
            pt = POINT(rect.left, rect.top)
            if not self._user.ClientToScreen(self._hwnd, ctypes.byref(pt)):
                raise ctypes.WinError(ctypes.get_last_error())
            width, height = rect.right - rect.left, rect.bottom - rect.top
            x, y = pt.x, pt.y
            vx, vy = self._user.GetSystemMetrics(76), self._user.GetSystemMetrics(77)
            vw, vh = self._user.GetSystemMetrics(78), self._user.GetSystemMetrics(79)
            if width <= 0 or height <= 0 or x < vx or y < vy or x + width > vx + vw or y + height > vy + vh:
                raise RuntimeError("Window client area is empty or extends outside the virtual desktop")
            return x, y, x + width, y + height

    def dimensions(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self._rect()
        return x2 - x1, y2 - y1

    def capture(self):
        from PIL import ImageGrab
        self._foreground()
        with self._dpi():
            rect = self._rect()
            x1, y1, x2, y2 = rect
            # Detect common overlays; no screen capture API can prove every pixel is unobscured.
            for x, y in ((x1, y1), (x2 - 1, y1), (x1, y2 - 1), (x2 - 1, y2 - 1), ((x1 + x2) // 2, (y1 + y2) // 2)):
                top = self._user.WindowFromPoint(POINT(x, y))
                if not top or self._user.GetAncestor(top, 2) != self._hwnd:  # GA_ROOT
                    raise RuntimeError("Bound window client area is obscured")
            image = ImageGrab.grab(bbox=rect, all_screens=True)
            self._foreground()
            if self._rect() != rect or image.size != (x2 - x1, y2 - y1):
                raise RuntimeError("Window moved or capture dimensions changed during capture")
            image = image.convert("RGB")
            if image.getbbox() is None:
                raise RuntimeError("Captured window is entirely black or unavailable")
            return image

    def _mouse_target(self):
        """Require pointer actions to land on the bound client area."""
        with self._dpi():
            x1, y1, x2, y2 = self._rect()
            pt = POINT()
            if not self._user.GetCursorPos(ctypes.byref(pt)):
                raise ctypes.WinError(ctypes.get_last_error())
            top = self._user.WindowFromPoint(pt)
            if not (x1 <= pt.x < x2 and y1 <= pt.y < y2 and top and
                    self._user.GetAncestor(top, 2) == self._hwnd):
                raise RuntimeError("Mouse cursor must be over the bound window client area")

    def _send(self, item):
        if self._user.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT)) != 1:
            raise OSError(ctypes.get_last_error(), "SendInput failed (possibly blocked by Windows integrity restrictions)")

    def _key(self, token, up):
        if token not in KEY_VOCAB or token == "Twelve":
            raise ValueError(f"Unsupported or reserved input token: {token!r}")
        if not up:
            self._foreground()
        if token in _MOUSE_BUTTONS:
            if not up:
                self._mouse_target()
            flags = _MOUSE_BUTTONS[token][int(up)]
            item = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, flags, 0, 0))
        else:
            # Pause uses its virtual key in both modes because it has an E1 sequence.
            item = _keyboard_input(token, up, self.keyboard_mode)
        self._send(item)

    def key_down(self, token: str):
        self._key(token, False)

    def key_up(self, token: str):
        # Always allow release, including after focus is lost or the window closes.
        self._key(token, True)

    def mouse_move(self, dx: int, dy: int):
        if type(dx) is not int or type(dy) is not int or not all(-2147483648 <= v <= 2147483647 for v in (dx, dy)):
            raise ValueError("Mouse movement must use signed 32-bit integers")
        self._foreground()
        self._send(INPUT(type=0, mi=MOUSEINPUT(dx, dy, 0, 0x0001, 0, 0)))

    def scroll(self, notches: int):
        if type(notches) is not int or not -17895697 <= notches <= 17895697:
            raise ValueError("Scroll amount must be a signed integer within range")
        self._foreground()
        self._mouse_target()
        self._send(INPUT(type=0, mi=MOUSEINPUT(0, 0, (notches * 120) & 0xffffffff, 0x0800, 0, 0)))

    def stop_pressed(self) -> bool:
        return bool(self._user.GetAsyncKeyState(0x7b) & 0x8000)  # F12, including via Moonlight
