import pygame
import sys
import math
import numpy as np
from environment import SnakeEnv, GRID_SIZE
from trainer import (
    TrainingSession, EvalSession,
    create_agents, save_q_matrices, load_q_matrices, load_q_matrices_for_transfer,
)

# ─── Palette ─────────────────────────────────────────────────────────────────
BG        = (10,  10,  22)
PANEL_BG  = (16,  18,  36)
GRIDLINE  = (26,  28,  50)
FOOD_C    = (255,  55,  60)
TEXT_C    = (210, 255, 210)
TRAIN_ACC = (0,  255, 100)   # green  — training phase
EVAL_ACC  = (0,  210, 255)   # cyan   — evaluation phase
DIM       = (90, 150, 100)
DIM_EVAL  = (75, 135, 155)
INP_BG    = (16,  30,  16)
INP_ACT   = (0,   65,  24)
INP_BDR   = (0,  195,  78)
BTN_BG    = (0,   50,  18)
BTN_HOV   = (0,  105,  36)
BTN_TXT   = (0,  255, 100)
BTN2_BG   = (0,   18,  50)
BTN2_HOV  = (0,   50, 105)
BTN2_TXT  = (0,  210, 255)
ERR_C     = (255,  75,  75)
OK_C      = (0,  220, 100)
SEP_C     = (35,  55,  35)
DONE_BG   = (5,   18,   5)
OBST_C    = (160,  55,  55)
OBST_OUT  = (220,  90,  90)
BTN3_BG   = (40,  20,   0)
BTN3_HOV  = (90,  50,   0)
BTN3_TXT  = (255, 165,   0)

AGENT_HEAD  = [(0, 255, 100), (0, 210, 255), (255, 185,   0)]
AGENT_BODY  = [(0, 165,  65), (0, 135, 205), (205, 125,   0)]
AGENT_DIM   = [(0,  75,  30), (0,  80, 125), (125,  75,   0)]
AGENT_NAMES = ["Q-Learning", "SARSA", "Dyna-Q"]

DEFAULT_W, DEFAULT_H = 1400, 820

pygame.init()

# ─── Font cache ───────────────────────────────────────────────────────────────
_FONT_CACHE: dict = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (max(8, size), bold)
    if key not in _FONT_CACHE:
        for name in ("Courier New", "Consolas", "Lucida Console", "monospace"):
            try:
                _FONT_CACHE[key] = pygame.font.SysFont(name, key[0], bold=bold)
                break
            except Exception:
                pass
        else:
            _FONT_CACHE[key] = pygame.font.Font(None, key[0])
    return _FONT_CACHE[key]


def blit(surf, text, size, color,
         center=None, topleft=None, midleft=None, bold=False) -> pygame.Rect:
    s = font(size, bold).render(str(text), True, color)
    r = s.get_rect()
    if center:  r.center  = center
    if topleft: r.topleft = topleft
    if midleft: r.midleft = midleft
    surf.blit(s, r)
    return r


# ─── Responsive layout ────────────────────────────────────────────────────────
class Layout:
    MARGIN  = 14
    HDR_H   = 68
    INFO_H  = 132
    SB_FRAC = 0.17

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        m    = self.MARGIN
        sb_w = max(200, int(w * self.SB_FRAC))

        games_w = w - sb_w - m * 4
        panel_w = games_w // 3
        cell    = max(20, panel_w // GRID_SIZE)
        grid_px = cell * GRID_SIZE

        self.cell    = cell
        self.grid_px = grid_px
        self.panel_w = grid_px
        self.info_h  = self.INFO_H
        self.panel_h = grid_px + self.info_h
        self.hdr_h   = self.HDR_H
        self.sb_w    = max(200, w - m - (m + grid_px) * 3)
        self.margin  = m

    def panel_xy(self, idx: int) -> tuple[int, int]:
        m = self.margin
        return m + idx * (self.panel_w + m), self.hdr_h + m

    @property
    def sb_x(self) -> int:
        return self.margin + 3 * (self.panel_w + self.margin)

    def fs(self, scale: float = 1.0) -> int:
        """Font size scaled to the cell size so text fills panels naturally."""
        return max(11, int(self.cell * 0.52 * scale))


# ─── Shared snake grid renderer ───────────────────────────────────────────────
def draw_grid(surf, lo: Layout, env, ox: int, oy: int, idx: int):
    cell = lo.cell
    gp   = lo.grid_px

    pygame.draw.rect(surf, PANEL_BG, (ox, oy, gp, gp))
    pygame.draw.rect(surf, SEP_C,    (ox, oy, gp, gp), 2, border_radius=4)

    for i in range(GRID_SIZE + 1):
        pygame.draw.line(surf, GRIDLINE, (ox + i * cell, oy),    (ox + i * cell, oy + gp))
        pygame.draw.line(surf, GRIDLINE, (ox, oy + i * cell),    (ox + gp, oy + i * cell))

    for obs in env.obstacles:
        bx = ox + obs[0] * cell + 2
        by = oy + obs[1] * cell + 2
        w  = cell - 4
        pygame.draw.rect(surf, OBST_C,   (bx, by, w, w), border_radius=max(2, w // 6))
        pygame.draw.rect(surf, OBST_OUT, (bx, by, w, w), 2, border_radius=max(2, w // 6))

    for j, seg in enumerate(env.snake):
        sx = ox + seg[0] * cell + 2
        sy = oy + seg[1] * cell + 2
        w  = cell - 4
        if   j == 0: color = AGENT_HEAD[idx]
        elif j <  4: color = AGENT_BODY[idx]
        else:        color = AGENT_DIM[idx]
        pygame.draw.rect(surf, color, (sx, sy, w, w), border_radius=max(2, w // 5))
        if j == 0 and w >= 12:
            ex = sx + w // 2 - w // 5
            ey = sy + w // 3
            r  = max(2, w // 9)
            pygame.draw.circle(surf, BG, (ex, ey), r)
            pygame.draw.circle(surf, BG, (ex + w // 5 * 2, ey), r)

    t   = pygame.time.get_ticks() / 300
    rad = max(4, cell // 3) + int(2 * abs(math.sin(t)))
    fx  = ox + env.food[0] * cell + cell // 2
    fy  = oy + env.food[1] * cell + cell // 2
    pygame.draw.circle(surf, FOOD_C,          (fx, fy), rad)
    pygame.draw.circle(surf, (255, 210, 210), (fx, fy), max(2, rad - 3))


# ─── File-dialog helper (tkinter, hidden root) ────────────────────────────────
def _file_dialog(save: bool = False) -> str:
    """Return a file path from a native dialog, or '' if cancelled."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ft = [("Q-Matrix file", "*.pkl"), ("All files", "*.*")]
    if save:
        path = filedialog.asksaveasfilename(
            title="Save Q-Matrices",
            defaultextension=".pkl",
            filetypes=ft,
            initialfile="snake_q_matrices",
        )
    else:
        path = filedialog.askopenfilename(
            title="Load Q-Matrices",
            filetypes=ft,
        )
    root.destroy()
    return path or ""


# ─── Obstacle editor — click cells to place obstacles before training ────────
class ObstacleEditorScreen:
    def __init__(self, screen: pygame.Surface, grid_size: int, initial=None):
        self.screen     = screen
        self.grid_size  = grid_size
        self.obstacles  = set(initial) if initial else set()
        mid = grid_size // 2
        self.forbidden  = {(mid, mid), (mid, mid - 1), (mid, mid - 2)}
        self.done_btn   = pygame.Rect(0, 0, 0, 0)
        self.clear_btn  = pygame.Rect(0, 0, 0, 0)

    def _geometry(self):
        W, H = self.screen.get_size()
        avail = min(W - 360, H - 220)
        cell  = max(20, avail // self.grid_size)
        gp    = cell * self.grid_size
        ox    = W // 2 - gp // 2
        oy    = 130
        return ox, oy, cell, gp

    def _cell_at(self, pos):
        ox, oy, cell, gp = self._geometry()
        x, y = pos
        if ox <= x < ox + gp and oy <= y < oy + gp:
            return (int((x - ox) // cell), int((y - oy) // cell))
        return None

    def _draw(self):
        screen = self.screen
        W, H   = screen.get_size()
        screen.fill(BG)

        blit(screen, "OBSTACLE  EDITOR", max(22, H // 22), TRAIN_ACC,
             center=(W // 2, 48), bold=True)
        blit(screen, "LEFT CLICK  add / remove obstacle    GREY CELLS  reserved for the snake",
             max(11, H // 62), DIM, center=(W // 2, 84))

        ox, oy, cell, gp = self._geometry()
        pygame.draw.rect(screen, PANEL_BG, (ox, oy, gp, gp))
        pygame.draw.rect(screen, TRAIN_ACC, (ox, oy, gp, gp), 2, border_radius=4)
        for i in range(self.grid_size + 1):
            pygame.draw.line(screen, GRIDLINE, (ox + i * cell, oy), (ox + i * cell, oy + gp))
            pygame.draw.line(screen, GRIDLINE, (ox, oy + i * cell), (ox + gp, oy + i * cell))

        for fx, fy in self.forbidden:
            fbx, fby = ox + fx * cell + 2, oy + fy * cell + 2
            w = cell - 4
            pygame.draw.rect(screen, (45, 45, 65), (fbx, fby, w, w), border_radius=3)

        for gx, gy in self.obstacles:
            bx, by = ox + gx * cell + 2, oy + gy * cell + 2
            w = cell - 4
            pygame.draw.rect(screen, OBST_C, (bx, by, w, w), border_radius=max(2, w // 6))
            pygame.draw.rect(screen, OBST_OUT, (bx, by, w, w), 2, border_radius=max(2, w // 6))

        blit(screen, f"Obstacles placed: {len(self.obstacles)}", max(14, H // 50), TEXT_C,
             center=(W // 2, oy + gp + 28))

        btn_w, btn_h, gap = 200, 44, 14
        self.clear_btn = pygame.Rect(W // 2 - btn_w - gap // 2, oy + gp + 48, btn_w, btn_h)
        self.done_btn  = pygame.Rect(W // 2 + gap // 2,         oy + gp + 48, btn_w, btn_h)
        mouse = pygame.mouse.get_pos()

        pygame.draw.rect(screen, BTN3_HOV if self.clear_btn.collidepoint(mouse) else BTN3_BG,
                          self.clear_btn, border_radius=7)
        pygame.draw.rect(screen, BTN3_TXT, self.clear_btn, 2, border_radius=7)
        blit(screen, "CLEAR ALL", max(12, H // 55), BTN3_TXT, center=self.clear_btn.center, bold=True)

        pygame.draw.rect(screen, BTN_HOV if self.done_btn.collidepoint(mouse) else BTN_BG,
                          self.done_btn, border_radius=7)
        pygame.draw.rect(screen, INP_BDR, self.done_btn, 2, border_radius=7)
        blit(screen, "✓  DONE", max(12, H // 55), BTN_TXT, center=self.done_btn.center, bold=True)

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self._draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.get_surface()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.done_btn.collidepoint(event.pos):
                        return list(self.obstacles)
                    if self.clear_btn.collidepoint(event.pos):
                        self.obstacles.clear()
                        continue
                    cell = self._cell_at(event.pos)
                    if cell and cell not in self.forbidden:
                        if cell in self.obstacles:
                            self.obstacles.discard(cell)
                        else:
                            self.obstacles.add(cell)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        return list(self.obstacles)
            clock.tick(60)


# ─── Config screen ────────────────────────────────────────────────────────────
class ConfigScreen:
    LEFT_FIELDS = [
        ("Learning Rate  (α)",      "alpha",          "0.1"),
        ("Discount Factor  (γ)",    "gamma",          "0.9"),
        ("Initial Epsilon  (ε)",    "epsilon",        "1.0"),
        ("Epsilon Decay",           "epsilon_decay",  "0.995"),
        ("Dyna-Q  Planning Steps",  "planning_steps", "10"),
    ]
    RIGHT_FIELDS = [
        ("Max Training Episodes",   "max_episodes",   "3000"),
        ("Max Steps  /  Episode",   "max_steps",      "500"),
        ("Number of Obstacles",     "num_obstacles",  "0"),
    ]
    ALL_FIELDS = LEFT_FIELDS + RIGHT_FIELDS
    INT_KEYS   = {"planning_steps", "max_episodes", "max_steps", "num_obstacles"}

    TIPS = [
        "3000 episodes is enough for",
        "all three agents to converge",
        "on a 10×10 grid.  Dyna-Q",
        "usually learns the fastest.",
    ]

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.values = {k: d for _, k, d in self.ALL_FIELDS}
        self.active = None
        self.error  = ""
        self.msg    = ""   # transient status message (e.g. load success)
        self.obstacle_positions: list[tuple[int, int]] = []

    # ── drawing ───────────────────────────────────────────────────────────────
    def draw(self):
        screen = self.screen
        W, H   = screen.get_size()
        cx     = W // 2
        screen.fill(BG)

        scan = pygame.Surface((W, H), pygame.SRCALPHA)
        for y in range(0, H, 3):
            pygame.draw.line(scan, (0, 0, 0, 20), (0, y), (W, y))
        screen.blit(scan, (0, 0))

        title_fs = max(28, H // 20)
        sub_fs   = max(12, H // 56)
        title_y  = max(50, H // 13)
        blit(screen, "SNAKE  RL", title_fs, TRAIN_ACC, center=(cx, title_y), bold=True)
        blit(screen, "REINFORCEMENT  LEARNING  TRAINER", sub_fs, DIM,
             center=(cx, title_y + title_fs + 8))
        sep_y = title_y + title_fs + sub_fs + 24
        pygame.draw.line(screen, TRAIN_ACC, (cx - 250, sep_y), (cx + 250, sep_y), 1)

        col_w  = min(310, W // 4)
        lbl_fs = max(12, H // 58)
        val_fs = max(13, H // 53)
        fld_h  = max(36, H // 20)
        row_h  = fld_h + lbl_fs + 22
        gap    = max(60, W // 18)

        left_cx  = cx - col_w // 2 - gap // 2
        right_cx = cx + col_w // 2 + gap // 2
        start_y  = sep_y + 32

        field_rects: dict[str, pygame.Rect] = {}

        blit(screen, "─  Hyperparameters  ─", lbl_fs, DIM, center=(left_cx, start_y))
        for i, (label, key, _) in enumerate(self.LEFT_FIELDS):
            y    = start_y + 22 + i * row_h
            blit(screen, label, lbl_fs, DIM, center=(left_cx, y))
            rect = pygame.Rect(left_cx - col_w // 2, y + lbl_fs + 6, col_w, fld_h)
            field_rects[key] = rect
            self._draw_field(screen, rect, key, val_fs)

        blit(screen, "─  Training  Config  ─", lbl_fs, DIM, center=(right_cx, start_y))
        for i, (label, key, _) in enumerate(self.RIGHT_FIELDS):
            y    = start_y + 22 + i * row_h
            blit(screen, label, lbl_fs, DIM, center=(right_cx, y))
            rect = pygame.Rect(right_cx - col_w // 2, y + lbl_fs + 6, col_w, fld_h)
            field_rects[key] = rect
            self._draw_field(screen, rect, key, val_fs)

        tip_y  = start_y + 22 + len(self.RIGHT_FIELDS) * row_h + 20
        tip_fs = max(10, lbl_fs - 2)
        for j, line in enumerate(self.TIPS):
            blit(screen, line, tip_fs, (55, 100, 65),
                 center=(right_cx, tip_y + j * (tip_fs + 6)))

        # ── error / status + buttons ──────────────────────────────────────────
        bottom_y = start_y + 22 + max(len(self.LEFT_FIELDS), len(self.RIGHT_FIELDS)) * row_h + 30
        if self.error:
            blit(screen, self.error, lbl_fs, ERR_C, center=(cx, bottom_y))
            bottom_y += lbl_fs + 8
        elif self.msg:
            blit(screen, self.msg, lbl_fs, OK_C, center=(cx, bottom_y))
            bottom_y += lbl_fs + 8

        btn_h  = max(44, H // 17)
        gap_b  = 14
        btn_w  = 200
        total  = btn_w * 4 + gap_b * 3
        btn1   = pygame.Rect(cx - total // 2,                       bottom_y + 8, btn_w, btn_h)
        btn2   = pygame.Rect(cx - total // 2 + btn_w + gap_b,       bottom_y + 8, btn_w, btn_h)
        btn3   = pygame.Rect(cx - total // 2 + (btn_w + gap_b) * 2, bottom_y + 8, btn_w, btn_h)
        btn4   = pygame.Rect(cx - total // 2 + (btn_w + gap_b) * 3, bottom_y + 8, btn_w, btn_h)
        mouse  = pygame.mouse.get_pos()

        pygame.draw.rect(screen, BTN_HOV if btn1.collidepoint(mouse) else BTN_BG,
                         btn1, border_radius=7)
        pygame.draw.rect(screen, INP_BDR, btn1, 2, border_radius=7)
        blit(screen, "▶  START TRAINING", max(12, H // 54), BTN_TXT,
             center=btn1.center, bold=True)

        pygame.draw.rect(screen, BTN2_HOV if btn2.collidepoint(mouse) else BTN2_BG,
                         btn2, border_radius=7)
        pygame.draw.rect(screen, EVAL_ACC, btn2, 2, border_radius=7)
        blit(screen, "⬆  LOAD  (EVAL)", max(12, H // 54), BTN2_TXT,
             center=btn2.center, bold=True)

        pygame.draw.rect(screen, BTN3_HOV if btn3.collidepoint(mouse) else BTN3_BG,
                         btn3, border_radius=7)
        pygame.draw.rect(screen, BTN3_TXT, btn3, 2, border_radius=7)
        blit(screen, "⟳  TRANSFER LEARN", max(12, H // 54), BTN3_TXT,
             center=btn3.center, bold=True)

        pygame.draw.rect(screen, BTN3_HOV if btn4.collidepoint(mouse) else BTN3_BG,
                         btn4, border_radius=7)
        pygame.draw.rect(screen, OBST_OUT, btn4, 2, border_radius=7)
        blit(screen, f"⊞  OBSTACLES ({len(self.obstacle_positions)})", max(12, H // 54), OBST_OUT,
             center=btn4.center, bold=True)

        pygame.display.flip()
        return btn1, btn2, btn3, btn4, field_rects

    def _draw_field(self, screen, rect, key, val_fs):
        active = self.active == key
        pygame.draw.rect(screen, INP_ACT if active else INP_BG, rect, border_radius=5)
        pygame.draw.rect(screen, INP_BDR if active else SEP_C,  rect, 2, border_radius=5)
        vs = font(val_fs).render(self.values[key], True, TEXT_C)
        screen.blit(vs, vs.get_rect(midleft=(rect.x + 10, rect.centery)))
        if active and pygame.time.get_ticks() % 900 < 450:
            cx_ = rect.x + 10 + vs.get_width() + 3
            pygame.draw.line(screen, TRAIN_ACC, (cx_, rect.y + 6), (cx_, rect.bottom - 6), 2)

    # ── parsing ───────────────────────────────────────────────────────────────
    def _parse(self):
        try:
            cfg = {}
            for _, k, _ in self.ALL_FIELDS:
                cfg[k] = int(float(self.values[k])) if k in self.INT_KEYS \
                         else float(self.values[k])
            cfg["obstacle_positions"] = list(self.obstacle_positions) or None
            return cfg
        except ValueError:
            self.error = "Invalid value — numbers only"
            return None

    # ── main loop ─────────────────────────────────────────────────────────────
    # Returns ("train", cfg) or ("eval", agents)
    def run(self):
        clock    = pygame.time.Clock()
        all_keys = [k for _, k, _ in self.ALL_FIELDS]
        while True:
            btn1, btn2, btn3, btn4, field_rects = self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.get_surface()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    hit = False
                    for key, rect in field_rects.items():
                        if rect.collidepoint(event.pos):
                            self.active = key; hit = True; break
                    if not hit:
                        self.active = None
                    if btn1.collidepoint(event.pos):
                        cfg = self._parse()
                        if cfg:
                            return "train", cfg
                    if btn2.collidepoint(event.pos):
                        self._do_load()
                    if btn3.collidepoint(event.pos):
                        self._do_transfer_load()
                    if btn4.collidepoint(event.pos):
                        self._do_edit_obstacles()
                elif event.type == pygame.KEYDOWN:
                    if self.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.values[self.active] = self.values[self.active][:-1]
                        elif event.key in (pygame.K_RETURN, pygame.K_TAB):
                            idx = all_keys.index(self.active)
                            self.active = all_keys[(idx + 1) % len(all_keys)]
                        elif event.unicode in "0123456789.":
                            self.values[self.active] += event.unicode
                    elif event.key == pygame.K_RETURN:
                        cfg = self._parse()
                        if cfg:
                            return "train", cfg
            clock.tick(60)

    def _do_edit_obstacles(self):
        editor = ObstacleEditorScreen(self.screen, GRID_SIZE, self.obstacle_positions)
        self.obstacle_positions = editor.run()
        self.screen = pygame.display.get_surface()
        self.values["num_obstacles"] = str(len(self.obstacle_positions))

    def _do_load(self):
        path = _file_dialog(save=False)
        if not path:
            return
        try:
            agents = load_q_matrices(path)
            self._loaded_agents = agents
            self.msg   = "Q-Matrices loaded — evaluating now…"
            self.error = ""
            raise _LoadedSignal(agents)
        except _LoadedSignal:
            raise
        except Exception as exc:
            self.error = f"Load failed: {exc}"
            self.msg   = ""

    def _do_transfer_load(self):
        path = _file_dialog(save=False)
        if not path:
            return
        cfg = self._parse()
        if not cfg:
            return
        try:
            agents = load_q_matrices_for_transfer(path, cfg)
            self.msg   = "Q-Matrices loaded — resuming training…"
            self.error = ""
            raise _TransferSignal(agents, cfg)
        except _TransferSignal:
            raise
        except Exception as exc:
            self.error = f"Load failed: {exc}"
            self.msg   = ""


class _LoadedSignal(Exception):
    def __init__(self, agents):
        self.agents = agents


class _TransferSignal(Exception):
    def __init__(self, agents, cfg):
        self.agents = agents
        self.cfg    = cfg


# ─── Training screen ──────────────────────────────────────────────────────────
class TrainingScreen:
    SPEED_OPTIONS = [1, 2, 4, 8, 16, 32]

    def __init__(self, screen, session: TrainingSession):
        self.screen  = screen
        self.session = session

        self.paused    = False
        self.speed_idx = 2
        self._save_msg = ""   # feedback shown in overlay after save

        self._layout    = None
        self._last_size = (0, 0)

    def _get_layout(self) -> Layout:
        size = self.screen.get_size()
        if size != self._last_size:
            self._layout    = Layout(*size)
            self._last_size = size
        return self._layout

    # ── header with progress bar ──────────────────────────────────────────────
    def _draw_header(self, lo: Layout):
        s = self.session
        W = self.screen.get_width()
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, lo.hdr_h))
        pygame.draw.line(self.screen, TRAIN_ACC, (0, lo.hdr_h - 1), (W, lo.hdr_h - 1))

        done_ep  = min(s.episodes)
        progress = min(1.0, done_ep / max(1, s.max_episodes))
        bar_w    = int(W * 0.36)
        bar_x    = W // 2 - bar_w // 2
        bar_h    = max(7, lo.fs(0.48))
        bar_y    = lo.hdr_h - bar_h - 10

        pygame.draw.rect(self.screen, (28, 48, 28),
                         (bar_x, bar_y, bar_w, bar_h), border_radius=bar_h)
        if progress > 0:
            pygame.draw.rect(self.screen, TRAIN_ACC,
                             (bar_x, bar_y, int(bar_w * progress), bar_h),
                             border_radius=bar_h)

        blit(self.screen, "TRAINING  DASHBOARD",
             lo.fs(1.05), TRAIN_ACC,
             center=(W // 2, lo.hdr_h // 2 - bar_h - 6), bold=True)
        blit(self.screen,
             f"Episode  {done_ep:,} / {s.max_episodes:,}   ({int(progress * 100)} %)",
             lo.fs(0.78), DIM,
             center=(W // 2, bar_y - lo.fs(0.78) // 2 - 4))

        if self.paused:
            blit(self.screen, "[ PAUSED ]", lo.fs(0.95), FOOD_C,
                 midleft=(lo.margin, lo.hdr_h // 2))

    # ── stats strip below each grid ───────────────────────────────────────────
    def _draw_stats(self, lo: Layout, idx, ox, oy):
        s     = self.session
        iy    = oy + lo.grid_px
        gp    = lo.panel_w
        ih    = lo.info_h
        color = AGENT_HEAD[idx]
        pad   = lo.margin
        fs    = lo.fs()

        pygame.draw.rect(self.screen, (11, 13, 27), (ox, iy, gp, ih))
        pygame.draw.line(self.screen, color, (ox, iy), (ox + gp, iy), 2)

        r0 = blit(self.screen, AGENT_NAMES[idx], lo.fs(1.3), color,
                  topleft=(ox + pad, iy + pad), bold=True)
        r1 = blit(self.screen,
                  f"Episode  {s.episodes[idx]:>5}  /  {s.max_episodes}",
                  fs, TEXT_C, topleft=(ox + pad, r0.bottom + 8))
        r2 = blit(self.screen,
                  f"Score  {s.scores[idx]:>3}      Best  {s.records[idx]:>3}",
                  fs, TEXT_C, topleft=(ox + pad, r1.bottom + 5))
        blit(self.screen,
             f"Epsilon  {s.agents[idx].epsilon:.5f}",
             fs, DIM, topleft=(ox + pad, r2.bottom + 5))

    # ── sidebar ───────────────────────────────────────────────────────────────
    def _draw_sidebar(self, lo: Layout):
        s   = self.session
        sx  = lo.sb_x
        sy  = lo.hdr_h + lo.margin
        sw  = lo.sb_w - lo.margin
        sh  = lo.panel_h
        pad = lo.margin

        pygame.draw.rect(self.screen, PANEL_BG, (sx, sy, sw, sh), border_radius=6)
        pygame.draw.rect(self.screen, SEP_C,    (sx, sy, sw, sh), 1, border_radius=6)

        blit(self.screen, "LEADERBOARD", lo.fs(1.05), TRAIN_ACC,
             center=(sx + sw // 2, sy + pad + lo.fs(1.05) // 2), bold=True)
        pygame.draw.line(self.screen, TRAIN_ACC,
                         (sx + pad, sy + pad * 3), (sx + sw - pad, sy + pad * 3))

        # ── bottom section: build top-to-bottom position bottom-up so that
        #    the leaderboard slot height is derived from the real available space ──
        hints    = [("SPACE", "pause"), ("+ / -", "speed"), ("ESC", "menu")]
        hint_fs  = lo.fs(0.72)
        hint_gap = 4
        hints_block_h = len(hints) * hint_fs + (len(hints) - 1) * hint_gap
        hints_top = sy + sh - pad - hints_block_h

        for i, (k, desc) in enumerate(hints):
            blit(self.screen, f"{k:<7} {desc}", hint_fs, DIM,
                 center=(sx + sw // 2, hints_top + i * (hint_fs + hint_gap) + hint_fs // 2))

        spd_lbl_fs = lo.fs(0.82)
        spd_val_fs = lo.fs(0.95)
        spd_block_h = pad + spd_lbl_fs + pad + spd_val_fs
        sep_y = hints_top - pad - spd_block_h

        pygame.draw.line(self.screen, SEP_C, (sx + pad, sep_y), (sx + sw - pad, sep_y))
        spd = self.SPEED_OPTIONS[self.speed_idx]
        blit(self.screen, "SPEED", spd_lbl_fs, DIM,
             center=(sx + sw // 2, sep_y + pad + spd_lbl_fs // 2))
        blit(self.screen, f"[ - ]   {spd}x   [ + ]", spd_val_fs, TEXT_C,
             center=(sx + sw // 2, sep_y + pad * 2 + spd_lbl_fs + spd_val_fs // 2),
             bold=True)

        # ── leaderboard slots: divide the space actually left above sep_y ──
        sorted_i  = sorted(range(3), key=lambda i: -s.records[i])
        medals    = ["# 1", "# 2", "# 3"]
        slots_top = sy + pad * 4        # one pad below the separator line
        slot_h    = (sep_y - slots_top - pad) // 3

        for rank, i in enumerate(sorted_i):
            ry = slots_top + rank * slot_h
            c  = AGENT_HEAD[i]
            r0 = blit(self.screen, medals[rank],      lo.fs(0.9),  c, topleft=(sx + pad, ry), bold=True)
            r1 = blit(self.screen, AGENT_NAMES[i],    lo.fs(0.82), c, topleft=(sx + pad, r0.bottom + 3))
            r2 = blit(self.screen, str(s.records[i]), lo.fs(2.0),  c,
                      topleft=(sx + pad, r1.bottom + 3), bold=True)
            blit(self.screen, f"ep {s.episodes[i]:,}", lo.fs(0.78), DIM,
                 topleft=(sx + pad, r2.bottom + 2))

    # ── training-complete overlay ─────────────────────────────────────────────
    def _draw_done_overlay(self, lo: Layout):
        s    = self.session
        W, H = self.screen.get_size()
        ov   = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 165))
        self.screen.blit(ov, (0, 0))

        box_w = min(560, W - 60)
        box_h = 360
        bx    = W // 2 - box_w // 2
        by    = H // 2 - box_h // 2
        pygame.draw.rect(self.screen, DONE_BG,   (bx, by, box_w, box_h), border_radius=10)
        pygame.draw.rect(self.screen, TRAIN_ACC, (bx, by, box_w, box_h), 2, border_radius=10)

        cy = by + 36
        blit(self.screen, "TRAINING  COMPLETE !", lo.fs(1.65), TRAIN_ACC,
             center=(W // 2, cy), bold=True)
        cy += lo.fs(1.65) + 14
        pygame.draw.line(self.screen, TRAIN_ACC, (bx + 30, cy), (bx + box_w - 30, cy))
        cy += 16

        for i, name in enumerate(AGENT_NAMES):
            blit(self.screen,
                 f"{name:<12}  {s.episodes[i]:,} eps   |   Best: {s.records[i]}",
                 lo.fs(0.92), AGENT_HEAD[i], center=(W // 2, cy))
            cy += lo.fs(0.92) + 10

        cy += 12
        pygame.draw.line(self.screen, SEP_C, (bx + 30, cy), (bx + box_w - 30, cy))
        cy += 16
        blit(self.screen, "SPACE  →  Evaluate agents", lo.fs(1.02), TEXT_C,
             center=(W // 2, cy), bold=True)
        cy += lo.fs(1.02) + 10
        blit(self.screen, "S      →  Save Q-Matrices", lo.fs(1.02), EVAL_ACC,
             center=(W // 2, cy))
        cy += lo.fs(1.02) + 10
        blit(self.screen, "ESC    →  Back to menu",    lo.fs(1.02), DIM,
             center=(W // 2, cy))

        # Feedback line after save attempt
        if self._save_msg:
            cy += lo.fs(1.02) + 14
            color = OK_C if "Saved" in self._save_msg else ERR_C
            blit(self.screen, self._save_msg, lo.fs(0.88), color, center=(W // 2, cy))

    # ── full frame ────────────────────────────────────────────────────────────
    def draw(self):
        lo = self._get_layout()
        self.screen.fill(BG)
        self._draw_header(lo)
        for i in range(3):
            ox, oy = lo.panel_xy(i)
            draw_grid(self.screen, lo, self.session.envs[i], ox, oy, i)
            self._draw_stats(lo, i, ox, oy)
        self._draw_sidebar(lo)

        if self.session.training_done:
            self._draw_done_overlay(lo)
        elif self.paused:
            ov = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 120))
            self.screen.blit(ov, (0, 0))
            blit(self.screen, "PAUSED", lo.fs(3.5), TRAIN_ACC,
                 center=(self.screen.get_width() // 2,
                          self.screen.get_height() // 2), bold=True)
            blit(self.screen, "press SPACE to continue", lo.fs(0.9), DIM,
                 center=(self.screen.get_width() // 2,
                          self.screen.get_height() // 2 + lo.fs(3.5) + 10))
        pygame.display.flip()

    # ── save helper ───────────────────────────────────────────────────────────
    def _do_save(self):
        path = _file_dialog(save=True)
        if not path:
            return
        try:
            save_q_matrices(self.session.agents, path)
            self._save_msg = f"Saved  →  {path}"
        except Exception as exc:
            self._save_msg = f"Save failed: {exc}"

    # ── main loop — returns trained agents list or None ───────────────────────
    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.get_surface()
                elif event.type == pygame.KEYDOWN:
                    if self.session.training_done:
                        if event.key == pygame.K_SPACE:
                            return self.session.agents
                        elif event.key == pygame.K_s:
                            self._do_save()
                        elif event.key == pygame.K_ESCAPE:
                            return None
                    else:
                        if event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_ESCAPE:
                            return None
                        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                            self.speed_idx = min(self.speed_idx + 1, len(self.SPEED_OPTIONS) - 1)
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self.speed_idx = max(self.speed_idx - 1, 0)

            if not self.paused and not self.session.training_done:
                for _ in range(self.SPEED_OPTIONS[self.speed_idx]):
                    self.session.step()

            self.draw()
            clock.tick(60)


# ─── Evaluation screen ────────────────────────────────────────────────────────
class EvalScreen:
    STEP_MS = 95   # ~10.5 steps / second — watchable but not sluggish

    def __init__(self, screen, session: EvalSession):
        self.screen  = screen
        self.session = session

        self._layout    = None
        self._last_size = (0, 0)
        self.step_timer = 0

    def _get_layout(self) -> Layout:
        size = self.screen.get_size()
        if size != self._last_size:
            self._layout    = Layout(*size)
            self._last_size = size
        return self._layout

    # ── header ────────────────────────────────────────────────────────────────
    def _draw_header(self, lo: Layout):
        s = self.session
        W = self.screen.get_width()
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, lo.hdr_h))
        pygame.draw.line(self.screen, EVAL_ACC, (0, lo.hdr_h - 1), (W, lo.hdr_h - 1))
        blit(self.screen, "SNAKE  RL  —  EVALUATION  MODE",
             lo.fs(1.05), EVAL_ACC, center=(W // 2, lo.hdr_h // 2 - 8), bold=True)
        blit(self.screen,
             f"Episode  {min(s.eval_ep)}   |   Pure greedy  ( ε = 0.0 )",
             lo.fs(0.82), DIM_EVAL,
             center=(W // 2, lo.hdr_h // 2 + lo.fs(0.82) // 2 + 4))

    # ── stats strip below each eval grid ──────────────────────────────────────
    def _draw_eval_stats(self, lo: Layout, idx, ox, oy):
        s     = self.session
        iy    = oy + lo.grid_px
        gp    = lo.panel_w
        ih    = lo.info_h
        color = AGENT_HEAD[idx]
        pad   = lo.margin
        fs    = lo.fs()

        hist = s.eval_hist[idx]
        best = max(hist) if hist else 0
        avg  = sum(hist) / len(hist) if hist else 0.0

        pygame.draw.rect(self.screen, (10, 13, 30), (ox, iy, gp, ih))
        pygame.draw.line(self.screen, color, (ox, iy), (ox + gp, iy), 2)

        r0 = blit(self.screen, AGENT_NAMES[idx], lo.fs(1.3), color,
                  topleft=(ox + pad, iy + pad), bold=True)
        r1 = blit(self.screen,
                  f"Score  {s.scores[idx]:>3}      Best  {best:>3}",
                  fs, TEXT_C, topleft=(ox + pad, r0.bottom + 8))
        r2 = blit(self.screen,
                  f"Avg  {avg:>5.2f}     Episodes  {s.eval_ep[idx]}",
                  fs, TEXT_C, topleft=(ox + pad, r1.bottom + 5))

        recent = hist[-10:] if hist else []
        if recent:
            max_s  = max(max(recent), 1)
            bar_h  = lo.fs(0.55)
            bar_w  = max(6, (gp - pad * 2) // 12 - 3)
            base_y = r2.bottom + 4 + bar_h * 2
            for j, sc in enumerate(recent):
                h  = max(2, int((sc / max_s) * bar_h * 2))
                bx = ox + pad + j * (bar_w + 3)
                pygame.draw.rect(self.screen, AGENT_DIM[idx],
                                 (bx, base_y - h, bar_w, h), border_radius=2)

    # ── sidebar leaderboard ───────────────────────────────────────────────────
    def _draw_sidebar(self, lo: Layout):
        s   = self.session
        sx  = lo.sb_x
        sy  = lo.hdr_h + lo.margin
        sw  = lo.sb_w - lo.margin
        sh  = lo.panel_h
        pad = lo.margin

        pygame.draw.rect(self.screen, PANEL_BG, (sx, sy, sw, sh), border_radius=6)
        pygame.draw.rect(self.screen, (20, 40, 60), (sx, sy, sw, sh), 1, border_radius=6)

        blit(self.screen, "RESULTS", lo.fs(1.05), EVAL_ACC,
             center=(sx + sw // 2, sy + pad + lo.fs(1.05) // 2), bold=True)
        pygame.draw.line(self.screen, EVAL_ACC,
                         (sx + pad, sy + pad * 3), (sx + sw - pad, sy + pad * 3))

        def avg_score(i):
            h = s.eval_hist[i]
            return sum(h) / len(h) if h else 0.0

        sorted_i = sorted(range(3), key=avg_score, reverse=True)
        medals   = ["# 1", "# 2", "# 3"]
        slot_h   = (sh - pad * 6 - lo.fs(1.05) * 2) // 3

        for rank, i in enumerate(sorted_i):
            ry   = sy + pad * 4 + lo.fs(1.05) + rank * slot_h
            c    = AGENT_HEAD[i]
            hist = s.eval_hist[i]
            best = max(hist) if hist else 0
            avg  = avg_score(i)

            r0 = blit(self.screen, medals[rank],   lo.fs(0.9),  c, topleft=(sx + pad, ry), bold=True)
            r1 = blit(self.screen, AGENT_NAMES[i], lo.fs(0.82), c, topleft=(sx + pad, r0.bottom + 3))
            avg_lbl = f"avg  {avg:.2f}" if hist else "avg  —"
            r2 = blit(self.screen, avg_lbl, lo.fs(1.85), c,
                      topleft=(sx + pad, r1.bottom + 3), bold=True)
            blit(self.screen,
                 f"best {best:>3}   |   {s.eval_ep[i]} eps",
                 lo.fs(0.78), DIM_EVAL,
                 topleft=(sx + pad, r2.bottom + 2))

        pygame.draw.line(self.screen, SEP_C,
                         (sx + pad, sy + sh - pad * 4 - lo.fs(0.78)),
                         (sx + sw - pad, sy + sh - pad * 4 - lo.fs(0.78)))
        blit(self.screen, "ESC  →  back to menu", lo.fs(0.78), DIM_EVAL,
             center=(sx + sw // 2, sy + sh - pad * 2))

    # ── full frame ────────────────────────────────────────────────────────────
    def draw(self):
        lo = self._get_layout()
        self.screen.fill(BG)
        self._draw_header(lo)
        for i in range(3):
            ox, oy = lo.panel_xy(i)
            draw_grid(self.screen, lo, self.session.envs[i], ox, oy, i)
            self._draw_eval_stats(lo, i, ox, oy)
        self._draw_sidebar(lo)
        pygame.display.flip()

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        clock = pygame.time.Clock()
        while True:
            dt = clock.tick(60)
            self.step_timer += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.get_surface()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return

            while self.step_timer >= self.STEP_MS:
                self.session.step()
                self.step_timer -= self.STEP_MS

            self.draw()


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    screen = pygame.display.set_mode((DEFAULT_W, DEFAULT_H), pygame.RESIZABLE)
    pygame.display.set_caption("Snake RL  —  Q-Learning vs SARSA vs Dyna-Q")

    while True:
        cfg_screen = ConfigScreen(screen)
        try:
            mode, payload = cfg_screen.run()
        except _LoadedSignal as sig:
            # User loaded a Q-matrix file — skip training, go straight to eval (no obstacles)
            screen = pygame.display.get_surface()
            EvalScreen(screen, EvalSession(sig.agents)).run()
            screen = pygame.display.get_surface()
            continue
        except _TransferSignal as sig:
            # User loaded a Q-matrix for transfer learning — start training from loaded Q-tables
            screen = pygame.display.get_surface()
            cfg    = sig.cfg
            n_obs  = cfg.get("num_obstacles", 0)
            obs_pos = cfg.get("obstacle_positions")
            envs   = [SnakeEnv(num_obstacles=n_obs, obstacle_positions=obs_pos) for _ in range(3)]
            session = TrainingSession(sig.agents, envs, cfg)
            trained = TrainingScreen(screen, session).run()
            screen  = pygame.display.get_surface()
            if trained is not None:
                EvalScreen(screen, EvalSession(trained, num_obstacles=n_obs,
                                                obstacle_positions=obs_pos)).run()
                screen = pygame.display.get_surface()
            continue

        screen = pygame.display.get_surface()

        if mode == "eval":
            EvalScreen(screen, EvalSession(payload)).run()
            screen = pygame.display.get_surface()
            continue

        # mode == "train"
        cfg     = payload
        n_obs   = cfg.get("num_obstacles", 0)
        obs_pos = cfg.get("obstacle_positions")
        agents  = create_agents(cfg)
        envs    = [SnakeEnv(num_obstacles=n_obs, obstacle_positions=obs_pos) for _ in range(3)]

        session = TrainingSession(agents, envs, cfg)
        trained = TrainingScreen(screen, session).run()
        screen  = pygame.display.get_surface()

        if trained is not None:
            EvalScreen(screen, EvalSession(trained, num_obstacles=n_obs,
                                            obstacle_positions=obs_pos)).run()
            screen = pygame.display.get_surface()


if __name__ == "__main__":
    main()
