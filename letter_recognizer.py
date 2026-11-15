"""
Распознавание рукописной русской буквы обученной CNN-моделью.

Запуск:
    pip install tensorflow pillow numpy
    python letter_recognizer.py

Нужны два файла из Colab:
    russian_letters_cnn.keras  — модель
    classes.json               — список букв (лежит рядом с моделью)
"""

import json
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageTk

IMG_SIZE = 28
PREVIEW = 260      # размер превью исходной картинки
NET_VIEW = 140     # во сколько пикселей показывать картинку 28×28


# ---------- Предобработка: должна совпадать с обучением ----------

def to_square(img_l, size=IMG_SIZE, margin=0.15):
    """Обрезать пустые поля, положить букву в центр квадрата, уменьшить до 28×28."""
    bbox = img_l.getbbox()
    if bbox is None:
        return None
    img_l = img_l.crop(bbox)
    w, h = img_l.size
    side = int(max(w, h) * (1 + 2 * margin))
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img_l, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def prepare_image(path):
    """Вернуть картинку 28×28 'L': светлая буква на тёмном фоне, как в датасете."""
    img = Image.open(path)

    # Картинки из CoMNIST: буква хранится в альфа-канале
    if "A" in img.getbands():
        alpha = img.convert("RGBA").split()[-1]
        if alpha.getextrema()[0] < 255:          # прозрачность действительно используется
            return to_square(alpha)

    # Обычная фотография или скан
    img = ImageOps.exif_transpose(img).convert("L")
    img = ImageOps.autocontrast(img)
    if np.array(img).mean() > 127:               # тёмная буква на светлом фоне
        img = ImageOps.invert(img)
    img = img.point(lambda v: 0 if v < 100 else v)   # убрать серый фон бумаги
    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)
    k = max(3, (max(img.size) // 25) | 1)        # нечётный размер фильтра
    img = img.filter(ImageFilter.MaxFilter(k))   # утолщить линии
    return to_square(img)


# ---------- Окно приложения ----------

class App:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.classes = None
        self.image_path = None
        self._photo_refs = []                    # чтобы tkinter не потерял картинки

        root.title("Распознавание буквы")
        root.resizable(False, False)
        pad = {"padx": 10, "pady": 6}

        # --- Модель ---
        top = ttk.Frame(root)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="Выбрать модель…", command=self.choose_model).pack(side="left")
        self.model_label = ttk.Label(top, text="Модель не загружена", foreground="gray")
        self.model_label.pack(side="left", padx=10)

        # --- Картинка ---
        mid = ttk.Frame(root)
        mid.pack(fill="x", **pad)
        self.image_btn = ttk.Button(mid, text="Выбрать картинку…",
                                    command=self.choose_image, state="disabled")
        self.image_btn.pack(side="left")
        self.image_label = ttk.Label(mid, text="", foreground="gray")
        self.image_label.pack(side="left", padx=10)

        # --- Превью ---
        views = ttk.Frame(root)
        views.pack(**pad)

        left = ttk.Frame(views)
        left.pack(side="left", padx=10)
        ttk.Label(left, text="Исходная").pack()
        self.orig_canvas = tk.Canvas(left, width=PREVIEW, height=PREVIEW,
                                     bg="white", highlightthickness=1,
                                     highlightbackground="#ccc")
        self.orig_canvas.pack()

        right = ttk.Frame(views)
        right.pack(side="left", padx=10, anchor="n")
        ttk.Label(right, text="Что видит сеть (28×28)").pack()
        self.net_canvas = tk.Canvas(right, width=NET_VIEW, height=NET_VIEW,
                                    bg="black", highlightthickness=1,
                                    highlightbackground="#ccc")
        self.net_canvas.pack()

        # --- Результат ---
        self.answer = ttk.Label(right, text="", font=("Arial", 48, "bold"))
        self.answer.pack(pady=(14, 0))

        res = ttk.Frame(root)
        res.pack(fill="x", **pad)
        self.rows = []
        for _ in range(3):
            row = ttk.Frame(res)
            row.pack(fill="x", pady=2)
            letter = ttk.Label(row, text="", width=3, font=("Arial", 14, "bold"))
            letter.pack(side="left")
            bar = ttk.Progressbar(row, length=320, maximum=100)
            bar.pack(side="left", padx=8)
            pct = ttk.Label(row, text="", width=7)
            pct.pack(side="left")
            self.rows.append((letter, bar, pct))

        self.status = ttk.Label(root, text="Выберите файл модели .keras",
                                foreground="gray")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

    # ---------- Модель ----------

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="Файл модели",
            filetypes=[("Модель Keras", "*.keras *.h5"), ("Все файлы", "*.*")])
        if not path:
            return

        classes_path = os.path.join(os.path.dirname(path), "classes.json")
        if not os.path.exists(classes_path):
            classes_path = filedialog.askopenfilename(
                title="Не нашёл classes.json рядом с моделью — укажите его",
                filetypes=[("JSON", "*.json")])
            if not classes_path:
                return

        try:
            with open(classes_path, encoding="utf-8") as f:
                classes = json.load(f)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать classes.json:\n{e}")
            return

        self.status.config(text="Загружаю TensorFlow и модель… (первый раз до минуты)")
        self.image_btn.config(state="disabled")
        threading.Thread(target=self._load_model, args=(path, classes),
                         daemon=True).start()

    def _load_model(self, path, classes):
        """Работает в отдельном потоке, чтобы окно не зависало."""
        try:
            from tensorflow import keras
            model = keras.models.load_model(path)
            n_out = model.output_shape[-1]
            if n_out != len(classes):
                raise ValueError(f"У модели {n_out} выходов, "
                                 f"а в classes.json {len(classes)} букв")
            self.root.after(0, self._model_ready, model, classes, path)
        except Exception as e:
            self.root.after(0, self._model_failed, e)

    def _model_ready(self, model, classes, path):
        self.model, self.classes = model, classes
        self.model_label.config(text=f"{os.path.basename(path)}  ({len(classes)} букв)",
                                foreground="black")
        self.image_btn.config(state="normal")
        self.status.config(text="Модель готова. Выберите картинку с буквой.")
        if self.image_path:
            self.predict()

    def _model_failed(self, err):
        self.status.config(text="Модель не загружена")
        messagebox.showerror("Ошибка загрузки модели", str(err))

    # ---------- Картинка ----------

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Картинка с буквой",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        self.image_path = path
        self.image_label.config(text=os.path.basename(path), foreground="black")
        self.predict()

    def predict(self):
        self._photo_refs.clear()
        try:
            orig = Image.open(self.image_path)
            orig = ImageOps.exif_transpose(orig)
            sq = prepare_image(self.image_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть картинку:\n{e}")
            return

        # исходная картинка
        shown = orig.convert("RGBA")
        bg = Image.new("RGBA", shown.size, "white")
        shown = Image.alpha_composite(bg, shown).convert("RGB")
        shown.thumbnail((PREVIEW, PREVIEW))
        ph = ImageTk.PhotoImage(shown)
        self._photo_refs.append(ph)
        self.orig_canvas.delete("all")
        self.orig_canvas.create_image(PREVIEW // 2, PREVIEW // 2, image=ph)

        if sq is None:
            self.net_canvas.delete("all")
            self._clear_results()
            self.status.config(text="На картинке не нашлось буквы")
            return

        # что видит сеть: 28×28, увеличенная без сглаживания
        big = sq.resize((NET_VIEW, NET_VIEW), Image.NEAREST)
        ph2 = ImageTk.PhotoImage(big)
        self._photo_refs.append(ph2)
        self.net_canvas.delete("all")
        self.net_canvas.create_image(NET_VIEW // 2, NET_VIEW // 2, image=ph2)

        # предсказание
        x = np.array(sq, dtype="float32")[np.newaxis, ..., np.newaxis] / 255.0  # (1, 28, 28, 1)
        probs = self.model.predict(x, verbose=0)[0]
        top = np.argsort(probs)[::-1][:3]

        self.answer.config(text=self.classes[top[0]])
        for (letter, bar, pct), c in zip(self.rows, top):
            p = float(probs[c]) * 100
            letter.config(text=self.classes[c])
            bar["value"] = p
            pct.config(text=f"{p:.1f}%")
        self.status.config(text="Готово")

    def _clear_results(self):
        self.answer.config(text="")
        for letter, bar, pct in self.rows:
            letter.config(text="")
            bar["value"] = 0
            pct.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
