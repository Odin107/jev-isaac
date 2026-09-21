"""A masked, paste-friendly key prompt; credentials never leave process memory."""
from __future__ import annotations


def prompt_key_window(provider_name: str) -> str:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        raise RuntimeError("The key window needs Python with Tkinter installed.") from None

    entered = ""
    try:
        root = tk.Tk()
    except tk.TclError:
        raise RuntimeError("The key window could not open. No API request was made.") from None
    root.title(f"Jev - {provider_name} API key")
    root.resizable(False, False)
    panel = ttk.Frame(root, padding=20)
    panel.pack(fill="both", expand=True)
    ttk.Label(panel, text=f"Paste your {provider_name} API key below (Ctrl+V).\n"
              "It stays hidden and is used only for this session.").pack(anchor="w")
    entry = ttk.Entry(panel, show="*", width=64)
    entry.pack(fill="x", pady=(14, 8))
    error = ttk.Label(panel, text="", foreground="#b00020", wraplength=450)
    error.pack(anchor="w")

    def submit(_event=None):
        nonlocal entered
        value = entry.get().strip()
        if not value:
            error.configure(text="Paste the API key to continue, or choose Cancel.")
            return
        if any(ord(char) < 33 or ord(char) > 126 for char in value):
            error.configure(text="The pasted text contains a space or an unsupported character. "
                            "Copy only the API key value and paste it again.")
            entry.selection_range(0, "end")
            entry.focus_set()
            return
        entered = value
        root.destroy()

    def cancel(_event=None):
        root.destroy()

    buttons = ttk.Frame(panel)
    buttons.pack(anchor="e", pady=(12, 0))
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="left", padx=(0, 8))
    ttk.Button(buttons, text="Continue", command=submit).pack(side="left")
    root.bind("<Return>", submit)
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - root.winfo_width()) // 2)
    y = max(0, (root.winfo_screenheight() - root.winfo_height()) // 2)
    root.geometry(f"+{x}+{y}")
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    entry.focus_force()
    root.mainloop()
    return entered
