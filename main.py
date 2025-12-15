from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from pipeline import PipelineSummary, run_pipeline

BUTTON_IDLE_TEXT = "Presioname :)"
BUTTON_RUNNING_TEXT = "Procesando..."
BASE_DIR = Path(__file__).resolve().parent


class FotoHelperApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("FotoHelper · Pipeline")
        self.root.geometry("360x240")
        self.root.resizable(False, False)

        self.status_var = tk.StringVar(
            value="Selecciona una carpeta de fotos"
        )
        self.progress_var = tk.StringVar(value="")

        self.action_button = tk.Button(
            self.root,
            text=BUTTON_IDLE_TEXT,
            command=self._handle_button_press,
            bg="#1c7ed6",
            fg="white",
            activebackground="#1864ab",
            padx=20,
            pady=12,
            relief=tk.FLAT,
        )
        self.action_button.pack(pady=30)

        status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            wraplength=300,
            justify=tk.CENTER,
        )
        status_label.pack(padx=20)

        progress_label = tk.Label(
            self.root,
            textvariable=self.progress_var,
            fg="#343a40",
            justify=tk.CENTER,
        )
        progress_label.pack(padx=20, pady=(8, 16))

    def _handle_button_press(self) -> None:
        folder_path = filedialog.askdirectory(
            parent=self.root,
            title="Select folder containing images",
        )
        if not folder_path:
            return

        file_paths = self._collect_images_from_folder(Path(folder_path))
        if not file_paths:
            messagebox.showwarning(
                "No images found",
                "The selected folder does not contain any supported images.",
                parent=self.root,
            )
            return

        self._set_running_state(True)
        worker = threading.Thread(
            target=self._run_pipeline,
            args=(file_paths,),
            daemon=True,
        )
        worker.start()

    def _collect_images_from_folder(self, directory: Path) -> list[str]:
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images: list[str] = []
        for candidate in sorted(directory.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in supported_extensions:
                continue
            images.append(str(candidate))
        return images

    def _run_pipeline(self, file_paths: tuple[str, ...] | list[str]) -> None:
        try:
            summary = run_pipeline(
                file_paths,
                output_root=BASE_DIR,
                progress_callback=self._report_progress,
            )
        except Exception as exc:  # noqa: BLE001 - show user-friendly error
            error = exc
            self.root.after(0, lambda: self._handle_error(error))
            return

        summary_result = summary
        self.root.after(0, lambda: self._handle_success(summary_result))

    def _report_progress(self, stage: str, current: int, total: int) -> None:
        message = f"{stage}: {current}/{total}"
        self.root.after(0, lambda: self.progress_var.set(message))

    def _handle_success(self, summary: PipelineSummary) -> None:
        message = (
            f"Created {len(summary.batches)} person batch(es) "
            f"covering {summary.total_images} photo(s)."
        )
        if summary.skipped_files:
            message += (
                f" Skipped {len(summary.skipped_files)} without detectable faces."
            )

        self.progress_var.set("")
        self.status_var.set(message + " Output: 'result/' folder.")
        messagebox.showinfo("Pipeline complete", message, parent=self.root)
        self._set_running_state(False)

    def _handle_error(self, error: Exception) -> None:
        messagebox.showerror("Pipeline failed", str(error), parent=self.root)
        self.progress_var.set("")
        self.status_var.set(
            "Something went wrong. Please review your images and retry."
        )
        self._set_running_state(False)

    def _set_running_state(self, is_running: bool) -> None:
        if is_running:
            self.action_button.configure(
                state=tk.DISABLED,
                text=BUTTON_RUNNING_TEXT,
                bg="#adb5bd",
                fg="#495057",
            )
        else:
            self.action_button.configure(
                state=tk.NORMAL,
                text=BUTTON_IDLE_TEXT,
                bg="#1c7ed6",
                fg="white",
            )
            self.progress_var.set("")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = FotoHelperApp()
    app.run()


if __name__ == "__main__":
    main()
