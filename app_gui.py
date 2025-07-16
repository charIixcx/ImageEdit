"""Basic PyQt6 GUI for previewing image effects."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import Qt

from typing import List

from features.effects import (
    load_midas_model,
    generate_depth_map,
)

from app_terminal import (
    generate_effect_preview,
    numpy_to_qpixmap,
)


class MainWindow(QtWidgets.QWidget):
    """Main window allowing image selection and effect previews."""

    EFFECTS: List[str] = [
        "bokeh",
        "particles",
        "displacement",
        "distortion",
        "wave",
        "swirl",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Effect Preview")
        self.resize(900, 600)

        self.photo_path: str | None = None
        self.depth_arr: np.ndarray | None = None

        self.midas = None
        self.transform = None
        self.device = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        open_btn = QtWidgets.QPushButton("Open Image")
        open_btn.clicked.connect(self.open_image)
        layout.addWidget(open_btn)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(container)
        self.scroll_area.setWidget(container)

    def open_image(self) -> None:
        """Open an image file and generate previews."""

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        self.photo_path = file_path
        if self.midas is None:
            self.midas, self.transform, self.device = load_midas_model()

        self.depth_arr = generate_depth_map(
            self.photo_path, self.midas, self.transform, self.device
        )

        self.show_previews()

    def show_previews(self) -> None:
        """Generate and display previews for all effects."""

        if not self.photo_path or self.depth_arr is None:
            return

        # Clear previous previews
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for i, effect in enumerate(self.EFFECTS):
            preview = generate_effect_preview(self.photo_path, self.depth_arr, effect)
            pixmap = numpy_to_qpixmap(preview)
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            caption = QtWidgets.QLabel(effect.capitalize())
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)

            container = QtWidgets.QVBoxLayout()
            widget = QtWidgets.QWidget()
            container.addWidget(label)
            container.addWidget(caption)
            widget.setLayout(container)

            self.grid.addWidget(widget, i // 3, i % 3)


def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
