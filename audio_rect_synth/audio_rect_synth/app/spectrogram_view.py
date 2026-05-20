from __future__ import annotations

from typing import Iterable

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from audio_rect_synth.core.rectangle_model import RectangleFunction, TimeFrequencySelection


class SpectrogramView(QtWidgets.QWidget):
    """Interactive spectrogram widget with editable time-frequency ROIs."""

    selection_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")

        self._times: np.ndarray | None = None
        self._freqs: np.ndarray | None = None
        self._selection_rois: list[pg.RectROI] = []
        self._fit_rois: list[pg.RectROI] = []

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setMenuEnabled(True)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    @property
    def selection_count(self) -> int:
        return len(self._selection_rois)

    def set_spectrogram(self, db_spectrogram: np.ndarray, times: np.ndarray, freqs: np.ndarray) -> None:
        image = np.asarray(db_spectrogram, dtype=np.float32)
        time_axis = np.asarray(times, dtype=np.float64)
        freq_axis = np.asarray(freqs, dtype=np.float64)
        if image.shape != (freq_axis.size, time_axis.size):
            raise ValueError("db_spectrogram must have shape (len(freqs), len(times)).")

        self._times = time_axis
        self._freqs = freq_axis

        finite = image[np.isfinite(image)]
        if finite.size:
            low = float(np.quantile(finite, 0.02))
            high = float(np.quantile(finite, 0.995))
            if high <= low:
                high = low + 1.0
        else:
            low, high = -120.0, 0.0

        self.image_item.setImage(image, autoLevels=False, levels=(low, high))
        if time_axis.size > 1 and freq_axis.size > 1:
            rect = QtCore.QRectF(
                float(time_axis[0]),
                float(freq_axis[0]),
                max(1e-9, float(time_axis[-1] - time_axis[0])),
                max(1e-9, float(freq_axis[-1] - freq_axis[0])),
            )
            self.image_item.setRect(rect)
        self.plot.autoRange()

    def add_selection(
        self,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        f_low: float | None = None,
        f_high: float | None = None,
        region_id: str | None = None,
    ) -> TimeFrequencySelection:
        if self._times is None or self._freqs is None:
            raise RuntimeError("Load an audio file before adding a selection.")

        duration = float(self._times[-1]) if self._times.size else 1.0
        nyquist = float(self._freqs[-1]) if self._freqs.size else 22050.0
        t0 = 0.0 if t_start is None else float(t_start)
        t1 = min(duration, t0 + 1.0) if t_end is None else float(t_end)
        f0 = min(200.0, nyquist * 0.25) if f_low is None else float(f_low)
        f1 = min(4000.0, nyquist) if f_high is None else float(f_high)
        if f1 <= f0:
            f1 = min(nyquist, f0 + max(100.0, nyquist * 0.1))
        if t1 <= t0:
            t1 = min(duration, t0 + 0.1)

        selection = TimeFrequencySelection(t_start=t0, t_end=t1, f_low=f0, f_high=f1, region_id=region_id or "")
        selection = selection.normalized()
        if not selection.region_id:
            selection = TimeFrequencySelection(
                t_start=selection.t_start,
                t_end=selection.t_end,
                f_low=selection.f_low,
                f_high=selection.f_high,
            )

        roi = pg.RectROI(
            pos=[selection.t_start, selection.f_low],
            size=[selection.t_end - selection.t_start, selection.f_high - selection.f_low],
            pen=pg.mkPen("y", width=2),
            movable=True,
            removable=False,
        )
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 0], [0, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        roi._region_id = selection.region_id  # type: ignore[attr-defined]
        roi.sigRegionChanged.connect(lambda *_: self.selection_changed.emit())
        self.plot.addItem(roi)
        self._selection_rois.append(roi)
        self.selection_changed.emit()
        return selection

    def remove_selection(self, index: int) -> None:
        if not 0 <= index < len(self._selection_rois):
            return
        roi = self._selection_rois.pop(index)
        self.plot.removeItem(roi)
        self.selection_changed.emit()

    def clear_selections(self) -> None:
        for roi in self._selection_rois:
            self.plot.removeItem(roi)
        self._selection_rois.clear()
        self.selection_changed.emit()

    def get_selections(self) -> list[TimeFrequencySelection]:
        selections: list[TimeFrequencySelection] = []
        for index, roi in enumerate(self._selection_rois):
            pos = roi.pos()
            size = roi.size()
            t0 = float(pos.x())
            f0 = float(pos.y())
            t1 = t0 + float(size.x())
            f1 = f0 + float(size.y())
            region_id = getattr(roi, "_region_id", f"region-{index + 1}")
            selections.append(
                TimeFrequencySelection(
                    t_start=t0,
                    t_end=t1,
                    f_low=f0,
                    f_high=f1,
                    region_id=str(region_id),
                ).normalized()
            )
        return selections

    def set_fit_rectangles(self, rectangles: Iterable[RectangleFunction]) -> None:
        self.clear_fit_rectangles()
        for rectangle in rectangles:
            rect = rectangle.normalized()
            width = max(1e-9, rect.t_end - rect.t_start)
            height = max(1e-9, rect.f_high - rect.f_low)
            roi = pg.RectROI(
                pos=[rect.t_start, rect.f_low],
                size=[width, height],
                pen=pg.mkPen("c", width=1),
                movable=False,
                removable=False,
            )
            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            self.plot.addItem(roi)
            self._fit_rois.append(roi)

    def clear_fit_rectangles(self) -> None:
        for roi in self._fit_rois:
            self.plot.removeItem(roi)
        self._fit_rois.clear()
