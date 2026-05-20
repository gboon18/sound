from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

from audio_rect_synth.core.audio_io import AudioData, load_audio, write_wav
from audio_rect_synth.core.playback import play_audio, stop_audio
from audio_rect_synth.core.rectangle_fit import RectangleFitResult, RectangleFitSettings, fit_rectangle_model
from audio_rect_synth.core.rectangle_model import save_rectangle_model
from audio_rect_synth.core.reconstruct import ReconstructionResult, reconstruct_from_rectangles
from audio_rect_synth.core.stft import STFTConfig, compute_stft, stft_to_db
from audio_rect_synth.app.spectrogram_view import SpectrogramView


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Rect Synth")
        self.resize(1280, 820)

        self.audio: AudioData | None = None
        self.mono: np.ndarray | None = None
        self.config: STFTConfig | None = None
        self.freqs: np.ndarray | None = None
        self.times: np.ndarray | None = None
        self.zxx: np.ndarray | None = None
        self.fit_result: RectangleFitResult | None = None
        self.reconstruction: ReconstructionResult | None = None

        self.spectrogram_view = SpectrogramView()
        self.spectrogram_view.selection_changed.connect(self._refresh_selection_list)

        controls = self._build_controls()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.spectrogram_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Open a .wav, .mp3, or .m4a file.")

    def _build_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(330)
        panel.setMaximumWidth(430)
        layout = QtWidgets.QVBoxLayout(panel)

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        open_button = QtWidgets.QPushButton("Open audio")
        open_button.clicked.connect(self._open_audio_dialog)
        layout.addWidget(open_button)

        stft_group = QtWidgets.QGroupBox("Spectrogram")
        stft_layout = QtWidgets.QFormLayout(stft_group)
        self.n_fft_spin = QtWidgets.QSpinBox()
        self.n_fft_spin.setRange(256, 32768)
        self.n_fft_spin.setSingleStep(256)
        self.n_fft_spin.setValue(4096)
        self.hop_spin = QtWidgets.QSpinBox()
        self.hop_spin.setRange(64, 32768)
        self.hop_spin.setSingleStep(64)
        self.hop_spin.setValue(1024)
        recompute_button = QtWidgets.QPushButton("Recompute spectrogram")
        recompute_button.clicked.connect(self._recompute_spectrogram)
        stft_layout.addRow("n_fft", self.n_fft_spin)
        stft_layout.addRow("hop_length", self.hop_spin)
        stft_layout.addRow(recompute_button)
        layout.addWidget(stft_group)

        selection_group = QtWidgets.QGroupBox("Selections")
        selection_layout = QtWidgets.QVBoxLayout(selection_group)
        self.selection_list = QtWidgets.QListWidget()
        selection_layout.addWidget(self.selection_list)
        selection_buttons = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Add")
        delete_button = QtWidgets.QPushButton("Delete")
        clear_button = QtWidgets.QPushButton("Clear")
        add_button.clicked.connect(self._add_selection)
        delete_button.clicked.connect(self._delete_selection)
        clear_button.clicked.connect(self._clear_selections)
        selection_buttons.addWidget(add_button)
        selection_buttons.addWidget(delete_button)
        selection_buttons.addWidget(clear_button)
        selection_layout.addLayout(selection_buttons)
        layout.addWidget(selection_group, stretch=1)

        fit_group = QtWidgets.QGroupBox("Rectangle fitting")
        fit_layout = QtWidgets.QFormLayout(fit_group)
        self.min_rects_spin = QtWidgets.QSpinBox()
        self.min_rects_spin.setRange(1, 128)
        self.min_rects_spin.setValue(1)
        self.max_rects_spin = QtWidgets.QSpinBox()
        self.max_rects_spin.setRange(1, 128)
        self.max_rects_spin.setValue(6)
        self.slice_ms_spin = QtWidgets.QDoubleSpinBox()
        self.slice_ms_spin.setRange(5.0, 500.0)
        self.slice_ms_spin.setSingleStep(5.0)
        self.slice_ms_spin.setValue(40.0)
        self.overlap_spin = QtWidgets.QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.95)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setValue(0.50)
        fit_button = QtWidgets.QPushButton("Fit selected areas")
        fit_button.clicked.connect(self._fit_rectangles)
        fit_layout.addRow("min rectangles", self.min_rects_spin)
        fit_layout.addRow("max rectangles", self.max_rects_spin)
        fit_layout.addRow("slice ms", self.slice_ms_spin)
        fit_layout.addRow("slice overlap", self.overlap_spin)
        fit_layout.addRow(fit_button)
        layout.addWidget(fit_group)

        recon_group = QtWidgets.QGroupBox("Reconstruction")
        recon_layout = QtWidgets.QFormLayout(recon_group)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Rectangle synth only", "rectangles")
        self.mode_combo.addItem("Original inside rectangles", "masked_source")
        self.mode_combo.addItem("Original + rectangles", "mix")
        self.mode_combo.addItem("Original minus rectangles", "remove")
        reconstruct_button = QtWidgets.QPushButton("Reconstruct")
        reconstruct_button.clicked.connect(self._reconstruct)
        recon_layout.addRow("mode", self.mode_combo)
        recon_layout.addRow(reconstruct_button)
        layout.addWidget(recon_group)

        playback_layout = QtWidgets.QGridLayout()
        play_original_button = QtWidgets.QPushButton("Play original")
        play_recon_button = QtWidgets.QPushButton("Play reconstruction")
        stop_button = QtWidgets.QPushButton("Stop")
        play_original_button.clicked.connect(self._play_original)
        play_recon_button.clicked.connect(self._play_reconstruction)
        stop_button.clicked.connect(self._stop_playback)
        playback_layout.addWidget(play_original_button, 0, 0)
        playback_layout.addWidget(play_recon_button, 0, 1)
        playback_layout.addWidget(stop_button, 1, 0, 1, 2)
        layout.addLayout(playback_layout)

        export_layout = QtWidgets.QGridLayout()
        export_wav_button = QtWidgets.QPushButton("Export WAV")
        export_model_button = QtWidgets.QPushButton("Export JSON model")
        export_wav_button.clicked.connect(self._export_wav)
        export_model_button.clicked.connect(self._export_model)
        export_layout.addWidget(export_wav_button, 0, 0)
        export_layout.addWidget(export_model_button, 0, 1)
        layout.addLayout(export_layout)

        self.fit_summary_label = QtWidgets.QLabel("No fit yet")
        self.fit_summary_label.setWordWrap(True)
        layout.addWidget(self.fit_summary_label)
        layout.addStretch(0)
        return panel

    def _open_audio_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open audio file",
            str(Path.home()),
            "Audio files (*.wav *.mp3 *.m4a)",
        )
        if not path:
            return
        self._run_guarded(lambda: self._load_audio(Path(path)))

    def _load_audio(self, path: Path) -> None:
        self.statusBar().showMessage(f"Loading {path.name}...")
        QtWidgets.QApplication.processEvents()
        self.audio = load_audio(path, mono=True)
        self.mono = self.audio.mono()
        self.file_label.setText(f"{path}\n{self.audio.sample_rate} Hz, {self.audio.duration_seconds:.2f} s")
        self._compute_and_display_stft()
        self.spectrogram_view.clear_selections()
        self.spectrogram_view.clear_fit_rectangles()
        self.fit_result = None
        self.reconstruction = None
        self.fit_summary_label.setText("No fit yet")
        self.statusBar().showMessage(f"Loaded {path.name}")

    def _recompute_spectrogram(self) -> None:
        if self.audio is None:
            self._show_info("Open an audio file first.")
            return
        self._run_guarded(self._compute_and_display_stft)

    def _compute_and_display_stft(self) -> None:
        if self.audio is None or self.mono is None:
            raise RuntimeError("No audio loaded.")
        n_fft = int(self.n_fft_spin.value())
        hop_length = int(self.hop_spin.value())
        if self.mono.shape[0] > 0:
            n_fft = min(n_fft, max(16, self.mono.shape[0]))
        hop_length = min(hop_length, n_fft)
        self.config = STFTConfig(sample_rate=self.audio.sample_rate, n_fft=n_fft, hop_length=hop_length)
        self.statusBar().showMessage("Computing STFT...")
        QtWidgets.QApplication.processEvents()
        self.freqs, self.times, self.zxx = compute_stft(self.mono, self.config)
        db = stft_to_db(self.zxx)
        self.spectrogram_view.set_spectrogram(db, self.times, self.freqs)
        self.statusBar().showMessage(f"Spectrogram ready: n_fft={n_fft}, hop={hop_length}")

    def _add_selection(self) -> None:
        self._run_guarded(lambda: self.spectrogram_view.add_selection())

    def _delete_selection(self) -> None:
        row = self.selection_list.currentRow()
        if row < 0:
            row = self.spectrogram_view.selection_count - 1
        self.spectrogram_view.remove_selection(row)

    def _clear_selections(self) -> None:
        self.spectrogram_view.clear_selections()
        self.spectrogram_view.clear_fit_rectangles()
        self.fit_result = None
        self.reconstruction = None
        self.fit_summary_label.setText("No fit yet")

    def _fit_rectangles(self) -> None:
        self._run_guarded(self._fit_rectangles_impl)

    def _fit_rectangles_impl(self) -> None:
        if self.audio is None or self.config is None or self.freqs is None or self.times is None or self.zxx is None:
            raise RuntimeError("Open an audio file first.")
        selections = self.spectrogram_view.get_selections()
        if not selections:
            raise RuntimeError("Add at least one time-frequency selection first.")

        min_rects = int(self.min_rects_spin.value())
        max_rects = max(min_rects, int(self.max_rects_spin.value()))
        self.max_rects_spin.setValue(max_rects)
        settings = RectangleFitSettings(
            min_rects=min_rects,
            max_rects=max_rects,
            slice_duration_ms=float(self.slice_ms_spin.value()),
            slice_overlap=float(self.overlap_spin.value()),
        )

        self.statusBar().showMessage("Fitting rectangles...")
        QtWidgets.QApplication.processEvents()
        self.fit_result = fit_rectangle_model(
            self.zxx,
            self.freqs,
            self.times,
            self.config,
            selections,
            settings,
            source_path=str(self.audio.path) if self.audio.path is not None else None,
        )
        self.reconstruction = None
        self.spectrogram_view.set_fit_rectangles(self.fit_result.model.rectangles)
        self.fit_summary_label.setText(
            f"Rectangles: {self.fit_result.rectangle_count}\n"
            f"Active-bin MSE: {self.fit_result.mean_squared_error:.6g}"
        )
        self.statusBar().showMessage("Rectangle fit complete.")

    def _reconstruct(self) -> None:
        self._run_guarded(self._reconstruct_impl)

    def _reconstruct_impl(self) -> None:
        if self.audio is None or self.mono is None or self.freqs is None or self.times is None or self.zxx is None:
            raise RuntimeError("Open an audio file first.")
        if self.fit_result is None:
            self._fit_rectangles_impl()
            if self.fit_result is None:
                raise RuntimeError("Fit failed.")

        mode = str(self.mode_combo.currentData())
        self.statusBar().showMessage("Reconstructing audio...")
        QtWidgets.QApplication.processEvents()
        self.reconstruction = reconstruct_from_rectangles(
            self.fit_result.model,
            self.zxx,
            self.freqs,
            self.times,
            target_length=self.mono.shape[0],
            mode=mode,  # type: ignore[arg-type]
        )
        self.statusBar().showMessage(f"Reconstruction ready: {mode}")

    def _play_original(self) -> None:
        if self.audio is None or self.mono is None:
            self._show_info("Open an audio file first.")
            return
        self._run_guarded(lambda: play_audio(self.mono, self.audio.sample_rate))

    def _play_reconstruction(self) -> None:
        if self.audio is None:
            self._show_info("Open an audio file first.")
            return
        if self.reconstruction is None:
            self._reconstruct()
            if self.reconstruction is None:
                return
        self._run_guarded(lambda: play_audio(self.reconstruction.waveform, self.audio.sample_rate))

    def _stop_playback(self) -> None:
        self._run_guarded(stop_audio)

    def _export_wav(self) -> None:
        if self.audio is None:
            self._show_info("Open an audio file first.")
            return
        if self.reconstruction is None:
            self._reconstruct()
            if self.reconstruction is None:
                return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export reconstructed WAV",
            str(Path.home() / "reconstructed.wav"),
            "WAV audio (*.wav)",
        )
        if not path:
            return
        self._run_guarded(lambda: write_wav(Path(path), self.reconstruction.waveform, self.audio.sample_rate))
        self.statusBar().showMessage(f"Exported {path}")

    def _export_model(self) -> None:
        if self.fit_result is None:
            self._show_info("Fit rectangles before exporting a model.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export rectangle model",
            str(Path.home() / "rectangle_model.json"),
            "JSON files (*.json)",
        )
        if not path:
            return
        self._run_guarded(lambda: save_rectangle_model(Path(path), self.fit_result.model))
        self.statusBar().showMessage(f"Exported {path}")

    def _refresh_selection_list(self) -> None:
        self.selection_list.clear()
        for index, selection in enumerate(self.spectrogram_view.get_selections(), start=1):
            self.selection_list.addItem(
                f"{index}: {selection.t_start:.3f}-{selection.t_end:.3f} s, "
                f"{selection.f_low:.1f}-{selection.f_high:.1f} Hz"
            )

    def _run_guarded(self, function) -> None:  # type: ignore[no-untyped-def]
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage("Error")
            self._show_error(str(exc), traceback.format_exc())

    def _show_info(self, message: str) -> None:
        QtWidgets.QMessageBox.information(self, "Audio Rect Synth", message)

    def _show_error(self, message: str, details: str | None = None) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("Audio Rect Synth error")
        box.setText(message)
        if details:
            box.setDetailedText(details)
        box.exec()
