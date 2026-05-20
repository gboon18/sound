from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import uuid


@dataclass(frozen=True)
class TimeFrequencySelection:
    t_start: float
    t_end: float
    f_low: float
    f_high: float
    region_id: str = field(default_factory=lambda: f"region-{uuid.uuid4().hex[:8]}")

    def normalized(self) -> "TimeFrequencySelection":
        t0, t1 = sorted((float(self.t_start), float(self.t_end)))
        f0, f1 = sorted((float(self.f_low), float(self.f_high)))
        return TimeFrequencySelection(t_start=t0, t_end=t1, f_low=f0, f_high=f1, region_id=self.region_id)

    def validate(self) -> None:
        item = self.normalized()
        if item.t_end <= item.t_start:
            raise ValueError("Selection time span must be positive.")
        if item.f_high <= item.f_low:
            raise ValueError("Selection frequency span must be positive.")


@dataclass(frozen=True)
class RectangleFunction:
    t_start: float
    t_end: float
    f_low: float
    f_high: float
    amplitude: float
    source_region_id: str | None = None
    slice_index: int | None = None
    error: float | None = None

    def normalized(self) -> "RectangleFunction":
        t0, t1 = sorted((float(self.t_start), float(self.t_end)))
        f0, f1 = sorted((float(self.f_low), float(self.f_high)))
        return RectangleFunction(
            t_start=t0,
            t_end=t1,
            f_low=f0,
            f_high=f1,
            amplitude=max(0.0, float(self.amplitude)),
            source_region_id=self.source_region_id,
            slice_index=self.slice_index,
            error=self.error,
        )

    def validate(self) -> None:
        item = self.normalized()
        if item.t_end <= item.t_start:
            raise ValueError("Rectangle time span must be positive.")
        if item.f_high <= item.f_low:
            raise ValueError("Rectangle frequency span must be positive.")
        if item.amplitude < 0.0:
            raise ValueError("Rectangle amplitude must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self.normalized())

    @staticmethod
    def from_dict(data: dict[str, object]) -> "RectangleFunction":
        return RectangleFunction(
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            f_low=float(data["f_low"]),
            f_high=float(data["f_high"]),
            amplitude=float(data["amplitude"]),
            source_region_id=data.get("source_region_id") if data.get("source_region_id") is None else str(data.get("source_region_id")),
            slice_index=data.get("slice_index") if data.get("slice_index") is None else int(data.get("slice_index")),
            error=data.get("error") if data.get("error") is None else float(data.get("error")),
        ).normalized()


@dataclass(frozen=True)
class RectangleModel:
    sample_rate: int
    n_fft: int
    hop_length: int
    window: str
    rectangles: list[RectangleFunction]
    source_path: str | None = None
    description: str = "audio-rect-synth rectangle model"

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive.")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive.")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must be <= n_fft.")
        for rectangle in self.rectangles:
            rectangle.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "description": self.description,
            "source_path": self.source_path,
            "sample_rate": int(self.sample_rate),
            "n_fft": int(self.n_fft),
            "hop_length": int(self.hop_length),
            "window": self.window,
            "rectangles": [rect.to_dict() for rect in self.rectangles],
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "RectangleModel":
        rectangles_raw = data.get("rectangles", [])
        if not isinstance(rectangles_raw, list):
            raise ValueError("rectangles must be a list.")
        rectangles = [RectangleFunction.from_dict(item) for item in rectangles_raw]
        model = RectangleModel(
            sample_rate=int(data["sample_rate"]),
            n_fft=int(data["n_fft"]),
            hop_length=int(data["hop_length"]),
            window=str(data.get("window", "hann")),
            rectangles=rectangles,
            source_path=data.get("source_path") if data.get("source_path") is None else str(data.get("source_path")),
            description=str(data.get("description", "audio-rect-synth rectangle model")),
        )
        model.validate()
        return model


def save_rectangle_model(path: str | Path, model: RectangleModel) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_rectangle_model(path: str | Path) -> RectangleModel:
    input_path = Path(path).expanduser().resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Model file must contain a JSON object.")
    return RectangleModel.from_dict(data)
