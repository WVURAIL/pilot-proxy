# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("matplotlib")

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
# noinspection PyProtectedMember
from pilot_proxy.chime.frequency_offset import (
    FrequencyOffsetConfig,
    _rolling_nanmedian_baseline,
    _window,
    accumulate_noncoherent_fft_power,
    estimate_frame_peak_offset,
    plot_frequency_offset_products,
    run_frequency_offset_diagnostic,
)
from pilot_proxy.chime.hdf5_input import COMPLEX_FLOAT
from pilot_proxy.paths import CONFIGS_DIR


def _tone_streams(
    *,
    frequency_hz: float,
    sample_rate_hz: float,
    num_samples: int,
    phases: np.ndarray,
) -> np.ndarray:
    n = np.arange(int(num_samples), dtype=np.float64)
    tone = np.exp(2j * np.pi * float(frequency_hz) * n / float(sample_rate_hz))
    return np.asarray(
        [np.exp(1j * phase) * tone for phase in phases],
        dtype=np.complex64,
    )


def test_rolling_nanmedian_baseline_tracks_floor_without_tone() -> None:
    floor = np.linspace(-20.0, -5.0, 101)
    values = floor.copy()
    values[50] = 40.0
    values[10] = np.nan

    baseline = _rolling_nanmedian_baseline(values, window_bins=21)

    assert np.isfinite(baseline).all()
    assert baseline[50] == pytest.approx(floor[50], abs=2.0)
    assert values[50] - baseline[50] > 50.0


def test_known_complex_tone_offset_is_recovered_within_one_fft_bin() -> None:
    sample_rate_hz = 1024.0
    fft_size = 1024
    expected_offset_hz = 100.0
    true_error_hz = 17.0
    streams = _tone_streams(
        frequency_hz=expected_offset_hz + true_error_hz,
        sample_rate_hz=sample_rate_hz,
        num_samples=fft_size,
        phases=np.asarray([0.0, 0.4, 1.1]),
    )
    result = estimate_frame_peak_offset(
        streams[:, np.newaxis, :],
        sample_encoding=COMPLEX_FLOAT,
        spectral_sense="normal",
        sample_rate_hz=sample_rate_hz,
        expected_offset_hz=expected_offset_hz,
        fft_size=fft_size,
        stream_batch_size=2,
        peak_search_half_width_hz=50.0,
        window=_window("rect", fft_size),
        backend="numpy",
    )

    assert float(result["frequency_offset_hz"]) == pytest.approx(
        true_error_hz,
        abs=sample_rate_hz / fft_size,
    )


def test_parabolic_interpolation_improves_subbin_estimate() -> None:
    sample_rate_hz = 1024.0
    fft_size = 1024
    bin_width_hz = sample_rate_hz / fft_size
    expected_offset_hz = 100.0
    true_error_hz = 17.35
    streams = _tone_streams(
        frequency_hz=expected_offset_hz + true_error_hz,
        sample_rate_hz=sample_rate_hz,
        num_samples=fft_size,
        phases=np.asarray([0.0, 1.3, 2.7, 4.0]),
    )
    result = estimate_frame_peak_offset(
        streams[:, np.newaxis, :],
        sample_encoding=COMPLEX_FLOAT,
        spectral_sense="normal",
        sample_rate_hz=sample_rate_hz,
        expected_offset_hz=expected_offset_hz,
        fft_size=fft_size,
        stream_batch_size=2,
        peak_search_half_width_hz=50.0,
        window=_window("hann", fft_size),
        backend="numpy",
    )

    nearest_bin_error = abs(
        true_error_hz - round(true_error_hz / bin_width_hz) * bin_width_hz
    )
    measured_error = abs(float(result["frequency_offset_hz"]) - true_error_hz)
    assert measured_error < nearest_bin_error


def test_noncoherent_fft_power_is_not_a_coherent_stream_sum() -> None:
    sample_rate_hz = 1024.0
    fft_size = 1024
    streams = _tone_streams(
        frequency_hz=123.0,
        sample_rate_hz=sample_rate_hz,
        num_samples=fft_size,
        phases=np.asarray([0.0, np.pi]),
    )

    power, _ = accumulate_noncoherent_fft_power(
        streams,
        window=_window("rect", fft_size),
        stream_batch_size=1,
        backend="numpy",
    )
    coherent = np.fft.fftshift(np.fft.fft(np.sum(streams, axis=0)))

    assert float(np.max(power)) > 1.5 * fft_size**2
    assert float(np.max(np.abs(coherent) ** 2)) < 1.0e-6 * fft_size**2


def test_inverted_spectral_sense_returns_detector_coordinate_sign() -> None:
    sample_rate_hz = 1024.0
    fft_size = 1024
    expected_offset_hz = 100.0
    true_error_hz = 17.0
    streams = _tone_streams(
        frequency_hz=-(expected_offset_hz + true_error_hz),
        sample_rate_hz=sample_rate_hz,
        num_samples=fft_size,
        phases=np.asarray([0.0, 0.2, 0.9]),
    )

    result = estimate_frame_peak_offset(
        streams[:, np.newaxis, :],
        sample_encoding=COMPLEX_FLOAT,
        spectral_sense="inverted",
        sample_rate_hz=sample_rate_hz,
        expected_offset_hz=expected_offset_hz,
        fft_size=fft_size,
        stream_batch_size=2,
        peak_search_half_width_hz=50.0,
        window=_window("rect", fft_size),
        backend="numpy",
    )

    assert float(result["frequency_offset_hz"]) == pytest.approx(
        true_error_hz,
        abs=sample_rate_hz / fft_size,
    )


def test_run_frequency_offset_diagnostic_writes_expected_shapes(tmp_path) -> None:
    input_dir = tmp_path / "input"
    ch_dir = input_dir / "ch0844"
    ch_dir.mkdir(parents=True)
    physical_channel = 14
    frame_size = 1024
    num_frames = 2
    num_streams = 4
    sample_rate_hz = 390_625.0
    expected_offset_hz = 1_000.0
    pilot_hz = physical_channel_to_pilot_hz(physical_channel)
    center_hz = pilot_hz - expected_offset_hz
    streams = _tone_streams(
        frequency_hz=-(expected_offset_hz + 20.0),
        sample_rate_hz=sample_rate_hz,
        num_samples=frame_size * num_frames,
        phases=np.linspace(0.0, 1.0, num_streams),
    )
    data = streams.T
    with h5py.File(ch_dir / "001.h5", "w") as h5:
        h5.attrs["freq"] = center_hz / 1.0e6
        h5.attrs["freq_id"] = 844
        ds = h5.create_dataset("baseband", data=data)
        ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)

    output_dir = tmp_path / "run"
    outputs = run_frequency_offset_diagnostic(
        FrequencyOffsetConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            physical_channels=[physical_channel],
            receiver_profile=(
                CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
            ),
            frame_size_samples=frame_size,
            fft_size=frame_size,
            stream_batch_size=2,
            max_frames=num_frames,
            peak_search_half_width_hz=2_000.0,
            window_name="hann",
            backend="numpy",
            plot=False,
        )
    )

    products = np.load(outputs["frequency_offset_outputs"])
    assert products["frequency_offset_hz"].shape == (num_frames, 1)
    assert products["valid"].shape == (num_frames, 1)
    assert products["fft_frequency_axis_hz"].shape == (frame_size,)
    assert products["time_average_spectrum_power_linear"].shape == (1, frame_size)
    assert products["time_average_spectrum_count"].shape == (1,)
    assert int(products["time_average_spectrum_count"][0]) == num_frames
    assert float(products["fft_bin_width_hz"]) == pytest.approx(
        sample_rate_hz / frame_size
    )
    assert products["coordinate_system"].item() == (
        "post_spectral_sense_normalization_rf_offset"
    )
    assert outputs["summary_table"].exists()
    assert outputs["input_manifest"].exists()
    assert outputs["stats"].exists()


def test_frequency_offset_plots_from_synthetic_npz(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.savez_compressed(
        run_dir / "frequency_offset_outputs.npz",
        physical_channel=np.asarray([14, 30], dtype=np.int32),
        pilot_frequency_hz=np.asarray([470_309_441.0, 566_309_441.0]),
        chime_frequency_hz=np.asarray([470_312_500.0, 566_406_250.0]),
        coarse_channel_center_hz=np.asarray([470_312_500.0, 566_406_250.0]),
        expected_pilot_offset_hz=np.asarray([-3059.0, -96_809.0]),
        frame_index=np.asarray([0, 1, 2], dtype=np.int64),
        relative_time_s=np.asarray([0.0, 1.0, 2.0]),
        peak_offset_hz=np.zeros((3, 2)),
        frequency_offset_hz=np.asarray([[0.0, 20.0], [5.0, -30.0], [-4.0, 15.0]]),
        peak_power_linear=np.ones((3, 2)),
        local_floor_power_linear=np.full((3, 2), 0.1),
        peak_prominence_db=np.asarray([[10.0, 8.0], [11.0, 7.0], [9.0, 8.5]]),
        valid=np.ones((3, 2), dtype=np.uint8),
        fft_size=np.asarray(1024),
        fft_frequency_axis_hz=np.fft.fftshift(
            np.fft.fftfreq(1024, d=1.0 / 390_625.0)
        ),
        time_average_spectrum_power_linear=np.vstack(
            [
                np.linspace(1.0, 4.0, 1024),
                np.linspace(2.0, 5.0, 1024),
            ]
        ),
        time_average_spectrum_count=np.asarray([3, 3], dtype=np.uint64),
        fft_bin_width_hz=np.asarray(381.4697265625),
        sample_rate_hz=np.asarray(390_625.0),
        window_name=np.asarray("hann"),
        peak_search_half_width_hz=np.asarray(5_000.0),
        coordinate_system=np.asarray("post_spectral_sense_normalization_rf_offset"),
    )

    outputs = plot_frequency_offset_products(run_dir)

    expected = {
        run_dir / "figures" / "frequency_offset_histogram_by_pilot.png",
        run_dir / "figures" / "frequency_offset_spectrogram.png",
        run_dir / "figures" / "peak_prominence_spectrogram.png",
        run_dir
        / "figures"
        / "frequency_offset_time_average_spectrum_by_pilot.png",
    }
    assert expected.issubset(set(outputs))
    for path in expected:
        assert path.exists()
