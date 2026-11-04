"""Controlled channels. No reference-based alignment or silent codec fallback."""
from __future__ import annotations

import hashlib
import re
import struct
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import read_audio, resample_audio, validate_audio, write_audio
from .features import vad_mask
from .utils import sha256_file, write_json


CHANNELS = {"clean", "downsample_8k", "noise20_downsample_8k",
            "noise15_downsample_8k", "opus8k_8kbps"}


def power(x: np.ndarray) -> float:
    return float(np.mean((x - np.mean(x)) ** 2))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float | None:
    a, b = power(signal), power(noise)
    return float(10 * np.log10(a / b)) if a > 0 and b > 0 else None


def paired_noise(reference: np.ndarray, uid: str, seed: int, snr: float) -> tuple[np.ndarray, dict]:
    """Same source ID/seed gives one normalized realization across labels and SNRs."""
    reference = validate_audio(reference, 16000)
    if not np.isfinite(snr) or power(reference) <= 1e-20:
        raise ValueError("Noise requires finite SNR and nonzero de-meaned reference power")
    seed_bytes = hashlib.sha256(f"speechwm-noise-v1\0{seed}\0{uid}".encode()).digest()
    stable_seed = int.from_bytes(seed_bytes[:16], "little")
    z = np.random.Generator(np.random.PCG64(stable_seed)).standard_normal(len(reference))
    z -= z.mean()
    z /= np.sqrt(np.mean(z ** 2))
    noise = z * np.sqrt(power(reference) / 10 ** (snr / 10))
    # VAD here is for power diagnostics only; the detector independently uses received audio.
    times = np.arange(len(reference)) / 16000
    voiced = vad_mask(reference, 16000, times)
    after_ref = resample_audio(reference, 16000, 8000)
    after_noise = resample_audio(noise, 16000, 8000)
    return noise, {
        "noise_seed": str(stable_seed), "noise_rng": "PCG64",
        "noise_seed_derivation": "SHA256(speechwm-noise-v1 NUL seed NUL utterance_id), first 128 bits little-endian",
        "standard_noise_sha256": hashlib.sha256(z.astype("<f8").tobytes()).hexdigest(),
        "noise_sha256": hashlib.sha256(noise.astype("<f8").tobytes()).hexdigest(),
        "noise_reference": "unmarked source with common gain; whole segment, de-meaned power",
        "nominal_snr_db": snr, "reference_power": power(reference), "noise_power": power(noise),
        "actual_reference_snr_pre_db": snr_db(reference, noise),
        "actual_reference_snr_post_db": snr_db(after_ref, after_noise),
        "reference_voiced_samples": int(voiced.sum()),
        "reference_voiced_power": power(reference[voiced]) if voiced.any() else None,
        "actual_reference_voiced_snr_pre_db": snr_db(reference[voiced], noise[voiced]) if voiced.any() else None,
        "noise_order": "add at 16000 Hz, then scipy resample_poly to 8000 Hz",
    }


@lru_cache(maxsize=1)
def codec_runtime() -> dict:
    import imageio_ffmpeg

    # Do not use get_ffmpeg_exe(): it may silently select an unrelated PATH binary.
    folder = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
    candidates = [p for p in folder.glob("ffmpeg-*") if p.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one packaged FFmpeg executable: {candidates}")
    executable = candidates[0]
    version = subprocess.run([str(executable), "-version"], check=True, capture_output=True, text=True)
    help_result = subprocess.run([str(executable), "-hide_banner", "-h", "encoder=libopus"],
                                 check=True, capture_output=True, text=True)
    help_text = help_result.stdout + help_result.stderr
    if "Encoder libopus" not in help_text:
        raise RuntimeError("Packaged FFmpeg lacks libopus encoder")
    embedded = sorted({s.decode() for s in re.findall(rb"libopus [0-9][0-9A-Za-z.\-]*", executable.read_bytes())})
    return {"executable": str(executable), "executable_sha256": sha256_file(executable),
            "imageio_ffmpeg_version": imageio_ffmpeg.__version__,
            "version_output": version.stdout, "encoder_help": help_text,
            "libopus_embedded_version_strings": embedded,
            "libopus_version_basis": "embedded binary strings; not a runtime opus_get_version_string call",
            "dtx_control_exposed": "-dtx " in help_text,
            "dtx_policy": "explicit off if exposed; otherwise unchanged libopus default off (FFmpeg 7.1 wrapper)",
            "selection": "imageio_ffmpeg package binary only; no PATH or environment fallback"}


def _ogg_crc(page: bytes) -> int:
    crc = 0
    for b in page:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)) & 0xFFFFFFFF
    return crc


def inspect_opus(path: Path) -> dict:
    """Validate our single-stream Ogg files and inspect RFC 7845/6716 headers."""
    data = path.read_bytes()
    pos, packets, pending, pages = 0, [], bytearray(), []
    serial = None
    while pos < len(data):
        if pos + 27 > len(data) or data[pos:pos+5] != b"OggS\x00":
            raise ValueError("Invalid or truncated Ogg page")
        flags, granule, stream, sequence, checksum, count = struct.unpack_from("<BQIIIB", data, pos+5)
        lacing = data[pos+27:pos+27+count]
        end = pos + 27 + count + sum(lacing)
        if len(lacing) != count or end > len(data):
            raise ValueError("Truncated Ogg payload")
        page = bytearray(data[pos:end])
        page[22:26] = b"\x00" * 4
        if _ogg_crc(page) != checksum:
            raise ValueError("Ogg CRC mismatch")
        if serial is None:
            serial = stream
        if stream != serial or sequence != len(pages) or bool(flags & 1) != bool(pending):
            raise ValueError("Unsupported chained, out-of-order or broken Ogg stream")
        if (not pages and not flags & 2) or (pages and pages[-1]["eos"]):
            raise ValueError("Invalid Ogg stream boundaries")
        cursor = pos + 27 + count
        for length in lacing:
            pending.extend(data[cursor:cursor+length])
            cursor += length
            if length < 255:
                packets.append(bytes(pending))
                pending.clear()
        pages.append({"granule": granule, "eos": bool(flags & 4)})
        pos = end
    if pending or not pages or not pages[-1]["eos"] or len(packets) < 3:
        raise ValueError("Missing Ogg end-of-stream or Opus packets")
    head, tags, *audio = packets
    if len(head) != 19 or head[:8] != b"OpusHead" or tags[:8] != b"OpusTags":
        raise ValueError("Expected mono mapping-family-0 Ogg Opus")
    version, channels, pre_skip, input_sr, gain, mapping = struct.unpack_from("<BBHIhB", head, 8)
    if (version, channels, gain, mapping) != (1, 1, 0, 0):
        raise ValueError("Unexpected Opus identification parameters")
    infos = []
    for packet in audio:
        if not packet:
            raise ValueError("Empty Opus packet")
        config, stereo, code = packet[0] >> 3, (packet[0] >> 2) & 1, packet[0] & 3
        if config < 12:
            mode, bandwidth, duration = "SILK", ["NB", "MB", "WB"][config//4], [10,20,40,60][config%4]
        elif config < 16:
            mode, bandwidth, duration = "HYBRID", ["SWB", "FB"][(config-12)//2], [10,20][config%2]
        else:
            mode, bandwidth, duration = "CELT", ["NB", "WB", "SWB", "FB"][(config-16)//4], [2.5,5,10,20][config%4]
        if code == 3 and len(packet) < 2:
            raise ValueError("Truncated Opus frame count")
        frames = [1,2,2,packet[1] & 63 if len(packet)>1 else 0][code]
        if frames == 0 or frames*duration > 120 or stereo:
            raise ValueError("Invalid or non-mono Opus packet")
        infos.append({"mode": mode, "bandwidth": bandwidth, "duration_ms": frames*duration})
    duration_s = sum(p["duration_ms"] for p in infos) / 1000
    granule = pages[-1]["granule"]
    if granule < pre_skip or granule > round(duration_s*48000):
        raise ValueError("Invalid final Opus granule")
    payload_bytes = sum(map(len, audio))
    coded_samples_48k = round(duration_s*48000)
    return {"format": "Ogg Opus", "ogg_crc_valid": True, "ogg_pages": len(pages),
            "input_sample_rate_header": input_sr, "channels": channels,
            "pre_skip_48k_samples": pre_skip, "pre_skip_ms": pre_skip/48,
            "final_granule_48k": granule,
            "container_pcm_duration_s": (granule-pre_skip)/48000,
            "end_trim_48k_samples": round(duration_s*48000)-granule,
            "audio_packets": len(audio), "packet_duration_ms": sorted({p["duration_ms"] for p in infos}),
            "packet_sizes_bytes": sorted({len(p) for p in audio}),
            "mode_counts": dict(Counter(p["mode"] for p in infos)),
            "bandwidth_counts": dict(Counter(p["bandwidth"] for p in infos)),
            "payload_sha256": hashlib.sha256(b"".join(audio)).hexdigest(),
            "payload_bytes": payload_bytes, "file_bytes": len(data),
            "coded_samples_48k": coded_samples_48k,
            "payload_bitrate_bps": payload_bytes*8*48000/coded_samples_48k,
            "file_bitrate_per_pcm_second_bps": len(data)*8/((granule-pre_skip)/48000),
            "bitrate_basis": "payload rate uses coded packet time including padding; file rate uses retained PCM time"}


def opus_roundtrip(audio: np.ndarray, sr: int, destination: Path, *, bitrate_bps: int = 8000) -> tuple[np.ndarray, int, dict]:
    if bitrate_bps not in {6000,8000,12000,16000}:
        raise ValueError("Unsupported controlled Opus bitrate")
    if sr != 8000:
        raise ValueError("Controlled Opus channel expects already downsampled 8 kHz mono")
    audio = validate_audio(audio, sr)
    source, compressed = destination.with_suffix(".codec-input.wav"), destination.with_suffix(".opus")
    metadata_path = destination.with_suffix(".channel.json")
    for path in [source, compressed, destination, metadata_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite channel artifact: {path}")
    runtime = codec_runtime()
    ffmpeg = runtime["executable"]
    write_audio(source, audio, sr)
    encode = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-n", "-i", str(source),
              "-map", "0:a:0", "-map_metadata", "-1", "-ac", "1", "-ar", "8000", "-c:a", "libopus",
              "-b:a", str(bitrate_bps), "-vbr", "off", "-application", "voip", "-frame_duration", "20",
              "-compression_level", "10", "-packet_loss", "0", "-fec", "0", "-cutoff", "4000"]
    if runtime["dtx_control_exposed"]:
        encode += ["-dtx", "0"]
    encode.append(str(compressed))
    decode = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-n", "-i", str(compressed),
              "-map", "0:a:0", "-ac", "1", "-ar", "8000", "-c:a", "pcm_f32le", str(destination)]
    metadata = {"encoder": "libopus", "runtime_sha256": runtime["executable_sha256"],
                "encode_command": encode, "decode_command": decode,
                "requested": {"bitrate_bps": bitrate_bps, "vbr": "off", "application": "voip", "frame_ms": 20,
                              "complexity": 10, "packet_loss_percent": 0, "fec": False, "cutoff_hz": 4000,
                              "dtx": "off via API default; no CLI option" if not runtime["dtx_control_exposed"] else "explicit off"},
                "input_samples": len(audio), "input_sample_rate": sr,
                "alignment": "FFmpeg Ogg pre-skip and EOS trim; no reference correlation, manual crop or pad"}
    try:
        for name, command in [("encode", encode), ("decode", decode)]:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            metadata[name+"_returncode"] = result.returncode
            metadata[name+"_stderr"] = result.stderr
            if result.returncode:
                raise RuntimeError(f"Opus {name} failed: {result.stderr}")
        inspection = inspect_opus(compressed)
        received, rate = read_audio(destination)
        info = sf.info(destination)
        if rate != 8000 or info.channels != 1 or len(received) != len(audio):
            raise ValueError("Decoded rate/channels/length mismatch; no automatic alignment applied")
        if inspection["input_sample_rate_header"] != 8000 or inspection["packet_duration_ms"] != [20]:
            raise ValueError("Opus header/packet duration differs from protocol")
        if inspection["packet_sizes_bytes"] != [bitrate_bps//400] or not np.isclose(inspection["payload_bitrate_bps"],bitrate_bps,atol=1e-8,rtol=0):
            raise ValueError("Opus payload differs from requested hard CBR")
        if not np.isclose(inspection["container_pcm_duration_s"], len(audio)/sr, atol=1e-10, rtol=0):
            raise ValueError("Opus granule duration differs from input")
        metadata.update({"status": "ok", "decoded_sample_rate": rate, "decoded_samples": len(received),
                         "decoded_subtype": info.subtype, "compressed_path": str(compressed),
                         "compressed_sha256": sha256_file(compressed), "decoded_sha256": sha256_file(destination),
                         "inspection": inspection})
        write_json(metadata_path, metadata)
        return received, rate, metadata
    except Exception as exc:
        metadata.update({"status": "failed", "error": repr(exc)})
        write_json(metadata_path, metadata)
        raise


def apply_channel(audio: np.ndarray, sr: int, name: str, destination: Path,
                  reference: np.ndarray, uid: str, seed: int) -> tuple[np.ndarray, int, dict]:
    if name not in CHANNELS or sr != 16000 or len(audio) != len(reference):
        raise ValueError("Invalid channel, rate or paired reference length")
    if destination.exists() or destination.with_suffix(".channel.json").exists():
        raise FileExistsError(f"Channel output exists: {destination}")
    metadata = {"channel": name, "input_sample_rate": sr, "input_samples": len(audio)}
    if name == "opus8k_8kbps":
        received, rate, codec = opus_roundtrip(resample_audio(audio, sr, 8000), 8000, destination)
        metadata.update(codec)
        metadata.update({"codec_input_sample_rate":8000,"codec_input_samples":metadata["input_samples"]})
    else:
        rate = sr
        received = audio.copy()
        if name.startswith("noise"):
            snr = {"noise20_downsample_8k":20, "noise15_downsample_8k":15}[name]
            noise, stats = paired_noise(reference, uid, seed, snr)
            received += noise
            metadata.update(stats)
            metadata.update({"actual_variant_snr_pre_db": snr_db(audio, noise),
                             "actual_variant_snr_post_db": snr_db(resample_audio(audio,sr,8000),
                                                                   resample_audio(noise,sr,8000)),
                             "reference_snr_excludes_watermark_distortion": True})
        if name != "clean":
            received, rate = resample_audio(received, sr, 8000), 8000
        write_audio(destination, received, rate)
        received, rate = read_audio(destination)
    metadata.update({"input_sample_rate":sr,"input_samples":len(audio),
                     "output_sample_rate": rate, "output_samples": len(received),
                     "output_peak": float(np.max(np.abs(received))), "clipping_applied": False,
                     "waveform_sha256": sha256_file(destination)})
    if metadata["output_peak"] >= 1:
        raise ValueError("Channel exceeded audio range; refusing implicit clipping")
    write_json(destination.with_suffix(".channel.json"), metadata)
    return received, rate, metadata
