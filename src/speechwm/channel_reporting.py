from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def make_channel_report(out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    summary = json.loads((out/"metrics.json").read_text())
    validation = json.loads((out/"channel_validation.json").read_text())
    config = yaml.safe_load((out/"config.yaml").read_text())
    channels = config["evaluation"]["channels"]
    metrics = summary["metrics"]
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    names = {"clean":"Clean 16 kHz", "downsample_8k":"Resample 8 kHz",
             "noise20_downsample_8k":"AWGN 20 dB then 8 kHz",
             "noise15_downsample_8k":"AWGN 15 dB then 8 kHz", "opus8k_8kbps":"8 kHz then Opus 8 kb/s"}
    def select(channel,delta=None,frames=None):
        return [r for r in metrics if r["channel"]==channel and r["detector"]=="correlation"
                and (delta is None or r["delta_db"]==delta) and (frames is None or r["frames"]==frames)]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    for channel in channels:
        rows=select(channel,frames=30)
        axes[0].plot([r["delta_db"] for r in rows],[100*r["eer"] for r in rows],"o-",label=names[channel])
        rows=sorted(select(channel,delta=3),key=lambda r:r["frames"])
        axes[1].plot([r["frames"] for r in rows],[100*r["eer"] for r in rows],"o-",label=names[channel])
    axes[0].set(xlabel="Embedding strength (dB)",ylabel="Development EER (%)",title="30 valid frames per group")
    axes[1].set(xlabel="Valid frames per group",ylabel="Development EER (%)",title="Strength 3 dB")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.set_ylim(-1,55)
    fig.suptitle("M2 smoke only: 20 segments / 3 utterances; no held-out test")
    fig.savefig(figures/"channel_eer.png",dpi=160)
    plt.close(fig)
    post=list(csv.DictReader((out/"post_channel_quality.csv").open()))
    coverage=list(csv.DictReader((out/"coverage.csv").open()))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    positions=np.arange(len(channels))
    for delta,shift in [(1,-.18),(3,.18)]:
        means=[np.mean([float(r["pesq_nb"]) for r in post if r["channel"]==c and float(r["delta_db"])==delta
                        and int(r["label"])==1 and r["pesq_nb"]]) for c in channels]
        axes[0].bar(positions+shift,means,width=.36,label=f"Strength {delta} dB")
    for label,shift in [(0,-.18),(1,.18)]:
        means=[np.mean([float(r["voiced_fraction"]) for r in coverage if r["channel"]==c
                        and float(r["delta_db"])==3 and int(r["label"])==label]) for c in channels]
        axes[1].bar(positions+shift,means,width=.36,label=f"Label {label}")
    for ax in axes:
        ax.set_xticks(positions,["Clean","8 kHz","20 dB","15 dB","Opus"])
        ax.legend()
        ax.grid(axis="y",alpha=.2)
    axes[0].set(ylabel="NB PESQ",title="Received audio vs unmarked reference (total distortion)",ylim=(1,4.6))
    axes[1].set(ylabel="Fraction of valid STFT frames",title="Received-file VAD, strength 3 dB",ylim=(0,1.05))
    fig.savefig(figures/"channel_quality_coverage.png",dpi=160)
    plt.close(fig)
    lines=["# M2 噪声与 Opus 通道开发实验", "",
        f"运行：`{summary['experiment_id']}/{summary['run_id']}`；20 个片段、3 条源语音、3 位说话人，全部属于 smoke。",
        "", "本次验证通道实现与记录完整性。源文件是公开 HQ Ogg 示例，尚无正式独立数据划分；EER 不代表论文复现性能，不能用于选择最终参数或部署阈值。", "",
        "## 固定处理协议", "",
        "所有正负类均经过 STFT/ISTFT、浮点 WAV 保存重读，再进入同一通道。噪声以共同增益后的未嵌入语音全段去均值功率为参考，使用源片段 ID 与固定种子派生的 PCG64 随机序列；同源所有强度、正负类共享噪声，同一序列用于 20/15 dB，仅缩放功率。先在 16 kHz 加噪声，再抗混叠降采样到 8 kHz。", "",
        "Opus 在 8 kHz 下真实编解码：单声道、8 kb/s hard CBR、20 ms、VOIP、复杂度 10、FEC 与丢包仿真关闭、截止 4 kHz。FFmpeg 7.1 没有 DTX CLI 参数，沿用 libopus 默认关闭；运行包中检出的版本字符串为 libopus 1.3.1，识别依据及二进制哈希在 codec_runtime.json，未通过运行时 API 查询版本。", "",
        "解码器依据 Ogg pre-skip 与 EOS granule 恢复受控起止时间，不利用原音频互相关对齐，不额外裁剪或补样本。每个文件逐项检查样本数、采样率、CRC、帧长、包大小与容器时长。包标记的 NB 表示编码器模式带宽，不是对模拟频响的测量。未知裁剪与时间伸缩仍不在能力范围内。", "",
        "## 通道验证", "",
        f"- 接收 WAV {validation['received_wav_hashes_verified']} 个，实际 Opus {validation['opus_files_verified']} 个；全部长度、采样率、峰值和哈希检查通过。",
        f"- 强度 0 的 {validation['delta_zero_identical_pcm_pairs']} 对音频解码后采样值逐点相同，{validation['delta_zero_eer_settings']} 组 EER 均为 50%。WAV PEAK 时间戳可能不同，不作为类别线索。",
        f"- Opus 有效载荷码率为 8000 bit/s，每包 20 字节 / 20 ms；pre-skip 为 {validation['opus_pre_skip_ms']} ms。文件码率还包含 Ogg 头与尾部补齐。",
        f"- 所有设置最少 {validation['minimum_valid_frames']} 个有效语音帧；失败记录 {summary['failures']} 项。", "",
        "宽带与脉冲探针的互相关峰残余偏移为 +1 个 8 kHz 样本（0.125 ms），仅作诊断；未据此移动实际语音。探针、奇数长度和静音检查见 probes/verification.json。", "",
        "| 噪声阶段 | 实际参考 SNR 最小 dB | 最大 dB |", "| --- | ---: | ---: |"]
    for name,stats in validation["reference_snr_ranges"].items():
        lines.append(f"| {name} | {stats['min']:.6f} | {stats['max']:.6f} |")
    lines += ["", "降采样会改变保留频带内的噪声和语音功率，因此后级 SNR 不等于前级标称值。这里排除水印失真，使用固定未嵌入参考；每个变体自身的 SNR 另列于 channel_manifest.jsonl。", "",
        "## 开发 EER", "", "以下使用 30 个有效帧平均；单位为 %，不足的尾帧丢弃。分组可以跨越 VAD 排除间隙，因此 30 帧并不总是连续 300 ms。", "",
        "| 强度 dB | Clean | 8 kHz | 20 dB 噪声→8 kHz | 15 dB 噪声→8 kHz | 8 kHz→Opus |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for delta in config["evaluation"]["deltas_db"]:
        values=[f"{select(c,delta,30)[0]['eer']*100:.2f}" for c in channels]
        lines.append(f"| {delta:g} | " + " | ".join(values) + " |")
    lines += ["", "所有分组 EER 以组为单位；metrics.csv 同时提供每片段首组 EER、两类分组数、配对覆盖数，用于观察样本依赖和覆盖变化。不同通道独立 VAD，结果同时反映传输失真与留帧变化。", "",
        "![各通道开发 EER](figures/channel_eer.png)", "",
        "## 接收音质与覆盖率", "",
        "post_channel_quality.csv 的 NB/WB PESQ 比较接收音频与采样率匹配的未嵌入参考，反映嵌入与通道的总失真。WB 在 8 kHz 通道上通过升采样计算，不能表述为恢复了宽带。通道前嵌入音质单独保存在 quality_scores.csv，二者不混用。", "",
        "![接收音质与 VAD 覆盖](figures/channel_quality_coverage.png)", "",
        "## 复查入口", "",
        "- config.yaml、environment.json、codec_runtime.json：配置与实际运行环境。",
        "- channel_manifest.jsonl、audio/*.channel.json：种子、SNR、执行命令、stderr、压缩文件哈希、时间与码率。",
        "- audio/*.opus、*.codec-input.wav、接收 WAV：真实编解码证据。",
        "- frame_scores.csv、aggregate_scores.csv、metrics.csv、threshold_curves.csv：原始分数与描述性阈值曲线。",
        "- coverage.csv、failures.jsonl、channel_validation.json：覆盖率、失败与整轮自动审计。", "",
        "## 音频示例", ""]
    first=json.loads((out/"input_manifest.jsonl").read_text().splitlines()[0])["utterance_id"]
    for channel in channels:
        lines.append(f"- [强度 3 dB，{names[channel]}](audio/{first}_d3_label1_{channel}.wav)")
    lines += ["", "后续 M3/M4 可以复用本次固定通道和特征协议。正式主实验须先落实合法语料、说话人隔离划分与验证集参数选择；本阶段未训练 LPC 或神经检测器。", ""]
    path=out/"M2通道实验报告.md"
    path.write_text("\n".join(lines))
    return path
