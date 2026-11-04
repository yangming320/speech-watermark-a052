from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _load_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def make_report(out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary=json.loads((out/"metrics.json").read_text())
    metrics=summary["metrics"]
    quality=_load_csv(out/"quality_scores.csv")
    scores=_load_csv(out/"aggregate_scores.csv")
    figures=out/"figures"
    figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,channel in zip(axes,["clean","downsample_8k"]):
        for frames in [1,10,30]:
            rows=sorted([r for r in metrics if r["channel"]==channel and r["frames"]==frames
                         and r["detector"]=="correlation"],key=lambda r:r["delta_db"])
            ax.plot([r["delta_db"] for r in rows],[100*r["eer"] for r in rows],"o-",label=f"{frames} frames")
        ax.set(title=channel,xlabel="Embedding strength (dB)",ylabel="Development EER (%)",ylim=(-1,55))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Development smoke only: 20 segments from 3 source utterances")
    fig.savefig(figures/"eer_vs_strength.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
    deltas=sorted({float(r["delta_db"]) for r in quality})
    for field,label in [("pesq_nb","NB PESQ"),("pesq_wb","WB PESQ")]:
        values=[]
        for delta in deltas:
            found=[float(r[field]) for r in quality if float(r["delta_db"])==delta and r[field]]
            values.append(np.mean(found) if found else np.nan)
        ax.plot(deltas,values,"o-",label=label)
    ax.set(xlabel="Embedding strength (dB)",ylabel="Pre-channel embedding quality",title="Quality of reconstructed WAV audio")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(figures/"quality_vs_strength.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,channel in zip(axes,["clean","downsample_8k"]):
        for label,name in [(0,"No watermark"),(1,"Watermarked")]:
            found=[float(r["score"]) for r in scores if float(r["delta_db"])==1 and r["channel"]==channel
                   and int(r["frames"])==30 and int(r["label"])==label and r["detector"]=="correlation"]
            ax.hist(found,bins=18,alpha=.55,density=True,label=name)
        ax.set(title=channel,xlabel="Mean correlation score, 30 frames",ylabel="Density")
        ax.legend()
    fig.suptitle("Strength 1 dB; development distributions")
    fig.savefig(figures/"score_distributions.png",dpi=160)
    plt.close(fig)
    if (out/"diagnostic.npz").exists():
        data=np.load(out/"diagnostic.npz")
        sr=int(data["sr"])
        fig,axes=plt.subplots(3,1,figsize=(9,8),layout="constrained")
        count=min(len(data["reference"]),sr)
        times=np.arange(count)/sr
        axes[0].plot(times,data["reference"][:count],label="Reference",linewidth=.8)
        axes[0].plot(times,data["positive"][:count],label="Watermarked",linewidth=.6,alpha=.7)
        axes[0].set(xlabel="Time (s)",ylabel="Amplitude",title="First segment, strength 1 dB")
        axes[0].legend()
        extent=[float(data["times_s"][0]),float(data["times_s"][-1]),500,3650]
        for ax,field,title in [(axes[1],"target_band_delta_db","Designed log-magnitude change"),
                                (axes[2],"actual_delta_db","Measured change after ISTFT and WAV reload")]:
            values=data[field] if field=="target_band_delta_db" else data[field][data["band_indices"]]
            image=ax.imshow(values,origin="lower",aspect="auto",extent=extent,cmap="RdBu_r",vmin=-1.5,vmax=1.5)
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=title)
            fig.colorbar(image,ax=ax,label="dB")
        fig.savefig(figures/"spectral_diagnostic.png",dpi=160)
        plt.close(fig)
    lines=["# 传统扩频水印基线开发验证记录","",
           f"运行编号：`{summary['run_id']}`。本次使用 {summary['segments']} 个片段，来自 {summary['parents']} 条源语音、{summary['speakers']} 位说话人。",
           "", "这些是公开 HQ Ogg 朗读示例的开发切片，全部属于 smoke 分区；没有独立训练、验证或测试划分。下面的 EER 只说明当前样本上的分数分离情况，不能视为论文复现性能或泛化结果。", "",
           "处理链路为：嵌入、ISTFT、浮点 WAV 保存、重新读取、可选抗混叠降采样、再次保存读取、WebRTC VAD、完整频谱倒谱滤波与相关检测。正负类使用相同处理路径。", "",
           f"运行用时 {summary['elapsed_s']:.2f} 秒，不含绘图；失败或有效帧不足记录 {summary['failures']} 项。", "",
           "## 开发结果", "", "下表使用 30 个有效帧平均，EER 以百分比显示；PESQ 为通道前嵌入音质，NB 在 8 kHz、WB 在 16 kHz 计算。", "",
           "| 强度 dB | 无通道 EER % | 降采样 EER % | 平均 NB PESQ | 平均 WB PESQ |",
           "| --- | --- | --- | --- | --- |"]
    for delta in deltas:
        eers=[]
        for channel in ["clean","downsample_8k"]:
            found=[r["eer"]*100 for r in metrics if r["delta_db"]==delta and r["channel"]==channel
                   and r["frames"]==30 and r["detector"]=="correlation"]
            eers.append(f"{found[0]:.2f}" if found else "无有效结果")
        q=[]
        for field in ["pesq_nb","pesq_wb"]:
            found=[float(r[field]) for r in quality if float(r["delta_db"])==delta and r[field]]
            q.append(f"{np.mean(found):.3f}" if found else "不可用")
        lines.append(f"| {delta:g} | {eers[0]} | {eers[1]} | {q[0]} | {q[1]} |")
    lines.extend(["", "强度为 0 的正负类经历相同处理，EER 应为 50%，用来检查评估路径。所有 EER 阈值均为本次样本的描述性插值结果，未校准部署阈值。", "",
                  "降采样条件在某些强度下的 EER 低于无通道条件，当前仅有三条源语音，且接收频谱与 VAD 覆盖存在差异，不能据此推断降采样通常有助于水印检测。", "",
                  "## 图表", "", "![开发 EER 对嵌入强度](figures/eer_vs_strength.png)", "",
                  "![嵌入音质](figures/quality_vs_strength.png)", "",
                  "![分数分布](figures/score_distributions.png)", "",
                  "![实际频谱扰动](figures/spectral_diagnostic.png)", "",
                  "## 可复查文件", "",
                  "- `e00_checks.json`：重建、频率与指标检查。",
                  "- `config.yaml`、`environment.json`、`input_manifest.jsonl`：配置、环境和来源。",
                  "- `frame_scores.csv`、`aggregate_scores.csv`：帧级分数、有效帧及聚合时间。",
                  "- `metrics.csv`、`threshold_curves.csv`：全部强度、通道和聚合设置的结果。",
                  "- `quality_scores.csv`、`coverage.csv`、`failures.jsonl`：音质、VAD 覆盖与失败。",
                  "- `audio_manifest.jsonl`、`audio/`：保存重读的接收音频与哈希。", "",
                  "## 当前边界", "",
                  "本轮未实现 LPC、深度检测、Opus 主实验、固定部署阈值或云端部署。错误密钥分数作为诊断保存，不要求有限样本上严格为随机水平。",
                  "", "后续先落实正式语料和说话人划分，再进入 M2 的通道与评估协议；当前样本不能用于选择最终参数或声称达到论文数值。", ""])
    import yaml
    config=yaml.safe_load((out/"config.yaml").read_text())
    first=json.loads((out/"input_manifest.jsonl").read_text().splitlines()[0])["utterance_id"]
    lines.extend(["## 音频示例", "", "以下文件均为实际生成的浮点 WAV，可直接试听。", "",
                  f"- [同处理路径的无水印音频](audio/{first}_d1_label0_clean.wav)"])
    for delta in config["evaluation"].get("save_audio_deltas",[1,3]):
        lines.append(f"- [强度 {delta:g} dB 的音频](audio/{first}_d{delta:g}_label1_clean.wav)")
    lines.extend([f"- [强度 1 dB 且降采样至 8 kHz](audio/{first}_d1_label1_downsample_8k.wav)", ""])
    (out/"基线验证报告.md").write_text("\n".join(lines))
    return out/"基线验证报告.md"

