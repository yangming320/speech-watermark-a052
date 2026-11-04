from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .evaluation import write_csv


def make_lpc_report(out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary=json.loads((out/"metrics.json").read_text());selection=summary["quality_selection"]
    validation=json.loads((out/"validation.json").read_text());lpc=json.loads((out/"lpc_summary.json").read_text())
    quality=list(csv.DictReader((out/"quality_scores.csv").open()))
    metrics=summary["metrics"]
    deltas=sorted({float(r["delta_db"]) for r in quality});methods=["SS","SS_LPC"]
    def qmean(method,delta,field="pesq_nb"):
        values=[float(r[field]) for r in quality if r["method"]==method and float(r["delta_db"])==delta and r[field]]
        return float(np.mean(values)) if values else np.nan
    def metric(method,delta,channel,frames):
        return next(r for r in metrics if (r["method"],r["delta_db"],r["channel"],r["frames"])==(method,delta,channel,frames))
    channels=list(dict.fromkeys(r["channel"] for r in metrics))
    frames=sorted({r["frames"] for r in metrics})
    # Measured points only; line segments guide the eye and never generate table values.
    figures=out/"figures";figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,field in zip(axes,["pesq_nb","pesq_wb"]):
        for method in methods:ax.plot(deltas,[qmean(method,d,field) for d in deltas],"o-",label=method)
        ax.set(xlabel="Nominal embedding strength (dB)",ylabel=field.upper().replace("_"," "),title="Measured pre-channel quality")
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle("M3 smoke: 20 segments / 3 utterances; alpha=0.35, LPC order=16")
    fig.savefig(figures/"quality_vs_strength.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout="constrained")
    for row,aggregate_frames in enumerate([1,30]):
        for col,channel in enumerate(["noise20_downsample_8k","noise15_downsample_8k"]):
            ax=axes[row,col]
            for method in methods:
                ax.plot([100*metric(method,d,channel,aggregate_frames)["eer"] for d in deltas if d>0],
                        [qmean(method,d) for d in deltas if d>0],"o-",label=method)
            ax.set(xlabel="Development EER (%)",ylabel="Pre-channel NB PESQ",title=f"{channel.split('_')[0]}, {aggregate_frames} valid frame(s)")
            ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Quality / robustness trade-off; measured global strengths, smoke only")
    fig.savefig(figures/"quality_vs_eer.png",dpi=160);plt.close(fig)
    with np.load(out/"lpc_distributions.npz") as data:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        axes[0].hist(data["raw_db"],bins=80,density=True,color="tab:gray")
        axes[0].set(title=f"Raw LPC log power: SD={lpc['raw_std_db']:.2f} dB",xlabel="dB",ylabel="Density")
        axes[1].hist(data["centered_db"],bins=80,density=True,alpha=.65,label="All valid LPC frames")
        x=np.linspace(-45,45,300);axes[1].plot(x,norm.pdf(x,scale=13.5),label="Fixed N(0, 13.5^2)")
        axes[1].set(title=f"Per-frame band centered: SD={lpc['centered_std_db']:.2f} dB",xlabel="dB",ylabel="Density");axes[1].legend(fontsize=8)
        fig.suptitle("Empirical smoke distributions; Gaussian scale is not fitted")
        fig.savefig(figures/"lpc_distributions.png",dpi=160);plt.close(fig)
    with np.load(out/"diagnostic.npz") as data:
        frame=int(np.argmax(data["frame_power"]))
        f=np.arange(data["reference_spectrum_db"].shape[0])*50
        fig,axes=plt.subplots(3,1,figsize=(9,9),layout="constrained")
        raw=data["envelope_db"][:,frame]
        # Offset envelope for shape display only; raw values remain saved unshifted.
        offset=float(data["reference_spectrum_db"][:,frame].mean()-raw.mean())
        axes[0].plot(f,data["reference_spectrum_db"][:,frame],label="STFT log magnitude")
        axes[0].plot(f,raw+offset,label="LPC envelope (display offset only)")
        axes[0].set(xlim=(0,4000),ylabel="dB",title=f"First utterance, maximum-energy analysis frame {frame}");axes[0].legend()
        freq=data["frequency_hz"]
        axes[1].plot(freq,data["centered_band_db"][:,frame],label="Centered LPC log power")
        axes[1].set(xlabel="Frequency (Hz)",ylabel="Centered dB");axes[1].grid(alpha=.2)
        right=axes[1].twinx();right.plot(freq,data["weights"][:,frame],color="tab:orange",label="Weight gamma")
        right.set(ylabel="Weight",ylim=(0,1.05))
        axes[1].legend(loc="upper left");right.legend(loc="upper right")
        axes[2].plot(freq,data["target_ss_db"][:,frame],label="SS target")
        axes[2].plot(freq,data["target_lpc_db"][:,frame],label="LPC target")
        axes[2].plot(freq,data["actual_lpc_db"][:,frame],alpha=.6,label="LPC after ISTFT / WAV reload")
        axes[2].set(xlabel="Frequency (Hz)",ylabel="Spectral change (dB)",title="Strength 3 dB; reconstructed change differs from target")
        axes[2].legend();axes[2].grid(alpha=.2)
        fig.savefig(figures/"lpc_mechanism.png",dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
        for ax,field,title in [(axes[0,0],"target_ss_db","SS designed"),(axes[0,1],"target_lpc_db","LPC designed"),
                               (axes[1,0],"actual_ss_db","SS reconstructed WAV"),(axes[1,1],"actual_lpc_db","LPC reconstructed WAV")]:
            im=ax.imshow(data[field],origin="lower",aspect="auto",extent=[data["times_s"][0],data["times_s"][-1],500,3650],cmap="RdBu_r",vmin=-5,vmax=5)
            ax.set(title=title,xlabel="Time (s)",ylabel="Frequency (Hz)");fig.colorbar(im,ax=ax,label="dB")
        fig.suptitle("Strength 3 dB: target and actual perturbations of the first segment")
        fig.savefig(figures/"spectral_changes.png",dpi=160);plt.close(fig)
    same=[];matched=[]
    for delta in deltas:
        for channel in channels:
            for count in frames:
                a=metric("SS",delta,channel,count);b=metric("SS_LPC",delta,channel,count)
                same.append({"delta_db":delta,"channel":channel,"frames":count,"ss_eer":a["eer"],"lpc_eer":b["eer"],
                    "lpc_minus_ss_eer":b["eer"]-a["eer"],"ss_nb_pesq":qmean("SS",delta),"lpc_nb_pesq":qmean("SS_LPC",delta)})
    for channel in channels:
        for count in frames:
            a=metric("SS",selection["ss_delta_db"],channel,count);b=metric("SS_LPC",selection["lpc_delta_db"],channel,count)
            matched.append({"channel":channel,"frames":count,"ss_delta_db":selection["ss_delta_db"],"lpc_delta_db":selection["lpc_delta_db"],
                "ss_nb_pesq":selection["ss_mean_nb_pesq"],"lpc_nb_pesq":selection["lpc_mean_nb_pesq"],
                "ss_eer":a["eer"],"lpc_eer":b["eer"],"lpc_minus_ss_eer":b["eer"]-a["eer"],
                "ss_negative_groups":a["negative_groups"],"ss_positive_groups":a["positive_groups"],
                "lpc_negative_groups":b["negative_groups"],"lpc_positive_groups":b["positive_groups"],
                "ss_paired_segments":a["paired_eligible_segments"],"lpc_paired_segments":b["paired_eligible_segments"]})
    write_csv(out/"same_strength_comparison.csv",same);write_csv(out/"matched_quality_comparison.csv",matched)
    per_source=[]
    for uid in sorted({r["utterance_id"] for r in quality}):
        a=next(r for r in quality if r["utterance_id"]==uid and r["method"]=="SS" and float(r["delta_db"])==selection["ss_delta_db"])
        b=next(r for r in quality if r["utterance_id"]==uid and r["method"]=="SS_LPC" and float(r["delta_db"])==selection["lpc_delta_db"])
        per_source.append({"utterance_id":uid,"ss_nb_pesq":float(a["pesq_nb"]),"lpc_nb_pesq":float(b["pesq_nb"]),"lpc_minus_ss_nb_pesq":float(b["pesq_nb"])-float(a["pesq_nb"])})
    write_csv(out/"matched_quality_per_segment.csv",per_source)
    differences=[r["lpc_minus_ss_nb_pesq"] for r in per_source]
    lines=["# M3 LPC 加权与传统扩频开发对照", "",
        f"运行 `{summary['experiment_id']}/{summary['run_id']}`；20 个切片、3 条公开 HQ Ogg 源语音、3 位说话人，全部属于 smoke。没有正式独立测试、性别分组或部署阈值；本轮是 E02/E03 的开发版。", "",
        "## 实现与公平对照", "",
        "LPC 固定 16 阶、自相关法、相对对角正则 1e-6；20 ms 周期 Hann、10 ms 帧移，与 M1/M2 对齐。每帧去直流、加窗，无预加重。以残差方差/|A|² 获取 LPC 功率包络，取 500～3600 Hz 的 63 点逐帧减去频带均值，代入固定 N(0,13.5²) 的生存函数，再取 0.35 次方。强度在对数谱中乘以权重，原相位和频带外设计谱保持原值。", "",
        "LPC 阶数、正则、中心化和窗处理是本项目的固定实现选择；论文未完整说明这些细项。图 3 的指数未单独明确，本项目两类实验统一用方案的 alpha=0.35。图 4 主音质轴固定为通道前嵌入 PESQ，不混入噪声造成的总音质下降。", "",
        f"扫描高至 8 dB 后，{summary['joint_gain_attenuated_segments']} 个片段需要额外衰减。每个源片段的两种方法、全部强度和正负类共享同一增益，目标峰值 0.90；所有噪声也依据该共同参考计算，并跨方法/标签/强度共享。未削波、未对某个方法单独归一化。", "",
        "## 数值与记录验证", "",
        f"- {validation['received_wav_hashes_and_redetections_verified']} 个接收 WAV 重读、哈希、独立重算相关分数与 VAD 掩码通过。",
        f"- {validation['lpc_frames_recomputed']} 个 LPC 帧从原始准备音频重新计算；低能量零权重 {validation['lpc_low_energy_frames']} 帧，数值失败 0。",
        f"- {validation['delta_zero_identical_pcm_pairs']} 对零强度 PCM 相同；{validation['delta_zero_eer_settings']} 个零强度 EER 均为 50%。",
        f"- {validation['shared_noise_groups']} 组噪声在方法/标签/强度间共享；{validation['eer_settings_recomputed_from_csv']} 条 EER 从保存 CSV 重算一致；最低有效帧 {validation['minimum_valid_frames']}。",
        f"- 中心化包络经验标准差 {lpc['centered_std_db']:.3f} dB，参考语音 VAD 区域 {lpc['voiced_centered_std_db']:.3f} dB；权重仍固定用 13.5，没有在 smoke 上拟合替换。", "",
        "## 相同强度的通道前音质", "",
        "| 强度 dB | SS NB | LPC NB | SS WB | LPC WB |", "| ---: | ---: | ---: | ---: | ---: |"]
    for d in deltas:lines.append(f"| {d:g} | {qmean('SS',d):.3f} | {qmean('SS_LPC',d):.3f} | {qmean('SS',d,'pesq_wb'):.3f} | {qmean('SS_LPC',d,'pesq_wb'):.3f} |")
    lines += ["", "同强度 EER 全部保存在 same_strength_comparison.csv；较弱的有效嵌入会同时改善音质并改变鲁棒性，不能只凭同强度的音质增益推断鲁棒性增益。下面列出同为 3 dB 的实际检测结果：", "",
        "| 通道 | 聚合有效帧 | SS EER % | LPC EER % |", "| --- | ---: | ---: | ---: |"]
    for channel in ["noise20_downsample_8k","noise15_downsample_8k"]:
        for count in [1,30]:
            lines.append(f"| {channel} | {count} | {metric('SS',3,channel,count)['eer']*100:.2f} | {metric('SS_LPC',3,channel,count)['eer']*100:.2f} |")
    lines += ["", "当前同强度噪声对照中 LPC EER 更高，反映有效嵌入较弱；后面的接近音质比较才用于观察音质/鲁棒性权衡。", "",
        "![同强度音质曲线](figures/quality_vs_strength.png)", "",
        "## 接近音质的鲁棒性", "",
        f"先完成全局音质扫描，再冻结选择：SS 强度 {selection['ss_delta_db']:g} dB，LPC 强度 {selection['lpc_delta_db']:g} dB，平均 NB PESQ 分别 {selection['ss_mean_nb_pesq']:.6f}/{selection['lpc_mean_nb_pesq']:.6f}，均值差 {selection['absolute_mean_difference']:.6f} ≤ 0.02。这是实际测量点，无插值、无逐片段选强度，选择时尚未计算通道 EER。", "",
        f"片段级 LPC−SS 的 NB PESQ 差范围 {min(differences):.4f}～{max(differences):.4f}；平均匹配不表示每条语音都等音质。详细分布在 matched_quality_per_segment.csv。", "",
        "| 通道 | 聚合有效帧 | SS EER % | LPC EER % | LPC−SS 百分点 |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in matched:lines.append(f"| {r['channel']} | {r['frames']} | {r['ss_eer']*100:.2f} | {r['lpc_eer']*100:.2f} | {r['lpc_minus_ss_eer']*100:+.2f} |")
    lines += ["", "负差表示本批样本上 LPC 的 EER 更低，正差则更高；两者都保留。主鲁棒性口径是单帧，多帧是开发补充。每条接收音频独立 VAD，因此留帧/组数会变化，metrics.csv 同时提供首组 EER 与配对覆盖数。组内有效帧可跨 VAD 间隙，不能把 30 帧直接称为连续 300 ms。", "",
        "![音质与鲁棒性权衡](figures/quality_vs_eer.png)", "",
        "## 机制与实际音频诊断", "",
        "包络越高，权重越低；LPC 曲线的显示偏移仅用于与 STFT 形状对照，实际用于 CDF 的是保存的逐帧中心化频带值。设计频谱到 ISTFT、WAV 重读再分析会有时频一致性变化，实际扰动和频带泄漏不能用设计值代替。", "",
        "![LPC 包络与权重](figures/lpc_mechanism.png)", "",
        "![LPC 数据分布](figures/lpc_distributions.png)", "",
        "![目标与重建后的实际扰动](figures/spectral_changes.png)", "",
        "## 历史回归与边界", "",
        f"M1/M2 冻结证据哈希均通过。M3 与 M2 有 {validation['historical_ss_comparison']['overlapping_m2_settings']} 个重合 SS 设置，{validation['historical_ss_comparison']['changed_settings']} 项 EER 有变化，最大绝对差 {validation['historical_ss_comparison']['max_absolute_eer_difference']*100:.4f} 个百分点；高强度引入的共同增益会影响 VAD，不能要求跨阶段的 EER 逐项相等。SS 的默认公式和原有测试保留，方法结论以 M3 内共同条件为准。", "",
        "当前样本既用于音质匹配也用于开发评估，这只验证对照流程；正式强度选择仍须验证集，最终结果须独立测试集。未按 EER 选 LPC 阶数/指数、未重拟合 13.5、未声称复现论文数值或性别差异。正式语料仍待落实。", "",
        "## 可复查文件", "",
        "config.yaml、input_manifest.jsonl、environment.json、key.json；lpc/*.npz/json、lpc_frames.csv、lpc_summary.json；quality_scores.csv、quality_selection.json、common_gains.csv；embedding_manifest.jsonl、audio_manifest.jsonl、channel_manifest.jsonl；frame_scores.csv、aggregate_scores.csv、metrics.csv、threshold_curves.csv；coverage.csv、failures.jsonl、validation.json、run_status.json。", "",
        "## 音频示例", ""]
    uid=json.loads((out/"input_manifest.jsonl").read_text().splitlines()[0])["utterance_id"]
    for method,d in dict.fromkeys([("SS",3),("SS_LPC",3),("SS",selection["ss_delta_db"]),("SS_LPC",selection["lpc_delta_db"])]):
        lines.append(f"- [{method} 强度 {d:g}，通道前](audio/{uid}_{method}_d{d:g}_label1_embed.wav)")
    for method,d in [("SS",selection["ss_delta_db"]),("SS_LPC",selection["lpc_delta_db"])]:
        lines.append(f"- [{method} 匹配强度 {d:g}，20 dB 噪声后降采样](audio/{uid}_{method}_d{d:g}_label1_noise20_downsample_8k.wav)")
    lines.append("")
    path=out/"M3对照实验报告.md";path.write_text("\n".join(lines));return path
