from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description="Speech watermark reproduction, M0-M6")
    parser.add_argument("--root",type=Path,default=Path.cwd())
    sub=parser.add_subparsers(dest="command",required=True)
    probe=sub.add_parser("probe",help="Record dependency availability without changing packages")
    probe.add_argument("--output",type=Path,default=Path("docs/environment.json"))
    prepare=sub.add_parser("prepare",help="Fetch hash-verified public development speech")
    prepare.add_argument("--sources",type=Path,default=Path("configs/public_sources.json"))
    prepare.add_argument("--segments",type=int,default=20)
    formal=sub.add_parser("prepare-formal",help="Freeze official public-corpus selection and speaker-disjoint roles")
    formal.add_argument("--config",type=Path,default=Path("configs/formal_data.yaml"))
    for name in ["smoke","evaluate"]:
        evaluate=sub.add_parser(name,help="Run E00 and the public-data E01 baseline")
        evaluate.add_argument("--config",type=Path,default=Path("configs/baseline.yaml"))
        evaluate.add_argument("--run-id")
    report=sub.add_parser("report",help="Regenerate figures/report from saved scores")
    report.add_argument("--run-dir",type=Path,required=True)
    validate=sub.add_parser("validate",help="Audit saved controlled-channel outputs and regenerate report")
    validate.add_argument("--run-dir",type=Path,required=True)
    channel=sub.add_parser("channel",help="Apply a controlled channel to an independent WAV")
    channel.add_argument("input",type=Path)
    channel.add_argument("output",type=Path)
    channel.add_argument("--name",required=True)
    channel.add_argument("--reference",type=Path,help="Unmarked paired reference; required for noise")
    channel.add_argument("--uid",default="independent")
    channel.add_argument("--seed",type=int,default=20261017)
    embed_parser=sub.add_parser("embed",help="Embed SS watermark into an input audio file")
    embed_parser.add_argument("input",type=Path)
    embed_parser.add_argument("output",type=Path)
    embed_parser.add_argument("--delta",type=float,default=1)
    embed_parser.add_argument("--method",choices=["ss","lpc"],default="ss")
    embed_parser.add_argument("--key",type=Path,default=Path("data/keys/watermark_20261017.json"))
    detect=sub.add_parser("detect",help="Score an independent received audio file; no original required")
    detect.add_argument("input",type=Path)
    detect.add_argument("--frames",type=int,default=30)
    detect.add_argument("--checkpoint",type=Path,help="Strict deep checkpoint inference; no correlation fallback")
    detect.add_argument("--key",type=Path,default=Path("data/keys/watermark_20261017.json"))
    args=parser.parse_args()
    root=args.root.resolve()
    os.environ.setdefault("MPLCONFIGDIR",str(root/".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME",str(root/".cache"))
    (root/".cache/fontconfig").mkdir(parents=True,exist_ok=True)
    if args.command=="probe":
        from .environment import probe_environment
        result=probe_environment(root/args.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=="prepare":
        from .data import prepare_public
        result=prepare_public(root,root/args.sources,args.segments)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=="prepare-formal":
        from .formal_data import prepare_formal
        print(prepare_formal(root,root/args.config))
    elif args.command in {"smoke","evaluate"}:
        import yaml
        kind=yaml.safe_load((root/args.config).read_text()).get("experiment_kind")
        if kind=="extension_v1":
            from .m6_experiment import run_extension
            result=run_extension(root,root/args.config,args.run_id)
        elif kind=="formal_main_v1":
            from .main_runner import run_main
            result=run_main(root,root/args.config,args.run_id)
        elif kind=="deep_detection_v1":
            from .deep_runner import run_deep
            result=run_deep(root,root/args.config,args.run_id)
        elif kind=="lpc_comparison_v1":
            from .comparison import run_comparison
            result=run_comparison(root,root/args.config,args.run_id)
        else:
            from .evaluation import run_baseline
            result=run_baseline(root,root/args.config,args.run_id)
        print(result)
    elif args.command=="validate":
        from .utils import write_json
        out=root/args.run_dir
        kind=json.loads((out/"metrics.json").read_text()).get("experiment_kind")
        lpc=kind=="lpc_comparison_v1";deep=kind=="deep_detection_v1";formal=kind=="formal_main_v1"
        if kind=="extension_v1":
            from .m6_validation import validate_extension as validate_run
            from .m6_reporting import make_extension_report as make_report
            formal=True
        elif formal:
            from .main_validation import validate_main as validate_run
            from .main_reporting import make_main_report as make_report
        elif deep:
            from .deep_validation import validate_deep as validate_run
            from .deep_reporting import make_deep_report as make_report
        elif lpc:
            from .lpc_validation import validate_comparison as validate_run
            from .lpc_reporting import make_lpc_report as make_report
        else:
            from .channel_validation import validate_run
            from .channel_reporting import make_channel_report as make_report
        try:
            result=validate_run(out)
        except Exception as exc:
            write_json(out/"run_status.json",{"status":"failed_validation","error":repr(exc)})
            raise
        write_json(out/("validation.json" if lpc or deep or formal else "channel_validation.json"),result)
        make_report(out)
        write_json(out/"run_status.json",{"status":"complete","validation_passed":True} if lpc or deep or formal else {"status":"complete","channel_validation_passed":True})
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=="report":
        out=root/args.run_dir
        summary=json.loads((out/"metrics.json").read_text())
        if summary.get("experiment_kind")=="extension_v1":
            from .m6_reporting import make_extension_report
            print(make_extension_report(out))
        elif summary.get("experiment_kind")=="formal_main_v1":
            from .main_reporting import make_main_report
            print(make_main_report(out))
        elif summary.get("experiment_kind")=="deep_detection_v1":
            from .deep_reporting import make_deep_report
            print(make_deep_report(out))
        elif summary.get("experiment_kind")=="lpc_comparison_v1":
            from .lpc_reporting import make_lpc_report
            print(make_lpc_report(out))
        elif summary.get("channel_protocol"):
            from .channel_reporting import make_channel_report
            print(make_channel_report(out))
        else:
            from .reporting import make_report
            print(make_report(out))
    elif args.command=="channel":
        from .audio import read_audio
        from .channels import apply_channel
        audio,sr=read_audio(root/args.input)
        if args.name.startswith("noise") and args.reference is None:
            raise ValueError("Noise channel requires --reference (unmarked paired source)")
        reference,ref_sr=read_audio(root/args.reference if args.reference else root/args.input)
        if ref_sr!=sr:
            raise ValueError("Reference and input sample rates differ")
        destination=root/args.output
        destination.parent.mkdir(parents=True,exist_ok=True)
        _,_,metadata=apply_channel(audio,sr,args.name,destination,reference,args.uid,args.seed)
        print(json.dumps(metadata,ensure_ascii=False,indent=2))
    elif args.command=="embed":
        from .audio import prepare_audio,read_audio,write_audio
        from .embedding import embed,embed_lpc
        from .keys import load_key
        audio,sr=read_audio(root/args.input)
        audio,prep=prepare_audio(audio,sr)
        embedder=embed_lpc if args.method=="lpc" else embed
        result,metadata=embedder(audio,16000,load_key(root/args.key),args.delta)
        if abs(result).max()>=1:
            raise ValueError("Output would exceed audio range; reduce embedding strength")
        destination=root/args.output
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing audio: {destination}")
        write_audio(destination,result,16000)
        print(json.dumps({**metadata,"preparation":prep,"output":str(destination)},indent=2))
    elif args.command=="detect":
        from .audio import read_audio,resample_audio
        from .features import aggregate,extract_features,score_frames
        from .keys import load_key
        audio,sr=read_audio(root/args.input)
        if args.checkpoint:
            from .models import infer_audio
            scores,feature,checkpoint=infer_audio(root/args.checkpoint,audio,sr,load_key(root/args.key))
            if args.frames not in checkpoint["contract"]["aggregate_frames"]:raise ValueError("Unsupported deep aggregation frame count")
            groups=aggregate(scores,feature.times_s[feature.valid_mask],args.frames)
            print(json.dumps({"input":str(args.input),"sample_rate":sr,"architecture":checkpoint["architecture"],
                "epoch":checkpoint["epoch"],"condition_assumption":checkpoint["contract"]["condition"],
                "contract_hash":checkpoint["contract_hash"],**feature.metadata,"aggregation_frames":args.frames,
                "groups":groups,"status":"scores_only" if groups else "insufficient_voiced_frames",
                "score_type":"linear logits, not calibrated probabilities","decision":None,
                "reason":"No deployment threshold enabled for arbitrary uploads; use the checkpoint-bound evaluation protocol"},indent=2))
            return
        if sr not in {8000,16000}:
            audio=resample_audio(audio,sr,16000)
            sr=16000
        feature=extract_features(audio,sr,load_key(root/args.key))
        valid=feature.valid_mask
        scores=score_frames(feature.features)[valid]
        groups=aggregate(scores,feature.times_s[valid],args.frames)
        print(json.dumps({"input":str(args.input),"sample_rate":sr,**feature.metadata,
                          "aggregation_frames":args.frames,"groups":groups,
                          "status":"scores_only" if groups else "insufficient_voiced_frames",
                          "decision":None,"reason":"No held-out deployment threshold calibrated"},indent=2))


if __name__=="__main__":
    main()

