"""
ULTRON Sentinel AI — FastAPI Backend (Ultimate Edition)
NEW: Video upload, analysis pipeline, downloadable JSON/CSV/video report
All existing live features preserved.
"""
import cv2, asyncio, base64, json, time, threading, logging, os, uuid, csv
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

import config
from modules.detector_tracker    import DetectorTracker
from modules.motion_analyzer     import MotionAnalyzer
from modules.anomaly_engine      import AnomalyEngine
from modules.gathering_detector  import GatheringDetector
from modules.alert_manager       import (AlertManager, set_crowd_limit,
                                         get_crowd_limit, set_email_cfg, set_sms_cfg)
from modules.aggression_detector import AggressionDetector
from modules.stampede_detector   import StampedeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("ULTRON")

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ULTRON Sentinel AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = {"frame":None,"heatmap":None,"flow":None,"stats":{},"alerts":[],"running":False,"fps":0,"recording":False}
heatmap_acc = None
det = motion = anomaly_eng = gathering = alerts_mgr = aggression_det = None
stampede_det = None
cap = None; is_running = False; writer = None; recording = False
clients: set = set()
video_jobs: dict = {}


def camera_thread():
    global det,motion,anomaly_eng,gathering,alerts_mgr,aggression_det,stampede_det
    global cap,is_running,writer,recording,heatmap_acc,state
    import numpy as np
    logger.info("ULTRON — Initialising modules...")
    det=DetectorTracker(); anomaly_eng=AnomalyEngine(); gathering=GatheringDetector()
    alerts_mgr=AlertManager(); aggression_det=AggressionDetector(); stampede_det=StampedeDetector()
    src=int(config.VIDEO_SOURCE) if config.VIDEO_SOURCE.isdigit() else config.VIDEO_SOURCE
    cap=cv2.VideoCapture(src)
    if not cap.isOpened(): logger.error(f"Cannot open: {config.VIDEO_SOURCE}"); is_running=False; return
    fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    motion=MotionAnalyzer(fw,fh,fps); heatmap_acc=np.zeros((fh,fw),dtype=np.float64)
    t_fps=time.time(); fps_cnt=0; show_fps=0.0; frame_n=0
    while is_running:
        ok,frame=cap.read()
        if not ok: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        frame_n+=1
        if frame_n%config.SKIP_FRAMES!=0: continue
        tracked=det.process(frame); feats=motion.analyze(tracked,det.position_history)
        anom=anomaly_eng.update(feats["feature_vector"]); aggr=aggression_det.analyze(feats["person_metrics"])
        stamp=stampede_det.analyze(feats,feats["person_metrics"])
        groups=gathering.find_gatherings(tracked); energy=gathering.energy_anomaly(feats["person_metrics"])
        zones=gathering.count_zones(tracked,config.ZONES); viol=gathering.social_distance(tracked)
        new_alerts=alerts_mgr.evaluate(feats,anom,groups,energy,zones,viol,aggression_flags=aggr,stampede=stamp)
        heatmap_acc*=config.HEATMAP_DECAY; r=config.HEATMAP_RADIUS; sigma=r/2.5
        for o in tracked:
            cx,cy=o["center"]; y0,y1=max(0,cy-r),min(fh,cy+r); x0,x1=max(0,cx-r),min(fw,cx+r)
            if y1>y0 and x1>x0:
                yy,xx=np.meshgrid(np.arange(y0,y1),np.arange(x0,x1),indexing="ij")
                heatmap_acc[y0:y1,x0:x1]+=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma**2))
        hm=heatmap_acc.copy(); mx2=hm.max()
        hm=(hm/mx2*255).astype(np.uint8) if mx2>0 else np.zeros_like(hm,dtype=np.uint8)
        hm_color=cv2.applyColorMap(hm,cv2.COLORMAP_JET)
        heatmap_out=cv2.addWeighted(frame,1-config.HEATMAP_ALPHA,hm_color,config.HEATMAP_ALPHA,0)
        flow_out=frame.copy()
        for fv in feats["flow_vectors"]:
            cx,cy=fv["center"]; dx=int(fv["dx"]*config.FLOW_ARROW_SCALE); dy=int(fv["dy"]*config.FLOW_ARROW_SCALE)
            if abs(dx)+abs(dy)>4: cv2.arrowedLine(flow_out,(cx,cy),(cx+dx,cy+dy),(0,255,255),2,tipLength=0.35)
        out=frame.copy(); is_anom=anom.get("confirmed",False); aggr_ids=set(aggr.get("track_ids",[]))
        for o in tracked:
            x1,y1,x2,y2=o["bbox"]; tid=o["track_id"]; pm=feats["person_metrics"].get(tid,{}); spd=pm.get("speed",0)
            color=(0,0,255) if tid in aggr_ids else (0,100,255) if is_anom else (255,255,255)
            cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
            cl=12
            for bx,by,sx,sy in[(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out,(bx+sx*cl,by),(bx,by),color,2); cv2.line(out,(bx,by+sy*cl),(bx,by),color,2)
            lbl=f"ID:{tid} {spd:.0f}"
            if tid in aggr_ids: lbl+=" AGGR"
            elif is_anom: lbl+=" ANOM"
            cv2.putText(out,lbl,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.38,color,1)
        for grp in groups:
            xs=[o["center"][0] for o in grp]; ys=[o["center"][1] for o in grp]
            cx2=int(sum(xs)/len(xs)); cy2=int(sum(ys)/len(ys))
            r2=int(max(max(xs)-min(xs),max(ys)-min(ys))//2+30)
            cv2.circle(out,(cx2,cy2),r2,(0,165,255),2)
        for zname,zpoly in config.ZONES.items():
            zdata=zones.get(zname,{}); zcol={"RED":(0,0,255),"AMBER":(0,165,255),"GREEN":(0,255,0)}.get(zdata.get("status","GREEN"),(0,255,0))
            cv2.polylines(out,[zpoly],True,zcol,2)
        limit=get_crowd_limit(); bar_pct=min(len(tracked)/max(limit,1),1.0)
        bar_c=(0,0,255) if len(tracked)>limit else (0,255,0)
        cv2.rectangle(out,(fw-22,10),(fw-10,fh-10),(60,60,60),-1)
        filled_h=int((fh-20)*bar_pct); cv2.rectangle(out,(fw-22,fh-10-filled_h),(fw-10,fh-10),bar_c,-1)
        cv2.putText(out,f"ULTRON {len(tracked)}/{limit} FPS:{show_fps:.0f}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        st_t="STAMPEDE" if stamp.get("detected") else("AGGRESSION" if aggr["detected"] else("ANOMALY" if is_anom else "NORMAL"))
        st_c=(0,0,255) if(aggr["detected"] or is_anom) else(0,255,0)
        cv2.putText(out,f"STATUS:{st_t}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.45,st_c,1)
        if recording and writer: writer.write(out)
        def enc(img): _,buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,72]); return base64.b64encode(buf).decode()
        fps_cnt+=1
        if time.time()-t_fps>=1.0: show_fps=fps_cnt/(time.time()-t_fps); fps_cnt=0; t_fps=time.time()
        state.update({"frame":enc(out),"heatmap":enc(heatmap_out),"flow":enc(flow_out),
            "fps":round(show_fps,1),"recording":recording,
            "stats":{"total_count":len(tracked),"crowd_limit":get_crowd_limit(),
                "over_limit":len(tracked)>get_crowd_limit(),"fps":round(show_fps,1),
                "avg_speed":round(feats["avg_speed"],1),"max_speed":round(feats["max_speed"],1),
                "flow_coherence":round(feats["flow_coherence"],3),"max_density":round(feats["max_density"],2),
                "anomaly_status":anom["status"],"anomaly_score":round(anom["anomaly_score"],4),
                "anomaly_confirmed":anom["confirmed"],"consec_anom_frames":anom["consec_frames"],
                "warmup_progress":anom.get("warmup_progress",0),"stampede_detected":stamp.get("detected",False),"stampede_risk":stamp.get("risk_level","NONE"),"stampede_signals":stamp.get("signals",{}),"aggression_detected":aggr["detected"],
                "aggression_count":aggr["count"],"aggression_ids":aggr["track_ids"],
                "gatherings":len(groups),"energy_ratio":energy["ratio"],"energy_abnormal":energy["is_abnormal"],
                "social_violations":len(viol),"zones":zones,"is_anomaly":is_anom}})
        if new_alerts: state["alerts"]=(new_alerts+state.get("alerts",[]))[:100]
        time.sleep(0.01)
    if cap: cap.release()
    if writer: writer.release()


def analyze_video_job(job_id: str, video_path: str):
    import numpy as np
    job=video_jobs[job_id]; job["status"]="processing"; job["progress"]=0
    try:
        cap2=cv2.VideoCapture(video_path)
        if not cap2.isOpened(): job["status"]="error"; job["error"]="Cannot open video"; return
        total_frames=int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v=cap2.get(cv2.CAP_PROP_FPS) or 30.0
        fw=int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)); fh=int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        det_v=DetectorTracker(); mot_v=MotionAnalyzer(fw,fh,fps_v)
        anom_v=AnomalyEngine(); gath_v=GatheringDetector(); aggr_v=AggressionDetector(); alm_v=AlertManager()
        frame_results=[]; peak_count=0; total_sum=0; all_speeds=[]
        aggression_events=0; anomaly_events=0; alert_summary=[]
        out_video_path=str(REPORT_DIR/f"{job_id}_annotated.mp4")
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer_v=cv2.VideoWriter(out_video_path,fourcc,fps_v,(fw,fh))
        frame_n=0
        while True:
            ok,frame=cap2.read()
            if not ok: break
            frame_n+=1
            if frame_n%2!=0: continue
            tracked=det_v.process(frame); feats=mot_v.analyze(tracked,det_v.position_history)
            anom=anom_v.update(feats["feature_vector"]); aggr=aggr_v.analyze(feats["person_metrics"])
            groups=gath_v.find_gatherings(tracked); energy=gath_v.energy_anomaly(feats["person_metrics"])
            zones=gath_v.count_zones(tracked,config.ZONES); viol=gath_v.social_distance(tracked)
            new_alts=alm_v.evaluate(feats,anom,groups,energy,zones,viol,aggression_flags=aggr)
            cnt=len(tracked); peak_count=max(peak_count,cnt); total_sum+=cnt
            if feats["avg_speed"]>0: all_speeds.append(feats["avg_speed"])
            if aggr["detected"]: aggression_events+=1
            if anom.get("confirmed"): anomaly_events+=1
            ts=round(frame_n/fps_v,2)
            frame_results.append({"frame":frame_n,"timestamp_sec":ts,"person_count":cnt,
                "avg_speed":round(feats["avg_speed"],2),"max_speed":round(feats["max_speed"],2),
                "flow_coherence":round(feats["flow_coherence"],3),"max_density":round(feats["max_density"],3),
                "gatherings":len(groups),"social_violations":len(viol),
                "aggression_detected":aggr["detected"],"aggression_count":aggr["count"],
                "anomaly_status":anom["status"],"anomaly_score":round(anom["anomaly_score"],4),
                "energy_ratio":round(energy["ratio"],3)})
            for alt in new_alts: alert_summary.append({**alt,"frame":frame_n,"timestamp_sec":ts})
            out_f=frame.copy(); is_anom=anom.get("confirmed",False); aggr_ids=set(aggr.get("track_ids",[]))
            for o in tracked:
                x1,y1,x2,y2=o["bbox"]; tid=o["track_id"]
                color=(0,0,255) if tid in aggr_ids else(0,140,255) if is_anom else(180,255,180)
                cv2.rectangle(out_f,(x1,y1),(x2,y2),color,2)
                cv2.putText(out_f,f"ID:{tid}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.38,color,1)
            st_t="AGGRESSION" if aggr["detected"] else("ANOMALY" if is_anom else "NORMAL")
            st_c=(0,0,255) if(aggr["detected"] or is_anom) else(0,220,0)
            cv2.putText(out_f,f"T:{ts:.1f}s  People:{cnt}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
            cv2.putText(out_f,f"Status:{st_t}",(10,42),cv2.FONT_HERSHEY_SIMPLEX,0.5,st_c,1)
            writer_v.write(out_f)
            if total_frames>0: job["progress"]=min(int(frame_n/total_frames*100),99)
        cap2.release(); writer_v.release()
        avg_count=round(total_sum/max(len(frame_results),1),1)
        avg_speed=round(sum(all_speeds)/max(len(all_speeds),1),1)
        report={"job_id":job_id,
            "video_info":{"width":fw,"height":fh,"fps":round(fps_v,1),"total_frames":total_frames,"duration_sec":round(total_frames/fps_v,1)},
            "summary":{"peak_crowd_count":peak_count,"avg_crowd_count":avg_count,"avg_speed_px_s":avg_speed,
                "total_aggression_events":aggression_events,"total_anomaly_events":anomaly_events,
                "total_alerts":len(alert_summary),"crowd_limit":get_crowd_limit(),"limit_exceeded":peak_count>get_crowd_limit()},
            "alerts":alert_summary,"frame_data":frame_results}
        json_path=str(REPORT_DIR/f"{job_id}_report.json")
        with open(json_path,"w") as f: json.dump(report,f,indent=2)
        csv_path=str(REPORT_DIR/f"{job_id}_report.csv")
        with open(csv_path,"w",newline="") as f:
            if frame_results:
                w2=csv.DictWriter(f,fieldnames=frame_results[0].keys()); w2.writeheader(); w2.writerows(frame_results)
        job["status"]="done"; job["progress"]=100
        job["result"]={"summary":report["summary"],"video_info":report["video_info"],
            "total_alerts":len(alert_summary),"alerts_preview":alert_summary[:10],
            "frame_count":len(frame_results),
            "downloads":{"json":f"/analysis/download/{job_id}/json","csv":f"/analysis/download/{job_id}/csv","video":f"/analysis/download/{job_id}/video"}}
        logger.info(f"Analysis job {job_id} complete")
    except Exception as e:
        logger.error(f"Job {job_id} error: {e}"); job["status"]="error"; job["error"]=str(e)
    finally:
        try: os.remove(video_path)
        except: pass


@app.get("/")
def root(): return FileResponse("../frontend/index.html")

@app.post("/start")
def start():
    global is_running
    if is_running: return {"status":"already_running"}
    is_running=True; state["running"]=True
    threading.Thread(target=camera_thread,daemon=True).start(); return {"status":"started"}

@app.post("/stop")
def stop():
    global is_running; is_running=False; state["running"]=False; return {"status":"stopped"}

@app.get("/status")
def status_ep(): return {"running":is_running,**state.get("stats",{})}

@app.get("/alerts")
def get_alerts(): return {"alerts":state.get("alerts",[])}

@app.post("/alerts/clear")
def clear_alerts(): state["alerts"]=[]; return {"status":"cleared"}

@app.post("/analysis/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed={".mp4",".avi",".mov",".mkv",".webm"}
    ext=Path(file.filename).suffix.lower()
    if ext not in allowed: return JSONResponse(status_code=400,content={"error":f"Use: {allowed}"})
    job_id=str(uuid.uuid4())[:8]; save_path=str(UPLOAD_DIR/f"{job_id}{ext}")
    content=await file.read()
    with open(save_path,"wb") as f: f.write(content)
    video_jobs[job_id]={"status":"queued","progress":0,"filename":file.filename,"result":None,"error":None,"created_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    background_tasks.add_task(analyze_video_job,job_id,save_path)
    return {"job_id":job_id,"status":"queued","filename":file.filename}

@app.get("/analysis/status/{job_id}")
def analysis_status(job_id: str):
    if job_id not in video_jobs: return JSONResponse(status_code=404,content={"error":"Job not found"})
    return video_jobs[job_id]

@app.get("/analysis/jobs")
def list_jobs(): return {"jobs":[{"job_id":k,**{kk:vv for kk,vv in v.items() if kk!="result"}} for k,v in video_jobs.items()]}

@app.get("/analysis/download/{job_id}/{format}")
def download_report(job_id: str, format: str):
    if job_id not in video_jobs or video_jobs[job_id]["status"]!="done":
        return JSONResponse(status_code=404,content={"error":"Report not ready"})
    paths={"json":(REPORT_DIR/f"{job_id}_report.json","application/json",f"ultron_report_{job_id}.json"),
           "csv":(REPORT_DIR/f"{job_id}_report.csv","text/csv",f"ultron_report_{job_id}.csv"),
           "video":(REPORT_DIR/f"{job_id}_annotated.mp4","video/mp4",f"ultron_annotated_{job_id}.mp4")}
    if format not in paths: return JSONResponse(status_code=400,content={"error":"Invalid format"})
    path,media_type,fname=paths[format]
    if not path.exists(): return JSONResponse(status_code=404,content={"error":"File not found"})
    return FileResponse(str(path),media_type=media_type,filename=fname)

class Settings(BaseModel):
    crowd_limit:Optional[int]=None; telegram_enabled:Optional[bool]=None
    telegram_token:Optional[str]=None; telegram_chat_id:Optional[str]=None
    email_enabled:Optional[bool]=None; email_host:Optional[str]=None; email_port:Optional[str]=None
    email_from:Optional[str]=None; email_to:Optional[str]=None; email_pass:Optional[str]=None
    sms_enabled:Optional[bool]=None; twilio_sid:Optional[str]=None; twilio_token:Optional[str]=None
    twilio_from:Optional[str]=None; twilio_to:Optional[str]=None
    aggression_speed_thresh:Optional[int]=None; anomaly_confirm_frames:Optional[int]=None; skip_frames:Optional[int]=None

@app.get("/settings")
def get_settings():
    return {"crowd_limit":get_crowd_limit(),"telegram_enabled":config.TELEGRAM_ENABLED,
            "telegram_token":config.TELEGRAM_TOKEN,"telegram_chat_id":config.TELEGRAM_CHAT_ID,
            "anomaly_confirm_frames":config.ANOMALY_CONFIRM_FRAMES,"aggression_speed_thresh":config.AGGRESSION_SPEED_THRESH}

@app.post("/settings")
def save_settings(s:Settings):
    if s.crowd_limit is not None: set_crowd_limit(s.crowd_limit)
    if s.telegram_enabled is not None: config.TELEGRAM_ENABLED=s.telegram_enabled
    if s.telegram_token: config.TELEGRAM_TOKEN=s.telegram_token
    if s.telegram_chat_id: config.TELEGRAM_CHAT_ID=s.telegram_chat_id
    if s.aggression_speed_thresh: config.AGGRESSION_SPEED_THRESH=s.aggression_speed_thresh
    if s.anomaly_confirm_frames: config.ANOMALY_CONFIRM_FRAMES=s.anomaly_confirm_frames
    if s.skip_frames: config.SKIP_FRAMES=s.skip_frames
    eu={}
    if s.email_enabled is not None: eu["enabled"]=s.email_enabled
    if s.email_host: eu["host"]=s.email_host
    if s.email_port: eu["port"]=s.email_port
    if s.email_from: eu["from"]=s.email_from
    if s.email_to: eu["to"]=s.email_to
    if s.email_pass: eu["pass"]=s.email_pass
    if eu: set_email_cfg(eu)
    su={}
    if s.sms_enabled is not None: su["enabled"]=s.sms_enabled
    if s.twilio_sid: su["sid"]=s.twilio_sid
    if s.twilio_token: su["token"]=s.twilio_token
    if s.twilio_from: su["from"]=s.twilio_from
    if s.twilio_to: su["to"]=s.twilio_to
    if su: set_sms_cfg(su)
    return {"status":"saved","crowd_limit":get_crowd_limit()}

@app.post("/recording/start")
def rec_start():
    global writer,recording
    if not is_running: return {"error":"not running"}
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"); fn=f"recording_{int(time.time())}.mp4"
    writer=cv2.VideoWriter(fn,fourcc,20.0,(640,480)); recording=True; return {"status":"started","file":fn}

@app.post("/recording/stop")
def rec_stop():
    global writer,recording; recording=False
    if writer: writer.release(); writer=None
    return {"status":"stopped"}

@app.websocket("/ws")
async def ws_endpoint(websocket:WebSocket):
    await websocket.accept(); clients.add(websocket)
    try:
        while True:
            if state.get("frame"):
                await websocket.send_text(json.dumps({"frame":state["frame"],"heatmap":state["heatmap"],"flow":state["flow"],"stats":state["stats"],"alerts":state["alerts"][:10],"fps":state["fps"]}))
            await asyncio.sleep(0.04)
    except WebSocketDisconnect: clients.discard(websocket)
    except: clients.discard(websocket)

app.mount("/static",StaticFiles(directory="../frontend"),name="static")

if __name__=="__main__":
    print("="*55); print("  ULTRON SENTINEL AI — ULTIMATE EDITION")
    print(f"  Crowd limit: {get_crowd_limit()}"); print("  http://localhost:8000"); print("="*55)
    uvicorn.run(app,host=config.HOST,port=config.PORT,log_level="warning")
