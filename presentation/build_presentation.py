"""Build the seven-slide IEEE CloudNet demo PDF and preview images.

Run with Python containing reportlab, Pillow and pymupdf.
Dashboard PNGs are captured separately with capture_demo.py.
"""
from pathlib import Path
import math
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
import fitz

HERE = Path(__file__).resolve().parent
A = HERE / 'assets'
W, H = 1600, 900
NAVY, BLUE, TEAL = '#10263D', '#0065A8', '#007C78'
MUTED, BG, RED = '#526578', '#F3F6FA', '#B53145'
for name, file in [('Arial', 'Arial.ttf'), ('Bold', 'Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, '/System/Library/Fonts/Supplemental/'+file))
PDF = HERE / 'CLOUDNET2026_Demo_Gyftodimos.pdf'
c = canvas.Canvas(str(PDF), pagesize=(W,H))
c.setTitle('PhySDV | IEEE CloudNet 2026 | Konstantinos Gyftodimos')
c.setAuthor('Konstantinos Gyftodimos | Harokopio University of Athens')

def rect(x,y,w,h,color,r=0):
    c.setFillColor(HexColor(color))
    if r: c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=0)
    else: c.rect(x,H-y-h,w,h,fill=1,stroke=0)

def text(x,y,s,size=24,color=NAVY,bold=False):
    c.setFillColor(HexColor(color)); c.setFont('Bold' if bold else 'Arial',size)
    c.drawString(x,H-y-size*.85,s)

def para(x,y,s,w,size=24,color=NAVY,bold=False,leading=None):
    p=Paragraph(s,ParagraphStyle('p',fontName='Bold' if bold else 'Arial',fontSize=size,
                leading=leading or size*1.32,textColor=HexColor(color)))
    _,h=p.wrap(w,H); p.drawOn(c,x,H-y-h)
    return h

def img(path,x,y,w,h):
    im=Image.open(path).convert('RGBA')
    iw,ih=im.size; scale=min(w/iw,h/ih)
    c.drawImage(ImageReader(im),x+(w-iw*scale)/2,H-y-h+(h-ih*scale)/2,
                width=iw*scale,height=ih*scale,mask='auto')

def crop(name,box,out):
    p=A/out; Image.open(A/name).crop(box).save(p); return p

def arrow(x1,y1,x2,y2,color=BLUE):
    c.setStrokeColor(HexColor(color)); c.setFillColor(HexColor(color)); c.setLineWidth(2.4)
    c.line(x1,H-y1,x2,H-y2)
    ang=math.atan2(y2-y1,x2-x1)
    p=c.beginPath(); p.moveTo(x2,H-y2)
    for a in (ang+2.65,ang-2.65): p.lineTo(x2+12*math.cos(a),H-(y2+12*math.sin(a)))
    p.close(); c.drawPath(p,fill=1,stroke=0)

def badge(x,y,n,color=BLUE):
    c.setFillColor(HexColor(color)); c.circle(x,H-y,16,stroke=0,fill=1)
    text(x-5,y-10,str(n),20,'#FFFFFF',True)

def base(n,title,sub):
    rect(0,0,W,H,BG); rect(0,0,W,105,'#FFFFFF')
    img(A/'hua.png',55,12,200,80)
    text(290,27,'PhySDV',26,BLUE,True)
    text(290,61,'QoS-aware perception offloading',18,MUTED)
    img(A/'cloudnet.webp',1170,17,220,70)
    rect(1420,24,125,54,BLUE,6); img(A/'ieee.png',1432,31,100,38)
    text(58,127,title,40,NAVY,True); text(60,183,sub,22,MUTED)
    rect(58,831,1484,1,'#D3DEE8')
    text(60,848,'Konstantinos Gyftodimos  |  PhD candidate  |  Harokopio University of Athens (HUA)',18,MUTED)
    text(1240,848,f'IEEE CloudNet 2026   /   {n:02d}',18,BLUE,True)

def card(x,y,w,h,title,body,color=BLUE):
    rect(x,y,w,h,'#FFFFFF',14); rect(x,y,6,h,color,3)
    text(x+26,y+25,title,27,color,True)
    para(x+26,y+76,body,w-52,24)

def end(): c.showPage()

# 1: substantive opening, no title-only introduction.
base(1,'Where should a vehicle run its perception AI?',
     'The same camera task can run in the car, at a nearby edge server, or in the cloud.')
para(60,243,'A moving vehicle must keep detecting road users while its connection changes.',675,32,bold=True)
para(60,378,'A small on-board model responds quickly. Larger remote models offer better detection accuracy, but add network delay.',665,27)
rect(60,522,670,223,NAVY,14)
text(88,549,'The decision is repeated for every frame',28,'#FFFFFF',True)
para(88,605,'Balance response time, model accuracy and server load. Keep a local fallback when remote execution fails.',610,26,'#FFFFFF')
for y,letter,title,desc,col in [
    (245,'A / B','IN THE VEHICLE','YOLOv8n • NPU / GPU • 8 / 5 ms inference',BLUE),
    (417,'C','NEARBY MEC SERVER','YOLOv8m • 6 ms inference + network',TEAL),
    (589,'D','CLOUD DATA CENTER','YOLOv8l • 4 ms inference + network','#7956A6')]:
    rect(800,y,740,145,'#FFFFFF',14); text(827,y+26,letter,35,col,True)
    text(955,y+25,title,27,col,True); text(955,y+81,desc,21,MUTED)
text(804,762,'Demo values are configured simulation inputs.',18,MUTED)
end()

base(2,'From radio physics to a live placement decision',
     'PhySDV connects synthetic data generation, learned network prediction and an explainable policy.')
for x,num,title,body in [
    (60,'01','GENERATE','50,000 labeled samples<br/>Five road environments<br/>Radio + traffic + server load'),
    (560,'02','PREDICT','Is the MEC reachable?<br/>What network delay is expected?<br/>How busy are the servers?'),
    (1060,'03','SELECT','Update the four execution tiers<br/>Check the QoS limits<br/>Choose a tier and explain why')]:
    rect(x,260,480,310,'#FFFFFF',14); text(x+26,286,num,44,BLUE,True)
    text(x+26,350,title,27,BLUE,True); para(x+26,408,body,425,25)
arrow(541,418,557,418); arrow(1041,418,1057,418)
rect(60,610,1480,151,NAVY,14)
text(87,634,'City center  →  City outskirts  →  Suburban  →  Highway  →  Rural',29,'#FFFFFF',True)
para(88,687,'3GPP-calibrated path loss, shadowing, shared bandwidth and queue delay create changing network conditions.',1390,24,'#FFFFFF')
text(62,790,'The live dashboard simulates the placement loop; it does not stream camera inference to real servers.',19,MUTED)
end()

base(3,'Three limits, then an edge-first choice',
     'A tier must pass all feasibility checks before the policy compares it with other eligible tiers.')
for x,value,label,col in [(60,'≤ 100 ms','End-to-end latency',BLUE),(565,'≤ 15%','Accuracy loss',TEAL),(1070,'< 95%','Server utilization','#7956A6')]:
    rect(x,247,470,155,'#FFFFFF',14); text(x+25,269,value,52,col,True); text(x+25,342,label,26,MUTED)
card(60,440,720,167,'1  Prefer the nearby edge',
     'If both remote tiers pass, choose MEC when the cloud gains less than 5 mAP points. Here the gain is 2.7.',TEAL)
card(820,440,720,167,'2  Use cloud when the edge fails',
     'Cloud is selected only while it still meets latency, accuracy and capacity limits.','#7956A6')
card(60,636,720,162,'3  Fall back to the on-board GPU',
     'If no tier passes, choose B: 5 ms inference, but 20.8% accuracy loss. This is a degraded decision.',RED)
card(820,636,720,162,'Optional risk and stability controls',
     'q90 gating adds a cautious delay estimate; dwell filtering reduces repeated switches. Defaults: mean gating, dwell off.',BLUE)
end()

base(4,'What the paper validates',
     'Results below come from the repository experiments on generated data, rather than road trials.')
card(60,245,465,340,'PHYSICS',
     '<b>Within 1.6%</b><br/><br/>Configured physical relationships can be recovered from the generated dataset.<br/><br/>Checks internal consistency of the generator.',BLUE)
card(555,245,485,340,'PREDICTION',
     '<b>6.12 ms MEC delay error</b><br/><br/>The compact HistGB predictor has 1.27 MB model size and 3.79 ms median prediction time.<br/><br/>Held-out synthetic samples.',TEAL)
card(1070,245,470,340,'PLACEMENT',
     '<b>94.0% → 96.6%</b><br/><br/>Latency compliance improves with q90 gating in the urban-sparse scenario.<br/><br/>EMA-smoothed policy, 1,000 frames, 10 Hz.','#7956A6')
rect(60,624,1480,136,NAVY,14)
text(88,647,'What the audience should take away',28,'#FFFFFF',True)
para(88,698,'The framework makes the trade-offs visible and repeatable, including the moments when full QoS cannot be met.',1390,27,'#FFFFFF')
text(61,789,'Sources: new_paper/main.tex; experiments/results_exp2b.json; experiments/results_exp4.json.',18,MUTED)
end()

base(5,'Live demo: change the network and watch the decision',
     'Run: python demo.py --outage-at 0     |     Default scenario: suburban     |     Press q to finish')
for x,key,title,body,col in [
    (60,'START','MEC / C','Normal connection: the edge gives sufficient accuracy with lower network cost.',TEAL),
    (562,'o','CLOUD / D','Cut only the MEC link. Cloud takes over if it remains within every budget.','#7956A6'),
    (1064,'x','LOCAL GPU / B','Cut the remote options. The local model keeps running with an accuracy-budget breach.',RED)]:
    rect(x,245,476,240,'#FFFFFF',14); text(x+25,268,key,37,col,True)
    text(x+25,326,title,28,col,True); para(x+25,379,body,425,24)
normal=crop('dashboard-blackout.png',(0,16+7*24,1748,16+14*24),'blackout-decision.png')
img(normal,60,518,1480,157)
text(60,697,'Faults last about 3 seconds at the current update rate, then the policy re-evaluates recovery.',24,MUTED)
text(60,750,'1–5: road scenario    space: pause    + / −: speed    c: cloud congestion    r: reset    k: kiosk',23,BLUE,True)
text(60,792,'Blackout effects build through delay smoothing; a return to MEC depends on current conditions.',18,MUTED)
end()

# 6: genuine upper dashboard capture, with numbered arrow targets.
base(6,'Dashboard walkthrough • radio, decision and tier comparison',
     'Presenter reference 1/2 — an actual rendered dashboard state after 20 simulated frames.')
upper=crop('dashboard-normal.png',(0,0,1748,16+24*24),'dashboard-upper.png')
img(upper,60,260,1050,356)
sx,sy,scale=60,260,1050/1748
for num,px,py,cy in [(1,1620,45,259),(2,760,140,360),(3,500,267,461),(4,1555,497,563)]:
    tx=sx+px*scale; ty=sy+py*scale
    arrow(1131,cy+16,tx,ty); badge(tx,ty,num)
for y,n,title,body in [
    (241,1,'Header','Road profile, radio parameters, Hz and frame count. PAUSED, AUTO and q90 appear when enabled.'),
    (351,2,'Radio link','Diamond = vehicle. Distance = MEC separation; RSSI / bars = signal strength. Reachability and red fault labels show link status.'),
    (461,3,'Selected tier','Large letter = destination. Latency, mAP and loss describe that tier. “why” explains it; “flag” is the reason code.'),
    (571,4,'Execution table','A/B/C/D compare candidates. The marker identifies the active tier; status shows eligibility or the first failed check.')]:
    badge(1156,y+15,n); text(1185,y,title,23,BLUE,True); para(1140,y+35,body,402,18,leading=22)
rect(60,710,1480,102,'#FFFFFF',12)
text(84,724,'Every table column',23,BLUE,True)
para(84,756,'<b>Inf</b> = inference ms · <b>Net</b> = smoothed predicted network ms · <b>Total</b> = Inf + Net · <b>mAP</b> = detection score (higher is better).<br/><b>Loss</b> = average relative class-AP drop from YOLOv8l · <b>Cap</b> = predicted utilization · <b>—</b> = unavailable · bar marker = 100 ms.',1420,19)
end()

base(7,'Dashboard walkthrough • budgets, session and controls',
     'Presenter reference 2/2 — a captured blackout state, showing the cost of the on-board fallback.')
lower=crop('dashboard-blackout.png',(0,16+24*24,1748,920),'dashboard-lower.png')
img(lower,60,253,1480,271)
# Targets in the displayed lower crop. Callouts below keep the screenshot readable.
for n,tx,ty,bx,by in [(1,365,300,230,565),(2,1197,311,680,565),(3,540,442,1110,565),(4,1318,503,1430,693)]:
    arrow(bx,by,tx,ty); badge(tx,ty,n)
for x,n,title,body in [
    (60,1,'Selected-tier gauges','Latency, loss and utilization against their limits. Red means exceeded; the light marker shows the threshold. Here loss is 20.8% > 15%.'),
    (560,2,'Session statistics','Uptime and frame count; average chosen-tier latency; tier switches; deadline misses; MEC-reachable frame share. Tier dwell = percentage of frames in each tier.'),
    (1060,3,'Timeline + compliance','Timeline runs oldest → newest. Color = tier; shaded background = injected fault. Compliance counts latency / accuracy breaches; capacity is checked separately by the policy.')]:
    rect(x,572,480,159,'#FFFFFF',10); badge(x+24,591,n)
    text(x+50,579,title,23,BLUE,True); para(x+18,618,body,443,19)
rect(60,747,1480,65,NAVY,10)
text(82,760,'4  Footer',21,'#FFFFFF',True)
para(225,755,'1–5 = dense city / outskirts / suburban / highway / rural · o = MEC outage · x = blackout · c = cloud spike<br/>space = pause · +/− = Hz · k = automatic scenario/fault cycle · r = reset counters/history · q = quit. Rotating text = tips.',1288,18,'#FFFFFF')
end()
c.save()

doc=fitz.open(PDF)
assert len(doc)==7
for i,page in enumerate(doc):
    page.get_pixmap(matrix=fitz.Matrix(.8,.8)).save(HERE/f'slide-{i+1:02d}.png')
thumbs=[]
for i in range(7):
    im=Image.open(HERE/f'slide-{i+1:02d}.png'); im.thumbnail((640,360)); thumbs.append(im)
contact=Image.new('RGB',(1280,4*360),'#D3DEE8')
for i,im in enumerate(thumbs): contact.paste(im,((i%2)*640,(i//2)*360))
contact.save(HERE/'overview.png')
print(PDF)
