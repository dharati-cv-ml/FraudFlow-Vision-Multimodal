"""Deterministic visual edit-seam features (no metadata and no ground truth).

This detector is intentionally specialized for visible replacement patches in
synthetic, template-like documents. These features are NOT fraud probabilities.
"""
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

FEATURE_NAMES = ['horizontal_peak', 'horizontal_top5', 'horizontal_above25',
                 'vertical_peak', 'vertical_top5', 'vertical_above16',
                 'horizontal_run', 'vertical_run']

@dataclass
class Evidence:
    features: list[float]
    suspicious_region: tuple[int,int,int,int] | None
    description: str


def _longest_run(matches):
    cur = best = 0
    for hit in matches:
        cur = (cur+1) if hit else 0
        best = max(best,cur)
    return best


def extract(image_or_path):
    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as source:
            image = source.convert('L')
    else:
        image = image_or_path.convert('L')
    original_width, original_height = image.size
    im = image.resize((340,450), Image.Resampling.BILINEAR)
    px=im.load()
    # Paper brightness discontinuity only, excluding dark text / table rules.
    xlo,xhi=60,320; ylo,yhi=66,435
    rows=[]; cols=[]
    for y in range(ylo, yhi):
        flags=[]
        for x in range(xlo,xhi):
            a=px[x,y]; b=px[x,y-1]
            flags.append(a>211 and b>211 and abs(a-b)>=5)
        rows.append((sum(flags),_longest_run(flags),y))
    for x in range(xlo,xhi):
        flags=[]
        for y in range(ylo,yhi):
            a=px[x,y]; b=px[x-1,y]
            flags.append(a>211 and b>211 and abs(a-b)>=5)
        cols.append((sum(flags),_longest_run(flags),x))
    # Native table separators often span the entire page; exclude these from
    # patch seam detection rather than treating them as edits.
    candidate_rows=[row for row in rows if row[0]<195]
    sorted_h=sorted(candidate_rows,reverse=True)
    sorted_v=sorted(cols,reverse=True)
    feats=[sorted_h[0][0]/100, sum(x[0] for x in sorted_h[:5])/500,
           sum(x[0]>=25 for x in candidate_rows)/10,
           sorted_v[0][0]/100, sum(x[0] for x in sorted_v[:5])/500,
           sum(x[0]>=16 for x in cols)/10,
           max(x[1] for x in candidate_rows)/100, max(x[1] for x in cols)/100]
    # Estimate a diagnostic rectangle from two nearly-aligned vertical seams.
    # No label or stored ground-truth edit coordinate is consulted.
    vertical_runs=[]
    for _, _, x in cols:
        cur=best=0; end_y=ylo
        for y in range(ylo,yhi):
            a=px[x,y]; b=px[x-1,y]
            hit=a>211 and b>211 and abs(a-b)>=5
            cur=cur+1 if hit else 0
            if cur>best:best=cur;end_y=y
        vertical_runs.append((best,x,end_y-best+1,end_y))
    largest=max(run[0] for run in vertical_runs)
    region=None
    if largest>=12:
        primary=max(vertical_runs,key=lambda item:item[0])
        nearby=[r for r in vertical_runs if r[0]>=largest-2 and
                abs(r[2]-primary[2])<=5 and abs(r[3]-primary[3])<=5]
        x1=max(0,min(r[1] for r in nearby)-5)
        x2=min(339,max(r[1] for r in nearby)+5)
        y1=max(0,min(r[2] for r in nearby)-4)
        y2=min(449,max(r[3] for r in nearby)+4)
        region=(round(x1*original_width/340),round(y1*original_height/450),
                round(x2*original_width/340),round(y2*original_height/450))
    description=(f'Largest non-divider horizontal paper-edge response: {sorted_h[0][0]} pixels; '
                 f'largest vertical response: {sorted_v[0][0]} pixels. '
                 'Highlight is a diagnostic hot-zone, not proof of manipulation.')
    return Evidence(feats, region, description)
