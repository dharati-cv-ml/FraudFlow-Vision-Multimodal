"""Canvas analytics graphics; no Matplotlib, browser, or SciPy dependency."""
from collections import Counter
BG='#142337'; PANEL='#1e3046'; TEXT='#eaf3f8'; MUTED='#9db6ca'; TEAL='#39d2c8'; ORANGE='#ff9779'

def render_histogram(cv, risks, threshold):
    cv.delete('all')
    w=max(cv.winfo_width(),550);h=max(cv.winfo_height(),200)
    cv.create_text(16,18,text='Visual alteration risk-score distribution',anchor='w',fill=TEXT,font=('Segoe UI',13,'bold'))
    cv.create_text(16,41,text='Model output (uncalibrated score) | synthetic samples',anchor='w',fill=MUTED,font=('Segoe UI',10))
    n=12;counts=[0]*n
    for p in risks:counts[min(n-1,max(0,int(float(p)*n)))]+=1
    left=40;right=w-32; top=64; bottom=h-36;maxv=max(counts or [1]) or 1
    bw=(right-left)/n
    for j,count in enumerate(counts):
        x1=left+j*bw+4;x2=left+(j+1)*bw-4
        y1=bottom-count/maxv*(bottom-top)
        cv.create_rectangle(x1,y1,x2,bottom,fill=TEAL if (j+.5)/n<threshold else ORANGE,outline='')
        cv.create_text((x1+x2)/2,max(top,y1-10),text=str(count),fill=MUTED,font=('Segoe UI',9))
    tx=left+threshold*(right-left)
    cv.create_line(tx,top,tx,bottom,fill='#ffcb73',width=2,dash=(3,3))
    cv.create_text(min(tx+20,right-50),top+12,text=f'Threshold {threshold:.2f}',fill='#ffcb73',font=('Segoe UI',9))
    cv.create_text(left,bottom+15,text='0.0',fill=MUTED,anchor='w')
    cv.create_text(right,bottom+15,text='1.0',fill=MUTED,anchor='e')


def render_decision_mix(cv, rows):
    cv.delete('all');w=max(cv.winfo_width(),350); h=max(cv.winfo_height(),180)
    normal=sum(r['prediction']=='NO FLAG' for r in rows);flagged=len(rows)-normal
    cv.create_text(15,20,text='Inspection decisions',anchor='w',fill=TEXT,font=('Segoe UI',13,'bold'))
    x1=18;y1=65;width=w-38
    if rows:
        cut=x1+width*normal/len(rows)
        cv.create_rectangle(x1,y1,cut,y1+33,fill=TEAL,outline='')
        cv.create_rectangle(cut,y1,x1+width,y1+33,fill=ORANGE,outline='')
    cv.create_text(18,122,text=f'No flag: {normal} ({normal/max(1,len(rows)):.1%})',anchor='w',fill=TEXT,font=('Segoe UI',11))
    cv.create_text(18,149,text=f'Review: {flagged} ({flagged/max(1,len(rows)):.1%})',anchor='w',fill=ORANGE,font=('Segoe UI',11))


def render_confusion(cv, m):
    cv.delete('all'); w=max(cv.winfo_width(),430)
    cv.create_text(16,19,text='Held-out test confusion matrix',anchor='w',fill=TEXT,font=('Segoe UI',13,'bold'))
    cv.create_text(16,42,text='Rows = actual label  |  Columns = prediction',anchor='w',fill=MUTED,font=('Segoe UI',10))
    left=120;top=80; size=90
    labels=[('TN',m['tn'],TEAL),('FP',m['fp'],ORANGE),('FN',m['fn'],ORANGE),('TP',m['tp'],TEAL)]
    for i,(key,value,color) in enumerate(labels):
        col=i%2;row=i//2
        x=left+col*(size+7);y=top+row*(size+7)
        cv.create_rectangle(x,y,x+size,y+size,fill=PANEL,outline=color,width=2)
        cv.create_text(x+size/2,y+27,text=key,fill=MUTED,font=('Segoe UI',11,'bold'))
        cv.create_text(x+size/2,y+56,text=str(value),fill=TEXT,font=('Segoe UI',24,'bold'))
    cv.create_text(65,top+45,text='Actual 0',fill=MUTED,font=('Segoe UI',10))
    cv.create_text(65,top+45+size+7,text='Actual 1',fill=MUTED,font=('Segoe UI',10))
    cv.create_text(left+size/2,top-17,text='Predict 0',fill=MUTED,font=('Segoe UI',10))
    cv.create_text(left+size+7+size/2,top-17,text='Predict 1',fill=MUTED,font=('Segoe UI',10))


def metric_text(m):
    return f"Precision {m['precision']:.1%}   Recall {m['recall']:.1%}   F1 {m['f1']:.1%}   Accuracy {m['accuracy']:.1%}"
