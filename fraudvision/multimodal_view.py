"""Desktop Multimodal Comparison — view measured saved OCR/VLM results only.

No VLM inference, model download, network call or ground-truth leakage occurs
in this screen. The ground-truth key is reconstructed for evaluation *after*
reading independently generated predictions.
"""
from __future__ import annotations
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from multimodal.visual_compare import (
    load_report, discover_reports, assemble, export_rows, REPORT_DIR,
)
from multimodal.answers import FIELDS, questions_for_display
from fraudvision.paths import DATA

BG='#0e1828'; PANEL='#1c2b40'; PANEL2='#22344c'; BORDER='#344961'
TEXT='#eaf3f8'; MUTED='#a2b9cf'; TEAL='#40d7ce'; ORANGE='#ff9a79'; BLUE='#8ab9ff'
FIELDS_LABEL={
    'document_type':'Document type',
    'date':'Date',
    'total':'Printed total',
    'payment_reference':'Payment reference',
    'arithmetic_consistency':'Arithmetic check',
}

class MultimodalComparisonView:
    def __init__(self, parent):
        self.parent=parent
        self.loaded=False
        self._load_queue=SimpleQueue()
        self._loading=False
        self._queued_request=None
        self.ocr_path=None
        self.vlm_path=None
        self.result={'documents': [], 'metrics': {}, 'by_field': {}}
        self.selection=None
        self.image_photo=None
        self.filter_value=tk.StringVar(value='All documents')
        self.file_value=tk.StringVar()
        self.info=tk.StringVar(value='Open this tab to load existing experiment reports.')
        self._ui()

    def _txt(self, parent, value, size=10, fg=TEXT, bold=False, **kwargs):
        return tk.Label(parent, text=value, bg=parent.cget('bg'), fg=fg,
                        font=('Segoe UI',size,'bold' if bold else 'normal'),**kwargs)

    def _button(self, parent, value, fn, primary=False):
        b=tk.Button(parent,text=value,command=fn,relief='flat',cursor='hand2',
                    bg=TEAL if primary else '#dce7ee',fg='#102035',
                    activebackground='#aaf7eb',font=('Segoe UI',9,'bold'),padx=12,pady=4)
        b.pack(side='left',padx=(0,7))
        return b

    def _ui(self):
        hdr=tk.Frame(self.parent,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        hdr.pack(fill='x',padx=7,pady=(9,5))
        self._txt(hdr,'MULTIMODAL LAB  /  OCR vs VLM',12,TEAL,True).pack(side='left',padx=(12,15),pady=8)
        self._txt(hdr,'Ground truth • matching held-out images • saved reports only',9,MUTED).pack(side='left',pady=8)
        actions=tk.Frame(self.parent,bg=BG)
        actions.pack(fill='x',padx=9,pady=(0,2))
        self._button(actions,'Refresh reports',self.autoload,True)
        self._button(actions,'Load OCR JSON',lambda:self.choose_report('ocr'))
        self._button(actions,'Load VLM JSON',lambda:self.choose_report('vlm'))
        self._button(actions,'Export comparison CSV',self.save_csv)
        self._button(actions,'Inspect raw responses',self.show_raw)
        self.status_lbl=tk.Label(self.parent,textvariable=self.info,bg=BG,fg=MUTED,
                                 font=('Segoe UI',9),anchor='w')
        self.status_lbl.pack(fill='x',padx=12,pady=(0,2))

        stats=tk.Frame(self.parent,bg=BG)
        stats.pack(fill='x',padx=5,pady=(0,3))
        self.metrics={}
        for key,title in [('matched','MATCHED IMAGES'),('ocr','OCR EXACT MATCH'),('vlm','VLM EXACT MATCH'),('gap','ACCURACY DIFFERENCE')]:
            card=tk.Frame(stats,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
            card.pack(side='left',fill='x',expand=True,padx=4)
            self._txt(card,title,9,MUTED,True).pack(anchor='w',padx=14,pady=(5,0))
            v=self._txt(card,'—',18,TEAL if key=='ocr' else TEXT,True)
            v.pack(anchor='w',padx=14,pady=(1,5))
            self.metrics[key]=v

        body=tk.Frame(self.parent,bg=BG)
        body.pack(fill='both',expand=True,padx=6,pady=(0,7))
        body.columnconfigure(1,weight=1);body.rowconfigure(0,weight=1)
        left=tk.Frame(body,bg=PANEL,width=314,highlightbackground=BORDER,highlightthickness=1)
        left.grid(row=0,column=0,sticky='nsew',padx=(0,6))
        left.grid_propagate(False)
        self._txt(left,'DOCUMENT INSPECTOR',11,TEAL,True).pack(anchor='w',padx=12,pady=(12,4))
        filterbar=tk.Frame(left,bg=PANEL)
        filterbar.pack(fill='x',padx=12,pady=(0,6))
        options=['All documents','VLM errors','OCR errors','Either errors','Model disagreement']
        filterbox=ttk.Combobox(filterbar,textvariable=self.filter_value,values=options,width=22,state='readonly')
        filterbox.pack(side='left',fill='x',expand=True)
        filterbox.bind('<<ComboboxSelected>>',lambda e:self.update_file_list())
        self.file_list=tk.Listbox(left,bg='#16283d',fg=TEXT,relief='flat',
                                  selectbackground='#23617b',selectforeground='white',
                                  height=4,exportselection=False,font=('Consolas',10))
        self.file_list.pack(fill='x',padx=12,pady=(0,8))
        self.file_list.bind('<<ListboxSelect>>',self.select_current)
        self.file_ids=[]
        self.image_caption=self._txt(left,'Select an image',10,MUTED)
        self.image_caption.pack(anchor='w',padx=12,pady=(2,7))
        open_btn=tk.Button(left,text='Open larger image',command=self.show_full_image,relief='flat',
                           bg='#dce7ee',fg='#102035',cursor='hand2',font=('Segoe UI',9,'bold'),pady=4)
        open_btn.pack(fill='x',padx=12,pady=(0,6))
        self.preview=tk.Canvas(left,bg='#263449',highlightthickness=0)
        self.preview.pack(fill='both',expand=True,padx=12,pady=(0,12))
        self.preview.bind('<Configure>',lambda e:self.draw_image())

        right=tk.Frame(body,bg=BG)
        right.grid(row=0,column=1,sticky='nsew',padx=(0,0))
        right.rowconfigure(0,weight=1,minsize=230);right.rowconfigure(1,weight=0,minsize=145)
        right.columnconfigure(0,weight=1)
        tablepane=tk.Frame(right,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        tablepane.grid(row=0,column=0,sticky='nsew',padx=0,pady=(0,5))
        self._txt(tablepane,'PER-QUESTION EVIDENCE  •  SELECTED IMAGE',11,TEAL,True).pack(anchor='w',padx=12,pady=(12,4))
        self._txt(tablepane,'Green = exact match    Coral = incorrect    — = not run or unavailable',9,MUTED).pack(anchor='w',padx=12,pady=(0,7))
        cols=('question','truth','ocr','vlm','status')
        self.table=ttk.Treeview(tablepane,columns=cols,show='headings',height=5,selectmode='browse')
        for key,title,width in [('question','QUESTION',160),('truth','GROUND TRUTH',155),
                                ('ocr','OCR ANSWER',140),('vlm','SMOLVLM2 ANSWER',155),('status','RESULT',110)]:
            self.table.heading(key,text=title)
            self.table.column(key,width=width,anchor='w',stretch=key in ('question','truth','ocr','vlm'))
        self.table.tag_configure('both_correct',background='#174a49',foreground='#f1fff8')
        self.table.tag_configure('vlm_wrong',background='#633941',foreground='#fff5f3')
        self.table.tag_configure('ocr_wrong',background='#624430',foreground='#fff5ef')
        self.table.tag_configure('plain',background=PANEL,foreground=TEXT)
        self.table.pack(fill='both',expand=True,padx=12,pady=(0,9))
        self.question_hint=self._txt(tablepane,'',9,MUTED,wraplength=850,justify='left',anchor='w')
        self.question_hint.pack(fill='x',padx=12,pady=(0,7))
        self.table.bind('<<TreeviewSelect>>',self.select_question)

        chartpane=tk.Frame(right,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        chartpane.grid(row=1,column=0,sticky='nsew',padx=0,pady=(5,0))
        self._txt(chartpane,'ACCURACY BY QUESTION TYPE  •  MATCHED IMAGES',11,TEAL,True).pack(anchor='w',padx=12,pady=(12,3))
        self._txt(chartpane,'Teal: OCR    Blue: SmolVLM2    (gray indicates report not loaded)',9,MUTED).pack(anchor='w',padx=12)
        self.chart=tk.Canvas(chartpane,bg=PANEL,highlightthickness=0,height=120)
        self.chart.pack(fill='both',expand=True,padx=12,pady=(2,7))
        self.chart.bind('<Configure>',lambda e:self.draw_chart())

    def on_open(self):
        if not self.loaded:
            self.loaded=True
            self.autoload()

    def autoload(self):
        """Discover reports and evaluate labels OFF the Tkinter/UI thread."""
        self._start_loading(discover=True)

    def choose_report(self,method):
        path=filedialog.askopenfilename(title=f'Load {method.upper()} evaluation JSON',
              initialdir=str(REPORT_DIR),filetypes=[('JSON report','*.json')])
        if not path:return
        try:
            load_report(path,method)  # Lightweight schema validation only
        except Exception as exc:
            messagebox.showerror('Invalid report',str(exc),parent=self.parent);return
        if method=='ocr':self.ocr_path=Path(path)
        else:self.vlm_path=Path(path)
        self.refresh()

    def refresh(self):
        self._start_loading(discover=False)

    def _start_loading(self,discover=False):
        # Never regenerate the ground-truth answer key on the Tk event loop.
        # On slower Windows machines this step can take several seconds and
        # previously made the desktop appear permanently frozen.
        request=(discover,self.ocr_path,self.vlm_path)
        if self._loading:
            self._queued_request=request
            self.info.set('Preparing report comparison in the background …')
            return
        self._loading=True
        self.info.set('Preparing comparison in the background — window stays responsive …')
        def worker():
            try:
                ocr_path,vlm_path=(discover_reports() if discover else (request[1],request[2]))
                ocr=load_report(ocr_path,'ocr') if ocr_path else None
                vlm=load_report(vlm_path,'vlm') if vlm_path else None
                result=assemble(ocr,vlm)
                self._load_queue.put(('ok',ocr_path,vlm_path,result))
            except Exception as exc:
                self._load_queue.put(('error',str(exc)))
        Thread(target=worker,name='FraudFlow-report-loader',daemon=True).start()
        self.parent.after(100,self._poll_load)

    def _poll_load(self):
        # All widget operations happen on Tk's main thread.
        try:message=self._load_queue.get_nowait()
        except Empty:
            self.parent.after(100,self._poll_load)
            return
        self._loading=False
        if message[0]=='ok':
            _,self.ocr_path,self.vlm_path,self.result=message
        else:
            self.info.set(f'Unable to load reports: {message[1]}')
            self.result={'documents':[], 'metrics':{}, 'by_field':{}}
        self._set_metrics()
        self.update_file_list()
        self.draw_chart()
        if self._queued_request is not None:
            request=self._queued_request
            self._queued_request=None
            self.ocr_path,self.vlm_path=request[1:]
            self._start_loading(discover=request[0])

    def _set_metrics(self):
        result=self.result
        metrics=result.get('metrics',{})
        n=len(result.get('documents',[]))
        self.metrics['matched'].config(text=str(n))
        for name in ('ocr','vlm'):
            m=metrics.get(name)
            self.metrics[name].config(text=f'{m["accuracy"]:.0%}  ({m["correct"]}/{m["total"]})' if m else 'NOT RUN',
                                      fg=TEAL if name=='ocr' else BLUE)
        if 'ocr' in metrics and 'vlm' in metrics:
            diff=(metrics['ocr']['accuracy']-metrics['vlm']['accuracy'])*100
            self.metrics['gap'].config(text=f'OCR {diff:+.1f} pp' if diff else 'TIED')
        else:self.metrics['gap'].config(text='—')
        if not n:
            self.info.set('No saved reports found. Run a VQA experiment or click Load OCR/VLM JSON.')
            return
        selected=', '.join(f'{x.upper()}: {Path(self.ocr_path if x=="ocr" else self.vlm_path).name}'
                           for x in ('ocr','vlm') if (self.ocr_path if x=='ocr' else self.vlm_path))
        qualifier='exactly matching images' if result.get('paired') else 'one evaluation report only'
        self.info.set(f'{selected}   |   {n} {qualifier}   |   Synthetic held-out test only')

    def update_file_list(self):
        cur=self.selection['file'] if self.selection else None
        docs=self.result.get('documents',[])
        mode=self.filter_value.get()
        filtered=[]
        for d in docs:
            if mode=='VLM errors' and not d['vlm_errors']:continue
            if mode=='OCR errors' and not d['ocr_errors']:continue
            if mode=='Either errors' and not(d['vlm_errors'] or d['ocr_errors']):continue
            if mode=='Model disagreement' and not any(
                 f['ocr_correct'] is not None and f['vlm_correct'] is not None and f['ocr'] != f['vlm']
                 for f in d['fields']):continue
            filtered.append(d)
        self.file_ids=[d['file'] for d in filtered]
        self.file_list.delete(0,'end')
        for d in filtered:
            suffix=f'  VLM ✗{d["vlm_errors"]}' if d['vlm_errors'] else ''
            self.file_list.insert('end',f'{Path(d["file"]).name}{suffix}')
        if filtered:
            idx=next((i for i,d in enumerate(filtered) if d['file']==cur),0)
            self.file_list.selection_set(idx)
            self.file_list.see(idx)
            self.show_document(filtered[idx])
        else:
            self.selection=None
            self.table.delete(*self.table.get_children())
            self.image_caption.config(text='No matching documents')
            self.preview.delete('all')
            self.question_hint.config(text='')

    def select_current(self,event=None):
        selection=self.file_list.curselection()
        if not selection:return
        file=self.file_ids[selection[0]]
        doc=next((d for d in self.result['documents'] if d['file']==file),None)
        if doc:self.show_document(doc)

    def show_document(self,doc):
        self.selection=doc
        self.image_caption.config(text=f'{Path(doc["file"]).name}   •   OCR {doc["runtime"]["ocr"] if doc["runtime"]["ocr"] is not None else "—"}s'
                                  f'   •   VLM {doc["runtime"]["vlm"] if doc["runtime"]["vlm"] is not None else "—"}s')
        self.table.delete(*self.table.get_children())
        for f in doc['fields']:
            ocr='—' if f['ocr_correct'] is None else (f['ocr'] or '(blank)')
            vlm='—' if f['vlm_correct'] is None else (f['vlm'] or '(blank)')
            if f['vlm_correct'] is False: tag='vlm_wrong'
            elif f['ocr_correct'] is False:tag='ocr_wrong'
            elif f['ocr_correct'] and f['vlm_correct']:tag='both_correct'
            else:tag='plain'
            if f['ocr_correct'] is None and f['vlm_correct'] is None:status='NOT RUN'
            elif f['vlm_correct'] is False and f['ocr_correct'] is False:status='BOTH WRONG'
            elif f['vlm_correct'] is False:status='VLM WRONG'
            elif f['ocr_correct'] is False:status='OCR WRONG'
            elif f['vlm_correct'] is None:status='OCR OK' if f['ocr_correct'] else 'OCR WRONG'
            elif f['ocr_correct'] is None:status='VLM OK' if f['vlm_correct'] else 'VLM WRONG'
            else: status='BOTH OK'
            self.table.insert('', 'end', iid=f['field'], values=(FIELDS_LABEL[f['field']],f['expected'],ocr,vlm,status),tags=(tag,))
        self.question_hint.config(text='Select a question row to view the exact prompt used by the experiment.')
        self.draw_image()

    def select_question(self,event=None):
        selected=self.table.selection()
        if not selected:return
        field=selected[0]
        self.question_hint.config(text=questions_for_display().get(field,field))

    def draw_image(self):
        self.preview.delete('all')
        if not self.selection:return
        w=self.preview.winfo_width();h=self.preview.winfo_height()
        if w<70 or h<70:return
        try:
            with Image.open(DATA/self.selection['file']) as orig:
                im=orig.convert('RGB')
                im.thumbnail((max(70,w-20),max(70,h-20)),Image.Resampling.LANCZOS)
            self.image_photo=ImageTk.PhotoImage(im)
            self.preview.create_image(w/2,h/2,image=self.image_photo,anchor='center')
        except (OSError,ValueError) as exc:
            self.preview.create_text(w/2,h/2,text=f'Image unavailable: {exc}',fill=ORANGE,width=w-20)

    def draw_chart(self):
        if not hasattr(self,'chart'):return
        self.chart.delete('all')
        w=max(self.chart.winfo_width(),500);h=max(self.chart.winfo_height(),185)
        data=self.result.get('by_field',{})
        if not data:
            self.chart.create_text(w/2,h/2,text='No measured VQA results to chart yet',fill=MUTED,font=('Segoe UI',11))
            return
        left=162;right=75
        room=max(180,w-left-right)
        unit=max(18,min(27,(h-24)//len(FIELDS)))
        for i,key in enumerate(FIELDS):
            y=8+i*unit
            self.chart.create_text(8,y+8,text=FIELDS_LABEL[key],anchor='w',fill=TEXT,font=('Segoe UI',9))
            for j,(method,color) in enumerate((('ocr',TEAL),('vlm',BLUE))):
                yy=y+j*10
                m=data[key].get(method)
                self.chart.create_rectangle(left,yy,left+room,yy+8,outline='',fill='#31465b')
                if m and m['accuracy'] is not None:
                    length=room*m['accuracy']
                    if length>0:self.chart.create_rectangle(left,yy,left+length,yy+8,outline='',fill=color)
                    label=f'{m["correct"]}/{m["total"]}'
                else:label='—'
                self.chart.create_text(left+room+10,yy+3,text=label,fill=color if m else MUTED,
                                       anchor='w',font=('Consolas',9,'bold'))

    def show_full_image(self):
        if not self.selection:return
        dlg=tk.Toplevel(self.parent)
        dlg.configure(bg=BG)
        dlg.title(f'Document image — {Path(self.selection["file"]).name}')
        dlg.geometry('740x860')
        image=Image.open(DATA/self.selection['file']).convert('RGB')
        image.thumbnail((660,760),Image.Resampling.LANCZOS)
        photo=ImageTk.PhotoImage(image)
        label=tk.Label(dlg,image=photo,bg=BG)
        label.image=photo
        label.pack(fill='both',expand=True,padx=8,pady=8)

    def show_raw(self):
        if not self.selection:
            messagebox.showinfo('Select an image','Select an image first.',parent=self.parent);return
        dlg=tk.Toplevel(self.parent)
        dlg.title(f'Measured model responses — {Path(self.selection["file"]).name}')
        dlg.configure(bg=BG)
        dlg.geometry('850x670')
        tk.Label(dlg,text='RAW OCR OUTPUT / RAW VLM GENERATIONS',bg=BG,fg=TEAL,
                 font=('Segoe UI',14,'bold')).pack(anchor='w',padx=15,pady=12)
        txt=tk.Text(dlg,wrap='word',bg=PANEL,fg=TEXT,font=('Consolas',10),padx=14,pady=13)
        txt.pack(fill='both',expand=True,padx=15,pady=(0,15))
        for key in ('ocr','vlm'):
            raw=self.selection['raw'].get(key,'')
            txt.insert('end',f'\n=== {key.upper()} ===\n')
            txt.insert('end',(str(raw) if raw else '(not available)')+'\n')
        txt.config(state='disabled')

    def save_csv(self):
        if not self.result.get('documents'):
            messagebox.showinfo('No results','Load at least one experiment report first.',parent=self.parent);return
        path=filedialog.asksaveasfilename(title='Save paired VQA evidence',defaultextension='.csv',
                                          initialfile='fraudflow_vqa_comparison.csv',
                                          filetypes=[('CSV','*.csv')])
        if path:
            n=export_rows(self.result,path)
            messagebox.showinfo('Saved',f'Saved {n} question-level evidence rows.\n{path}',parent=self.parent)
