"""FraudFlow Vision: desktop-only document alteration screening dashboard.

Launch: python -m fraudvision.desktop
Requires only Pillow + Python standard library; no localhost and no SciPy.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
from .core import VisualModel, load_manifest, train_pipeline, score_dataset
from .features import extract
from .ocr import read_text
from .verification import verify_amounts_text, combined_decision
from .paths import ROOT, DATA, MANIFEST, MODEL, REPORTS
from . import charts

BG='#0e1828'; PANEL='#1c2b40'; PANEL2='#22344c'; BORDER='#344961';TEXT='#eaf3f8'; MUTED='#a2b9cf';TEAL='#40d7ce';ORANGE='#ff9a79';SOFT='#142137'
FONT=('Segoe UI',10)

class FraudFlowApp:
    def __init__(self, root):
        self.root=root
        root.title('FraudFlow Vision | Document Forensics Studio')
        root.configure(bg=BG);root.geometry('1520x920');root.minsize(1180,740)
        self.model=VisualModel.load(MODEL) if MODEL.exists() else None
        self.rows=load_manifest() if MANIFEST.exists() else []
        self.scored=[];self.selected=None;self.selected_path=None;self.manual=False
        self.photo=None;self.highlight=tk.BooleanVar(value=True)
        self.ocr_checks={} # Optional on-demand OCR evidence per image, not synthetic labels
        self.filter_var=tk.StringVar(value='All documents');self.search_var=tk.StringVar()
        self._theme();self._ui()
        if self.model and self.rows:
            self._load_predictions()
        else:
            self.status('No trained demo model found. Click Train Model to begin.')

    def _theme(self):
        s=ttk.Style(self.root)
        s.theme_use('clam')
        s.configure('TNotebook',background=BG,borderwidth=0)
        s.configure('TNotebook.Tab',background=PANEL,padding=(19,12),foreground=TEXT,font=FONT)
        s.map('TNotebook.Tab',background=[('selected',PANEL2)],foreground=[('selected',TEAL)])
        s.configure('Treeview',background=PANEL,fieldbackground=PANEL,foreground=TEXT,rowheight=29,font=('Segoe UI',10),borderwidth=0)
        s.configure('Treeview.Heading',background='#31455d',foreground=TEXT,font=('Segoe UI',10,'bold'),relief='flat')
        s.map('Treeview',background=[('selected','#275c72')],foreground=[('selected','#ffffff')])
        s.configure('TCombobox',padding=3)

    def _label(self,parent,text,fg=TEXT,size=10,bold=False,**kw):
        label=tk.Label(parent,text=text,bg=parent.cget('bg'),fg=fg,font=('Segoe UI',size,'bold' if bold else 'normal'),**kw)
        return label

    def _button(self,parent,text,command,accent=False):
        b=tk.Button(parent,text=text,command=command,bg=TEAL if accent else '#dfe7ed',fg='#0d1c2b',
                    activebackground='#aff6eb',activeforeground='#0d1c2b',relief='flat',cursor='hand2',
                    font=('Segoe UI',10,'bold'),padx=16,pady=9)
        b.pack(side='left',padx=(0,9))
        return b

    def _ui(self):
        header=tk.Frame(self.root,bg=BG);header.pack(fill='x',padx=22,pady=(15,8))
        title=tk.Frame(header,bg=BG);title.pack(side='left')
        self._label(title,'FRAUDFLOW',fg=TEAL,size=25,bold=True).pack(side='left')
        self._label(title,'   VISION  /  DOCUMENT FORENSICS STUDIO',fg=MUTED,size=11).pack(side='left',pady=(8,0))
        self._label(header,'OFFLINE  •  PYTHON 3.14  •  SYNTHETIC DATA',fg='#ffcb80',size=9).pack(side='right',padx=5)
        toolbar=tk.Frame(self.root,bg=BG);toolbar.pack(fill='x',padx=22,pady=(2,10))
        self._button(toolbar,'Open image',self.open_image)
        self._button(toolbar,'Train model',self.train_model)
        self._button(toolbar,'Analyze all',self.analyze_all,True)
        self._button(toolbar,'Export CSV',self.export_csv)
        self._button(toolbar,'Save evidence',self.save_evidence)
        self._button(toolbar,'Verify totals (OCR)',self.verify_selected)
        self._button(toolbar,'Extract OCR text',self.open_ocr)
        self.status_var=tk.StringVar(value='Ready')
        tk.Label(self.root,textvariable=self.status_var,bg=BG,fg=MUTED,font=('Segoe UI',10),anchor='w').pack(fill='x',padx=22,pady=(0,9))
        cardrow=tk.Frame(self.root,bg=BG);cardrow.pack(fill='x',padx=22,pady=(0,13))
        names=['DOCUMENTS','VISUAL REVIEW FLAGS','VISUAL NO FLAGS','VISUAL TEST F1','TEST RECALL']
        self.cards={}
        for i,name in enumerate(names):
            box=tk.Frame(cardrow,bg=PANEL,highlightthickness=1,highlightbackground=BORDER)
            box.pack(side='left',expand=True,fill='x',padx=(0,9 if i<4 else 0))
            tk.Label(box,text=name,bg=PANEL,fg=MUTED,font=('Segoe UI',9,'bold')).pack(anchor='w',padx=16,pady=(12,0))
            lbl=tk.Label(box,text='—',bg=PANEL,fg=TEXT,font=('Segoe UI',25,'bold'))
            lbl.pack(anchor='w',padx=16,pady=(2,11))
            self.cards[name]=lbl
        self.tabs=ttk.Notebook(self.root)
        self.tabs.pack(fill='both',expand=True,padx=22,pady=(0,12))
        self.inspection=tk.Frame(self.tabs,bg=BG);self.analytics=tk.Frame(self.tabs,bg=BG)
        self.explorer=tk.Frame(self.tabs,bg=BG);self.evaluation=tk.Frame(self.tabs,bg=BG)
        self.errors=tk.Frame(self.tabs,bg=BG)
        self.multimodal=tk.Frame(self.tabs,bg=BG)
        for page,name in [(self.inspection,'Inspection Studio'),(self.analytics,'Analytics Dashboard'),
                          (self.explorer,'Dataset Explorer'),(self.evaluation,'Model & Evaluation'),(self.errors,'Error Analysis'),
                          (self.multimodal,'Multimodal Comparison')]:
            self.tabs.add(page,text=name)
        self._inspection_ui();self._analytics_ui();self._explorer_ui();self._evaluation_ui();self._errors_ui()
        from .multimodal_view import MultimodalComparisonView
        self.multimodal_view=MultimodalComparisonView(self.multimodal)
        self.tabs.bind('<<NotebookTabChanged>>',lambda e: self.multimodal_view.on_open()
                       if self.tabs.select()==str(self.multimodal) else None)

    def _inspection_ui(self):
        self.inspection.columnconfigure(1,weight=1);self.inspection.rowconfigure(0,weight=1)
        side=tk.Frame(self.inspection,bg=PANEL,width=245);side.grid(row=0,column=0,sticky='ns',padx=(0,8),pady=7)
        side.grid_propagate(False)
        self._label(side,'IMAGE GALLERY',fg=TEAL,size=11,bold=True).pack(anchor='w',padx=13,pady=(16,6))
        self._label(side,'Select a receipt or invoice to inspect',fg=MUTED,size=9,wraplength=215).pack(anchor='w',padx=13,pady=(0,10))
        self.gallery=tk.Listbox(side,bg=PANEL,fg=TEXT,selectbackground='#28617a',selectforeground='white',
                                relief='flat',highlightthickness=0,font=('Consolas',10),activestyle='none',exportselection=False)
        self.gallery.pack(side='left',fill='both',expand=True,padx=(12,0),pady=(0,14))
        scroll=tk.Scrollbar(side,command=self.gallery.yview);scroll.pack(side='right',fill='y',pady=(0,14))
        self.gallery.config(yscrollcommand=scroll.set)
        self.gallery.bind('<<ListboxSelect>>',self._gallery_select)
        preview=tk.Frame(self.inspection,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        preview.grid(row=0,column=1,sticky='nsew',pady=7,padx=(0,8))
        top=tk.Frame(preview,bg=PANEL);top.pack(fill='x',padx=14,pady=(12,3))
        self._label(top,'DOCUMENT IMAGE',fg=TEAL,size=12,bold=True).pack(side='left')
        tk.Checkbutton(top,text='Show inferred visual hot-zone',variable=self.highlight,command=self._refresh_image,
                       bg=PANEL,fg=TEXT,activebackground=PANEL,activeforeground=TEAL,selectcolor=PANEL2).pack(side='right')
        self.img_canvas=tk.Canvas(preview,bg='#253043',highlightthickness=0)
        self.img_canvas.pack(fill='both',expand=True,padx=14,pady=(0,14))
        self.img_canvas.bind('<Configure>',lambda e:self._refresh_image())
        details=tk.Frame(self.inspection,bg=PANEL,width=332,highlightbackground=BORDER,highlightthickness=1)
        details.grid(row=0,column=2,sticky='ns',pady=7)
        details.grid_propagate(False)
        self._label(details,'INSPECTION RESULT',fg=TEAL,size=12,bold=True).pack(anchor='w',padx=15,pady=(18,5))
        self.decision_lbl=tk.Label(details,text='SELECT A DOCUMENT',bg=PANEL,fg=MUTED,font=('Segoe UI',16,'bold'))
        self.decision_lbl.pack(anchor='w',padx=15,pady=(0,4))
        self.risk_lbl=self._label(details,'Risk score: —',size=12)
        self.risk_lbl.pack(anchor='w',padx=15,pady=(1,12))
        self._label(details,'VISUAL / OCR EVIDENCE',fg=MUTED,size=9,bold=True).pack(anchor='w',padx=15)
        self.details_box=tk.Text(details,wrap='word',bg=PANEL,fg=TEXT,relief='flat',font=('Consolas',10),
                                 padx=15,pady=12,height=29,highlightthickness=0)
        self.details_box.pack(fill='both',expand=True,padx=(2,8),pady=(0,12))
        self.details_box.config(state='disabled')

    def _analytics_ui(self):
        self.analytics.rowconfigure(0,weight=1);self.analytics.rowconfigure(1,weight=1)
        self.analytics.columnconfigure(0,weight=2);self.analytics.columnconfigure(1,weight=1)
        self.hist=self._canvas_panel(self.analytics,0,0,'')
        self.mix=self._canvas_panel(self.analytics,0,1,'')
        bottom=tk.Frame(self.analytics,bg=PANEL,highlightthickness=1,highlightbackground=BORDER)
        bottom.grid(row=1,column=0,columnspan=2,sticky='nsew',padx=5,pady=7)
        self._label(bottom,'RISK ANALYTICS — DOCUMENT TYPE & REVIEW DISTRIBUTION',fg=TEAL,size=12,bold=True).pack(anchor='w',padx=18,pady=(16,12))
        self.analytics_text=tk.Label(bottom,text='Train and analyze to see model results.',bg=PANEL,fg=TEXT,
                                     font=('Consolas',12),justify='left',anchor='nw',wraplength=950)
        self.analytics_text.pack(fill='both',expand=True,padx=22,pady=12,anchor='nw')
        self.hist.bind('<Configure>',lambda e:self._refresh_charts());self.mix.bind('<Configure>',lambda e:self._refresh_charts())

    def _canvas_panel(self,parent,r,c,name):
        frame=tk.Frame(parent,bg=PANEL,highlightthickness=1,highlightbackground=BORDER)
        frame.grid(row=r,column=c,sticky='nsew',padx=5,pady=7)
        canvas=tk.Canvas(frame,bg=PANEL,highlightthickness=0)
        canvas.pack(fill='both',expand=True)
        return canvas

    def _explorer_ui(self):
        toolbar=tk.Frame(self.explorer,bg=BG);toolbar.pack(fill='x',padx=6,pady=(13,8))
        self._label(toolbar,'Show',fg=MUTED,size=10).pack(side='left',padx=(0,8))
        filters=['All documents','Review only','No flag only','Receipts','Invoices','Train split','Validation split','Test split']
        selector=ttk.Combobox(toolbar,textvariable=self.filter_var,values=filters,state='readonly',width=19)
        selector.pack(side='left');selector.bind('<<ComboboxSelected>>',lambda e:self._fill_explorer())
        self._label(toolbar,'   Search file/type:',fg=MUTED,size=10).pack(side='left',padx=(15,8))
        entry=tk.Entry(toolbar,textvariable=self.search_var,width=28,bg='#edf4f7',fg='#102135',relief='flat',font=FONT)
        entry.pack(side='left');self.search_var.trace_add('write',lambda *a:self._fill_explorer())
        self.count_lbl=self._label(toolbar,'',fg=TEAL,size=10)
        self.count_lbl.pack(side='right')
        columns=['file','type','split','risk','prediction','ground_truth','edit']
        table_box=tk.Frame(self.explorer,bg=PANEL);table_box.pack(fill='both',expand=True,padx=6)
        self.tree=ttk.Treeview(table_box,columns=columns,show='headings',selectmode='browse')
        for col,title,width in [('file','Document image',195),('type','Type',115),('split','Split',100),
                                ('risk','Model score',110),('prediction','Decision',130),
                                ('ground_truth','Synthetic label',130),('edit','Edit category',130)]:
            self.tree.heading(col,text=title)
            self.tree.column(col,width=width,anchor='center' if col!='file' else 'w',stretch=(col=='file'))
        self.tree.tag_configure('review',background='#513841',foreground='#ffffff')
        self.tree.pack(side='left',fill='both',expand=True)
        ys=ttk.Scrollbar(table_box,orient='vertical',command=self.tree.yview);ys.pack(side='right',fill='y')
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.bind('<<TreeviewSelect>>',self._explorer_select)
        foot=tk.Frame(self.explorer,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        foot.pack(fill='x',padx=6,pady=(9,6))
        self._label(foot,'SELECTED RECORD',fg=TEAL,bold=True).pack(anchor='w',padx=12,pady=(10,0))
        self.explorer_text=tk.Label(foot,text='Select a row to inspect its image and model explanation.',bg=PANEL,
                                    fg=TEXT,font=('Consolas',10),justify='left',anchor='w',wraplength=1180)
        self.explorer_text.pack(fill='x',padx=12,pady=(6,12))
        self._button(foot,'Open in Inspection Studio',self.open_selected_in_inspection).pack_forget() if False else None

    def _evaluation_ui(self):
        self.evaluation.columnconfigure(0,weight=1);self.evaluation.columnconfigure(1,weight=1)
        self.evaluation.rowconfigure(1,weight=1)
        hdr=tk.Frame(self.evaluation,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        hdr.grid(row=0,column=0,columnspan=2,sticky='ew',padx=6,pady=(9,6))
        self._label(hdr,'TRAIN → VALIDATE THRESHOLD → TEST ON HELD-OUT IMAGES',fg=TEAL,size=13,bold=True).pack(anchor='w',padx=15,pady=(14,4))
        self._label(hdr,'Labels and edit boxes are not used at prediction time. Test data is excluded from training and threshold selection.',
                    fg=MUTED,size=10).pack(anchor='w',padx=15,pady=(0,14))
        self.confusion=self._canvas_panel(self.evaluation,1,0,'')
        panel=tk.Frame(self.evaluation,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        panel.grid(row=1,column=1,sticky='nsew',padx=6,pady=7)
        self._label(panel,'REPRODUCIBLE MODEL EVALUATION',fg=TEAL,size=12,bold=True).pack(anchor='w',padx=17,pady=(20,8))
        self.eval_box=tk.Text(panel,wrap='word',bg=PANEL,fg=TEXT,relief='flat',font=('Consolas',11),padx=17,pady=8)
        self.eval_box.pack(fill='both',expand=True)
        self.eval_box.config(state='disabled')
        self.confusion.bind('<Configure>',lambda e:self._refresh_charts())

    def _errors_ui(self):
        self._label(self.errors,'HELD-OUT TEST ERRORS — VISUAL MODEL',fg=TEAL,size=13,bold=True).pack(anchor='w',padx=14,pady=(16,4))
        self._label(self.errors,'False positives and false negatives only. This list uses ground truth for evaluation, NEVER prediction.',
                    fg=MUTED,size=10).pack(anchor='w',padx=14,pady=(0,12))
        columns=['file','document_type','label','prediction','score','error_type']
        area=tk.Frame(self.errors,bg=PANEL);area.pack(fill='both',expand=True,padx=12,pady=8)
        self.errors_tree=ttk.Treeview(area,columns=columns,show='headings',selectmode='browse')
        for key,name,width in [('file','Document',210),('document_type','Type',130),('label','Actual label',135),
                               ('prediction','Visual decision',135),('score','Visual score',130),('error_type','Error type',170)]:
            self.errors_tree.heading(key,text=name);self.errors_tree.column(key,width=width)
        self.errors_tree.pack(side='left',fill='both',expand=True)
        sb=ttk.Scrollbar(area,orient='vertical',command=self.errors_tree.yview);sb.pack(side='right',fill='y')
        self.errors_tree.configure(yscrollcommand=sb.set)
        self.errors_tree.bind('<<TreeviewSelect>>',self._error_select)
        self._label(self.errors,'Select a row to open the document in Inspection Studio. See reports/ocr_error_analysis.csv for the combined OCR audit.',
                    fg=MUTED,size=9).pack(anchor='w',padx=16,pady=(4,14))
        self._populate_errors()

    def _populate_errors(self):
        if not hasattr(self,'errors_tree'):return
        self.errors_tree.delete(*self.errors_tree.get_children())
        file=REPORTS/'error_analysis.csv'
        if not file.exists():return
        with file.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                self.errors_tree.insert('', 'end',iid=row['file'], values=(Path(row['file']).name,
                    row['document_type'], 'EDITED' if int(row['label']) else 'CLEAN',
                    'REVIEW' if int(row['prediction']) else 'NO FLAG',
                    f'{float(row["risk_score"]):.3f}',row['error_type']))

    def _error_select(self,event):
        keys=self.errors_tree.selection()
        if not keys:return
        row=next((r for r in self.scored if r['file']==keys[0]),None)
        if row:
            self.select_row(row)
            self.tabs.select(self.inspection)

    def _selected_evidence_text(self,ev):
        row=self.selected
        features=', '.join(f'{name}={val:.2f}' for name,val in zip(
            ['h_peak','h_top','h_edges','v_peak','v_top','v_edges','h_run','v_run'],ev.features))
        visual=row['prediction']
        audit=self.ocr_checks.get(str(self.selected_path))
        check=audit[0] if audit else None
        combined=combined_decision(visual=='REVIEW',check)
        self.decision_lbl.config(text=combined,fg=ORANGE if combined=='REVIEW' else TEAL)
        text=(f'FILE\n{self.selected_path.name}\n\n'
              f'VISUAL SCREEN\nDecision: {visual}\nScore: {float(row["risk_score"]):.4f}\n'
              f'Pixel evidence: {ev.description}\nFeatures: {features}\n\n'
              f'OCR ARITHMETIC CHECK\n')
        if check is None:
            text+='NOT RUN — select Verify totals (OCR) to check subtotal + tax.\n'
        else:
            text+=(f'Status: {check.status}\n{check.message}\n'
                   f'Subtotal: {check.subtotal or "?"} | Tax: {check.tax or "?"}\n'
                   f'Printed total: {check.printed_total or "?"}\n'
                   f'Calculated total: {check.expected_total or "?"}\n'
                   f'Difference: {check.difference or "?"}\n')
        text+=f'\nCOMBINED DECISION\n{combined} (visual OR arithmetic mismatch)\n'
        if check is None or check.status in ('UNREADABLE','OCR_UNAVAILABLE'):
            text+='OCR arithmetic unverified; NO FLAG never proves authenticity.\n'
        if not self.manual:
            text+=(f'\nSYNTHETIC GROUND TRUTH\n'
                   f'{"EDITED" if int(row["label"]) else "CLEAN"}\n'
                   '(Used for benchmark evaluation ONLY, never prediction.)\n')
        else:text+='\nEXTERNAL IMAGE — UNKNOWN GROUND TRUTH\n'
        text+='\nDISCLAIMER\nAn indicator for manual review, not proof of financial fraud.'
        self._write_text(self.details_box,text)

    def verify_selected(self):
        if not self.selected_path:return
        self.status('Running independent Tesseract total verification ...')
        self.root.update_idletasks()
        path=str(self.selected_path)
        if path not in self.ocr_checks:
            ocr_text=read_text(self.selected_path)
            self.ocr_checks[path]=(verify_amounts_text(ocr_text),ocr_text)
        check=self.ocr_checks[path][0]
        ev=extract(self.selected_path)
        self._selected_evidence_text(ev)
        self.status(f'OCR result: {check.status}. Visual and numerical evidence shown independently.')

    def status(self,message):self.status_var.set(message)
    def _write_text(self,widget,message):
        widget.config(state='normal');widget.delete('1.0','end');widget.insert('end',message);widget.config(state='disabled')

    def _load_predictions(self):
        precomputed=DATA/'demo_predictions.csv'
        if precomputed.exists():
            lookup={}
            with precomputed.open(newline='',encoding='utf-8') as f:
                for r in csv.DictReader(f): lookup[r['file']]=r
            if len(lookup)==len(self.rows):
                self.scored=[{**r,**{'risk_score':float(lookup[r['file']]['risk_score']),
                                'prediction':lookup[r['file']]['prediction'],'hot_zone':None,
                                'description':'Select the image to compute visual evidence.'}} for r in self.rows]
            else:self.scored=score_dataset(self.rows,self.model)
        else:self.scored=score_dataset(self.rows,self.model)
        self.status(f'Ready: {len(self.scored)} synthetic document images. Select one for visual inspection.')
        self._refresh_all()
        if self.scored:self.select_row(self.scored[0])

    def _refresh_all(self):
        self.gallery.delete(0,'end')
        for row in self.scored:
            mark='⚠' if row['prediction']=='REVIEW' else '•'
            self.gallery.insert('end',f'{mark} {Path(row["file"]).name}  {row["document_type"][:3].upper()}')
        flagged=sum(row['prediction']=='REVIEW' for row in self.scored)
        self.cards['DOCUMENTS'].config(text=str(len(self.scored)))
        self.cards['VISUAL REVIEW FLAGS'].config(text=str(flagged),fg=ORANGE)
        self.cards['VISUAL NO FLAGS'].config(text=str(len(self.scored)-flagged))
        report_path=REPORTS/'evaluation.json'
        if report_path.exists():
            report=json.loads(report_path.read_text(encoding='utf-8'))
            m=report['splits']['test']
            self.cards['VISUAL TEST F1'].config(text=f'{m["f1"]:.1%}')
            self.cards['TEST RECALL'].config(text=f'{m["recall"]:.1%}')
            audit_path=REPORTS/'ocr_audit.json'
            audit=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
            self._write_text(self.eval_box,
                'SOURCE: SYNTHETIC DOCUMENT DATASET\n\n'
                + f'Train samples: {report["splits"]["train"]["total"]}\n'
                + f'Validation samples: {report["splits"]["validation"]["total"]}\n'
                + f'Test samples: {m["total"]}\n\n'
                + f'Selected validation threshold: {report["threshold"]:.2f}\n\n'
                + f'Test precision: {m["precision"]:.1%}\n'
                + f'Test recall: {m["recall"]:.1%}\n'
                + f'Test F1: {m["f1"]:.1%}\n'
                + f'Test accuracy: {m["accuracy"]:.1%}\n\n'
                + f'False positives: {m["fp"]}\nFalse negatives: {m["fn"]}\n\n'
                + ('OCR-ASSISTED HELD-OUT TEST (FIXED OR RULE)\n'
                   + f'Precision: {audit["ocr_assisted_combined"]["precision"]:.1%} | Recall: {audit["ocr_assisted_combined"]["recall"]:.1%}\n'
                   + f'F1: {audit["ocr_assisted_combined"]["f1"]:.1%} | FP: {audit["ocr_assisted_combined"]["fp"]} | FN: {audit["ocr_assisted_combined"]["fn"]}\n'
                   + 'Separate audit; see reports/ocr_audit.json.\n\n' if audit else 'OCR ASSISTED AUDIT: NOT RUN\n\n')
                + 'LIMITATION\nScores detect synthetic visual replacement artifacts.\n'
                + 'No claim of reliable real-world document fraud detection.\n'
                + 'The suspicious region is inferred from pixels, not from labels.')
        self._fill_explorer();self._refresh_charts();self._populate_errors()
        rr=[r for r in self.scored if r['document_type']=='receipt']
        ii=[r for r in self.scored if r['document_type']=='invoice']
        def nflags(items):return sum(r['prediction']=='REVIEW' for r in items)
        self.analytics_text.config(text=f'RECEIPTS      {len(rr):3d} images  |  {nflags(rr):3d} flagged for review\n\n'
                                        f'INVOICES      {len(ii):3d} images  |  {nflags(ii):3d} flagged for review\n\n'
                                        'Validation/test metrics describe synthetic edit artifacts only.\n'
                                        'A review flag is not evidence of actual financial fraud.')

    def _refresh_charts(self):
        if not self.model or not self.scored:return
        charts.render_histogram(self.hist,[r['risk_score'] for r in self.scored],self.model.threshold)
        charts.render_decision_mix(self.mix,self.scored)
        rp=REPORTS/'evaluation.json'
        if rp.exists():charts.render_confusion(self.confusion,json.loads(rp.read_text(encoding='utf-8'))['splits']['test'])

    def _fill_explorer(self):
        if not hasattr(self,'tree'):return
        self.tree.delete(*self.tree.get_children())
        value=self.filter_var.get();term=self.search_var.get().strip().lower()
        filtered=[]
        for row in self.scored:
            if term and term not in (row['file']+' '+row['document_type']).lower():continue
            if value=='Review only' and row['prediction']!='REVIEW':continue
            if value=='No flag only' and row['prediction']!='NO FLAG':continue
            if value=='Receipts' and row['document_type']!='receipt':continue
            if value=='Invoices' and row['document_type']!='invoice':continue
            if value in ['Train split','Validation split','Test split'] and row['split']!=value.split(' ')[0].lower():continue
            filtered.append(row)
        for row in filtered:
            self.tree.insert('', 'end',iid=row['file'],values=(Path(row['file']).name,row['document_type'].capitalize(),row['split'],
                             f'{row["risk_score"]:.3f}',row['prediction'], 'edited' if int(row['label']) else 'clean',row['edit_type']),
                             tags=('review',) if row['prediction']=='REVIEW' else ())
        self.count_lbl.config(text=f'{len(filtered)} records')

    def _gallery_select(self,event):
        sel=self.gallery.curselection()
        if sel and sel[0]<len(self.scored):self.select_row(self.scored[sel[0]])
    def _explorer_select(self,event):
        selected=self.tree.selection()
        if selected:
            row=next((r for r in self.scored if r['file']==selected[0]),None)
            if row:
                self.select_row(row)
                self.explorer_text.config(text=(f'{row["file"]}  |  model score {row["risk_score"]:.3f}  |  {row["prediction"]}  |  '
                  f'Synthetic ground truth: {"EDITED" if int(row["label"]) else "CLEAN"}\n'
                  f'Click Inspection Studio to view the full document and pixel-based hot-zone.'))
    def open_selected_in_inspection(self):self.tabs.select(self.inspection)
    def select_row(self,row):
        self.selected=row;self.manual=False
        self.selected_path=DATA/row['file']
        ev=extract(self.selected_path)
        row['hot_zone']=ev.suspicious_region;row['description']=ev.description
        self.decision_lbl.config(text=row['prediction'],fg=ORANGE if row['prediction']=='REVIEW' else TEAL)
        self.risk_lbl.config(text=f'Model score: {float(row["risk_score"]):.3f}  |  threshold: {self.model.threshold:.2f}')
        self._selected_evidence_text(ev)
        self.status(f'Inspecting {Path(row["file"]).name} — {row["prediction"]} | model score {float(row["risk_score"]):.3f}')
        self._refresh_image()

    def _refresh_image(self):
        if not self.selected_path or not self.selected_path.exists():return
        w=self.img_canvas.winfo_width();h=self.img_canvas.winfo_height()
        if w<80 or h<80:return
        with Image.open(self.selected_path) as original:
            im=original.convert('RGB')
        if self.highlight.get() and self.selected and self.selected.get('hot_zone'):
            x1,y1,x2,y2=self.selected['hot_zone']
            overlay=Image.new('RGBA',im.size,(0,0,0,0))
            draw=ImageDraw.Draw(overlay)
            draw.rectangle((x1,y1,x2,y2),outline=(255,106,79,255),fill=(255,126,83,38),width=5)
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
        im.thumbnail((max(80,w-30),max(80,h-28)),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(im)
        self.img_canvas.delete('all');self.img_canvas.create_image(w/2,h/2,image=self.photo,anchor='center')

    def open_image(self):
        path=filedialog.askopenfilename(title='Select a receipt or invoice image',filetypes=[('Images','*.png *.jpg *.jpeg *.bmp'),('All','*.*')])
        if not path:return
        if not self.model:
            messagebox.showerror('Model required','Train or load the demo model first.');return
        try:e=extract(path)
        except Exception as exc: messagebox.showerror('Image error',str(exc));return
        risk=self.model.risk(e.features)
        self.selected={'file':str(path),'document_type':'external image','risk_score':risk,
                       'prediction':'REVIEW' if risk>=self.model.threshold else 'NO FLAG',
                       'hot_zone':e.suspicious_region,'description':e.description,'label':'0'}
        self.selected_path=Path(path);self.manual=True
        self.decision_lbl.config(text=self.selected['prediction'],fg=ORANGE if self.selected['prediction']=='REVIEW' else TEAL)
        self.risk_lbl.config(text=f'Model score: {risk:.3f}  |  threshold: {self.model.threshold:.2f}')
        self._selected_evidence_text(e)
        self.tabs.select(self.inspection);self._refresh_image();self.status('External image screened (outside the synthetic benchmark).')

    def train_model(self):
        if not MANIFEST.exists():return messagebox.showerror('Missing data','Synthetic manifest not found')
        if not messagebox.askyesno('Train model','Retrain on train split and select threshold using validation split? This may take a minute.'):return
        self.status('Extracting image features and training the model ...');self.root.update_idletasks()
        try:self.model,report=train_pipeline()
        except Exception as exc:return messagebox.showerror('Training failed',str(exc))
        self.scored=score_dataset(self.rows,self.model)
        # regenerate cache that exactly matches retrained model
        with (DATA/'demo_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['file','risk_score','prediction']);writer.writeheader()
            for row in self.scored:writer.writerow({k:row[k] for k in ('file','risk_score','prediction')})
        self.ocr_checks.clear()
        self._refresh_all();self._populate_errors();self.select_row(self.scored[0]);self.status('Training complete; test results regenerated with held-out test split.')

    def analyze_all(self):
        if not self.model:return messagebox.showerror('Model required','Train first')
        self.status('Analyzing all images from pixels ...');self.root.update_idletasks()
        try:self.scored=score_dataset(self.rows,self.model)
        except Exception as exc:return messagebox.showerror('Analysis error',str(exc))
        self._refresh_all();self.select_row(self.scored[0]);self.status('Analysis complete. Select a document to inspect its visible evidence.')

    def export_csv(self):
        if not self.scored:return
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV files','*.csv')],initialfile='fraudflow_predictions.csv')
        if not path:return
        fields=['file','split','document_type','label','edit_type','risk_score','prediction']
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in self.scored:w.writerow({k:r.get(k,'') for k in fields})
        self.status(f'Exported {len(self.scored)} predictions to {path}')

    def save_evidence(self):
        if not self.selected_path:return
        path=filedialog.asksaveasfilename(title='Save evidence report',defaultextension='.json',
                                          filetypes=[('JSON files','*.json')],initialfile='inspection_report.json')
        if not path:return
        ev=extract(self.selected_path)
        output={'source_image':str(self.selected_path),'model_score':round(float(self.selected['risk_score']),6),
                'decision':self.selected['prediction'],'inferred_hot_zone':ev.suspicious_region,
                'evidence':ev.description,'features':dict(zip(['h_peak','h_top','h_edges','v_peak','v_top','v_edges','h_run','v_run'],ev.features)),
                'warning':'Synthetic artifact screening only; not a conclusion of document fraud.'}
        audit=self.ocr_checks.get(str(self.selected_path))
        check=audit[0] if audit else None
        output['ocr_arithmetic_check']=check.to_dict() if check else {'status':'NOT_RUN'}
        output['combined_decision']=combined_decision(self.selected['prediction']=='REVIEW',check)
        output['decision_type']='visual OR OCR arithmetic mismatch, when OCR was run'
        Path(path).write_text(json.dumps(output,indent=2),encoding='utf-8')
        # Save annotated image next to report when a hot zone was inferred.
        if ev.suspicious_region:
            with Image.open(self.selected_path) as orig:pic=orig.convert('RGB')
            ImageDraw.Draw(pic).rectangle(ev.suspicious_region,outline='#ed5f49',width=6)
            pic.save(str(Path(path).with_suffix('.annotated.png')))
        self.status(f'Saved evidence report: {path}')

    def open_ocr(self):
        if not self.selected_path:return
        self.status('Running local Tesseract OCR ...');self.root.update_idletasks()
        path=str(self.selected_path)
        if path not in self.ocr_checks:
            text=read_text(self.selected_path)
            self.ocr_checks[path]=(verify_amounts_text(text),text)
        check,text=self.ocr_checks[path]
        self._selected_evidence_text(extract(self.selected_path))
        dialog=tk.Toplevel(self.root);dialog.title('Local OCR — Extracted Document Text')
        dialog.geometry('770x560');dialog.configure(bg=BG)
        self._label(dialog,'OCR TEXT — LOCAL TESSERACT',fg=TEAL,size=14,bold=True).pack(anchor='w',padx=18,pady=(17,8))
        tk.Label(dialog,text='OCR output is for inspection only. It is not used by the current image classifier.',
                 bg=BG,fg=MUTED).pack(anchor='w',padx=18)
        box=tk.Text(dialog,wrap='word',bg=PANEL,fg=TEXT,font=('Consolas',11),padx=12,pady=10)
        box.pack(fill='both',expand=True,padx=18,pady=14)
        box.insert('end',f'OCR CHECK: {check.status}\n{check.message}\n\n{text}');box.config(state='disabled')
        self.status('OCR result displayed.')


def main():
    root=tk.Tk(); FraudFlowApp(root);root.mainloop()

if __name__=='__main__':main()
