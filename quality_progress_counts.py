"""Keep completed first-pass work visible while a measured repair is running."""


def counts(progress,selection):
    records={str(ep):[row] for ep,row in progress.get('episodes',{}).items()}
    for history in progress.get('followup_history',[]):
        previous=history.get('previous',{})
        if previous:records.setdefault(str(history['episode']),[]).append(previous)
    reviewed={int(ep) for ep,rows in records.items() if any(row.get('status')=='completed' for row in rows)}
    # Older optimizer continuations recorded their prior limit without a full
    # previous row. Admission itself required the first review to be complete.
    reviewed.update(int(h['episode']) for h in progress.get('followup_history',[]) if h.get('kind')=='measured_optimizer_continuation')
    reviewed=sorted(reviewed)
    selected=sorted(int(ep) for ep,rows in records.items() if any(row.get('selected') and row.get('run')==selection.get('episodes',{}).get(ep,{}).get('run') for row in rows))
    followups=sorted(ep for ep in reviewed if progress.get('episodes',{}).get(str(ep),{}).get('status')!='completed')
    return dict(reviewed=reviewed,selected=selected,followups=followups)
