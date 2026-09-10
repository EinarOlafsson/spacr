"""Disclosed synthetic dose series with an independent, planted-truth oracle.

These are arithmetic examples, not measured compounds or biological evidence.
Balanced +/- 0.5 offsets deliberately preserve each dose's exact model mean;
their near-perfect fits do not assess statistical calibration on real data.
"""
import csv
import hashlib
import math
from pathlib import Path
import tempfile

DOSES=(0,.01,.03,.1,.3,1,3,10,30,100)
TRUTH={
    'SYNTHETIC inhibition':dict(bottom=5.,top=95.,ec50=1.5,hill=-1.2,vehicle=95.),
    'SYNTHETIC activation':dict(bottom=10.,top=90.,ec50=2.,hill=1.,vehicle=10.),
    'SYNTHETIC beyond range':dict(bottom=5.,top=95.,ec50=500.,hill=1.,vehicle=5.),
}
REVERSAL='SYNTHETIC reversal'


def synthetic_rows():
    """120 rows: four series, nine positive doses plus vehicle, three repeats."""
    rows=[]
    for group in (*TRUTH,REVERSAL):
        for index,dose in enumerate(DOSES):
            if group=='SYNTHETIC inhibition':mean=5+90/(1+(dose/1.5)**1.2)
            elif group=='SYNTHETIC activation':mean=10+80*dose/(2+dose)
            elif group=='SYNTHETIC beyond range':mean=5+90*dose/(500+dose)
            else:mean=(10,10,20,50,80,100,80,40,20,10)[index]
            for replicate,offset in enumerate((-.5,0,.5),1):
                rows.append(dict(series=group,concentration=dose,response=mean+offset,replicate=replicate))
    return rows


def prepare(stage):
    parent=Path(stage)/'dose_response_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='disclosed-synthetic-',dir=parent))
    path=work/'SYNTHETIC_dose_response_examples.csv';rows=synthetic_rows()
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=120,synthetic=True,random=False,offsets=[-.5,0,.5],biological_claim=False)


def close(actual,expected,field,tolerance=1e-6):
    if actual is None or not math.isfinite(float(actual)) or not math.isclose(float(actual),expected,abs_tol=tolerance,rel_tol=tolerance):
        raise ValueError('Dose-response output disagrees with planted truth: '+field)


def check_fit(group,record):
    """Compare exposed native values with the independent constructed series."""
    truth=TRUTH[group]
    for field in ('bottom','top','hill'):close(record[field],truth[field],field)
    close(record['ec50_unconstrained'],truth['ec50'],'unconstrained midpoint')
    expected=dict(n_obs=27,n_doses=9,dof=23,n_vehicle=3,n_excluded=0,
                  vehicle_response=truth['vehicle'],sse=4.5,rse=math.sqrt(4.5/23))
    for field,value in expected.items():close(record[field],value,field)
    rows=[r for r in synthetic_rows() if r['series']==group and r['concentration']>0]
    if len(record['dose'])!=27 or len(record['response'])!=27:
        raise ValueError('Fitted observations omit or add source rows')
    for index,row in enumerate(rows):
        close(record['dose'][index],row['concentration'],'dose order',1e-12)
        close(record['response'][index],row['response'],'response identity',1e-12)
    if truth['ec50']>max(DOSES):
        if record['status']!='unbounded' or record['ec50'] is not None or record['ec50_low'] is not None or record['ec50_high'] is not None:
            raise ValueError('Unsupported midpoint must not acquire a point estimate or interval')
        if record['bound_direction']!='above':raise ValueError('One-sided bound points the wrong way')
    else:
        if record['status']!='fitted':raise ValueError('The fully bracketed example must remain fitted')
        close(record['ec50'],truth['ec50'],'bounded midpoint')
        if not record['ec50_low']<truth['ec50']<record['ec50_high']:
            raise ValueError('The constructed midpoint is not inside its reported interval')
    return dict(observations_checked=27,source_values_checked=54,planted_midpoint=truth['ec50'],status=record['status'])


def profile_sse(dose,response,midpoint,sign):
    """Profile by SVD linear least squares and a separate slope optimiser.

    No spaCR fitting, prediction, initialisation or profile helper is called.
    The application's closed-form 2x2 solve is replaced by numpy's SVD solve;
    six starts for scipy least_squares replace its sampled/Brent slope path.
    """
    import numpy as np
    from scipy.optimize import least_squares
    x=np.asarray(dose,dtype=float);y=np.asarray(response,dtype=float)
    def residual(log_magnitude):
        h=sign*math.exp(float(log_magnitude[0]))
        weight=1/(1+(midpoint/x)**h)
        design=np.column_stack((np.ones(x.size),weight))
        coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
        return design@coefficients-y
    solutions=[least_squares(residual,[math.log(s)],bounds=([math.log(.01)],[math.log(50)]),
        ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=200) for s in (.2,.5,1,2,4,8)]
    return min(float(np.dot(s.fun,s.fun)) for s in solutions if s.success)


def check_profile(group,record):
    """Recompute endpoints to the application's declared log-space precision.

    Its PROFILE_TOLERANCE is a 0.001-wide log10 bracket, not an SSE error.
    The reported midpoint may differ from the root by half that width. A
    direct 0.001 SSE tolerance wrongly rejects a correctly rounded endpoint
    on these deliberately narrow intervals; compare the actual parameter.
    """
    from scipy.stats import t
    from scipy.optimize import brentq
    truth=TRUTH[group];sign=1 if truth['hill']>0 else -1
    target=4.5*(1+float(t.ppf(.975,23))**2/23)
    middle=profile_sse(record['dose'],record['response'],truth['ec50'],sign)
    close(middle,4.5,'profile at planted midpoint',1e-6)
    endpoints=[]
    centre=math.log10(truth['ec50'])
    def boundary(log_midpoint):
        return profile_sse(record['dose'],record['response'],10**log_midpoint,sign)-target
    for key,bracket in [('ec50_low',(centre-.2,centre)),('ec50_high',(centre,centre+.2))]:
        root=brentq(boundary,*bracket,xtol=1e-10)
        value=profile_sse(record['dose'],record['response'],record[key],sign)
        error=abs(math.log10(record[key])-root)
        if error>.0005:
            raise ValueError('Reported profile endpoint exceeds its declared log-space precision')
        endpoints.append(dict(field=key,midpoint=record[key],sse=value,
                              independent_midpoint=10**root,log10_error=error))
    return dict(target_sse=target,centre_sse=middle,endpoints=endpoints,
                log10_error_tolerance=.0005)


def check_groups(fits):
    if len(fits)!=4 or set(fits)!=set(TRUTH)|{REVERSAL}:
        raise ValueError('A fitted, unbounded or refused series disappeared')
    rejected=fits[REVERSAL]
    if rejected['result'] is not None or 'not monotone' not in rejected['error']:
        raise ValueError('The deliberately reversing example was not refused for its shape')


def check_wald(group,record):
    """Rebuild the log-midpoint covariance from the analytic 4PL Jacobian."""
    import numpy as np
    from scipy.stats import t
    truth=TRUTH[group];x=np.asarray(record['dose']);h=truth['hill']
    w=1/(1+(truth['ec50']/x)**h);span=truth['top']-truth['bottom']
    j=np.column_stack((1-w,w,-span*math.log(10)*h*w*(1-w),
                       -span*np.log(truth['ec50']/x)*w*(1-w)))
    covariance=np.linalg.inv(j.T@j)*(4.5/23)
    delta=float(t.ppf(.975,23))*math.sqrt(float(covariance[2,2]))
    centre=math.log10(truth['ec50']);endpoints={}
    for key,sign in [('ec50_low',-1),('ec50_high',1)]:
        expected=10**(centre+sign*delta);close(record[key],expected,key,1e-5)
        endpoints[key]=expected
    return dict(independent_endpoints=endpoints,parameter='log10_ec50',dof=23,tolerance=1e-5)


def check_drawn_curves(axes):
    """Check every drawn observation and all 600 curve coordinates."""
    lines=axes.get_lines();checks=[]
    if axes.get_xscale()!='log':raise ValueError('Actual concentration axis is not logarithmic')
    for group,truth in TRUTH.items():
        matches=[(i,line) for i,line in enumerate(lines) if line.get_label()==group]
        if len(matches)!=1:raise ValueError('Actual figure lost or duplicated a named series')
        index,points=matches[0];rows=[r for r in synthetic_rows() if r['series']==group and r['concentration']>0]
        xs=points.get_xdata();ys=points.get_ydata()
        if len(xs)!=27 or len(ys)!=27:raise ValueError('Actual figure includes vehicle or loses observations')
        for x,y,row in zip(xs,ys,rows):
            close(x,row['concentration'],'drawn dose',1e-12);close(y,row['response'],'drawn response',1e-12)
        curve=lines[index+1];cx=curve.get_xdata();cy=curve.get_ydata()
        if len(cx)!=200 or len(cy)!=200:raise ValueError('Actual fitted curve has missing coordinates')
        for i,(x,y) in enumerate(zip(cx,cy)):
            close(x,10**(-2.5+5*i/199),'curve concentration',1e-12)
            expected=truth['bottom']+(truth['top']-truth['bottom'])/(1+(truth['ec50']/x)**truth['hill'])
            close(y,expected,'curve response',1e-6)
        checks.append(dict(group=group,points=27,curve_points=200))
    return checks
