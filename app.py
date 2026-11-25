from pathlib import Path
from datetime import datetime, timezone, timedelta
import sqlite3, math, secrets, hashlib, hmac, time, os
from typing import Optional
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT=Path(__file__).parent; DB_PATH=ROOT/'easy_to_connect.db'; STATIC_DIR=ROOT/'static'
app=FastAPI(title='Easy to Connect',description='Personal travel planning, emergency decisions, budgets, nearby places and ML learning.',version='1.0.0')
app.mount('/static',StaticFiles(directory=STATIC_DIR),name='static')
MODEL=None; MODEL_METRICS={}; LAST_GEO_REQUEST=0.0

class Credentials(BaseModel):
    email:str=Field(min_length=5,max_length=160)
    password:str=Field(min_length=8,max_length=128)
class GoogleCredential(BaseModel):
    credential: str = Field(min_length=20, max_length=10000)

class CompareRequest(BaseModel):
    origin:str=Field(min_length=2,max_length=160); destination:str=Field(min_length=2,max_length=160)
    origin_lat:Optional[float]=None; origin_lon:Optional[float]=None; destination_lat:Optional[float]=None; destination_lon:Optional[float]=None
    urgency_hours:int=Field(default=8,ge=1,le=168); budget_inr:int=Field(default=5000,ge=0,le=1000000)
    priority:str=Field(default='balanced',pattern='^(fastest|cheapest|balanced)$')
class SavePlan(BaseModel):
    origin:str; destination:str; option:dict; reason:str=Field(min_length=3,max_length=500)
class BudgetEntry(BaseModel):
    title:str=Field(min_length=2,max_length=120); category:str=Field(default='Other',max_length=40); amount_inr:float=Field(gt=0,le=1000000); journey_id:Optional[int]=None
class NearbyRequest(BaseModel):
    lat:float=Field(ge=-90,le=90); lon:float=Field(ge=-180,le=180); radius_m:int=Field(default=5000,ge=500,le=25000)
class PredictionRequest(BaseModel):
    distance_km:float=Field(ge=1,le=3000)
    mode:str=Field(default='train',pattern='^(bus|train|flight)$')
    transfers:int=Field(default=0,ge=0,le=5)
    departure_hour:int=Field(default=9,ge=0,le=23)
    weather_factor:float=Field(default=0.35,ge=0,le=1)

def connect_db():
    c=sqlite3.connect(DB_PATH,timeout=10); c.row_factory=sqlite3.Row; c.execute('PRAGMA foreign_keys=ON'); return c

def init_db():
    c=connect_db(); c.executescript('''
    CREATE TABLE IF NOT EXISTS cities(id INTEGER PRIMARY KEY,name TEXT NOT NULL UNIQUE,lat REAL NOT NULL,lon REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,email TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,expires_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS saved_plans(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,origin TEXT NOT NULL,destination TEXT NOT NULL,option_json TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS budget_entries(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,title TEXT NOT NULL,category TEXT NOT NULL,amount_inr REAL NOT NULL,journey_id INTEGER,created_at TEXT NOT NULL);
    ''')
    cities=[('Cuttack',20.4625,85.8828),('Bhubaneswar',20.2707,85.8407),('Puri',19.8135,85.8312),('Kolkata',22.5726,88.3639),('Visakhapatnam',17.6868,83.2185),('Hyderabad',17.3850,78.4867),('Chennai',13.0827,80.2707),('Delhi',28.6139,77.2090),('Mumbai',19.0760,72.8777),('Bengaluru',12.9716,77.5946),('Patna',25.5941,85.1376),('Ranchi',23.3441,85.3096),('Jaipur',26.9124,75.7873),('Lucknow',26.8467,80.9462),('Pune',18.5204,73.8567),('Ahmedabad',23.0225,72.5714),('Bhopal',23.2599,77.4126),('Guwahati',26.1445,91.7362),('Indore',22.7196,75.8577),('Varanasi',25.3176,82.9739),('Amritsar',31.6340,74.8723),('Coimbatore',11.0168,76.9558),('Kochi',9.9312,76.2673),('Srinagar',34.0837,74.7973)]
    c.executemany('INSERT OR IGNORE INTO cities(name,lat,lon) VALUES(?,?,?)',cities); c.commit(); c.close()

def train_model():
    global MODEL,MODEL_METRICS
    rng=np.random.default_rng(27); n=1800
    distance=rng.integers(15,1900,n); mode=rng.integers(0,3,n); transfers=rng.integers(0,3,n); hour=rng.integers(0,24,n); weather=rng.uniform(0,1,n)
    factor=np.choose(mode,[.20,.13,.035]); y=np.maximum(0,3+distance/220+transfers*8+weather*24+factor*20+((hour>=7)&(hour<=10))*6+rng.normal(0,7,n))
    X=np.column_stack([distance,mode,transfers,hour,weather]); a,b,ya,yb=train_test_split(X,y,test_size=.25,random_state=27)
    MODEL=RandomForestRegressor(n_estimators=100,max_depth=9,random_state=27); MODEL.fit(a,ya); p=MODEL.predict(b)
    MODEL_METRICS={'mae_minutes':round(float(mean_absolute_error(yb,p)),2),'r2':round(float(r2_score(yb,p)),3),'test_rows':len(yb),'training_data':'synthetic demonstration data; not suitable for real-world travel predictions','features':['distance_km','mode_code (bus=0, train=1, flight=2)','transfers','departure_hour','simulated_weather_traffic_factor']}

init_db(); train_model()
def hash_password(pw:str,salt:Optional[bytes]=None):
    salt=salt or secrets.token_bytes(16); derived=hashlib.pbkdf2_hmac('sha256',pw.encode(),salt,240000); return salt.hex()+':'+derived.hex()
def verify_password(pw,stored):
    try:
        salt_hex,digest=stored.split(':'); expected=hashlib.pbkdf2_hmac('sha256',pw.encode(),bytes.fromhex(salt_hex),240000).hex(); return hmac.compare_digest(expected,digest)
    except Exception:return False
def token_hash(token):return hashlib.sha256(token.encode()).hexdigest()

def create_session_response(user_id: int, email: str, status: str = 'logged_in'):
    """Create a persistent browser session for an already verified account."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=365)
    connection = connect_db()
    connection.execute(
        'INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)',
        (token_hash(token), user_id, expires.isoformat()),
    )
    connection.commit()
    connection.close()
    response = Response(content=f'{{"status":"{status}"}}', media_type='application/json')
    response.set_cookie(
        'etc_session', token, max_age=365 * 24 * 3600, httponly=True,
        samesite='lax', secure=False, path='/',
    )
    return response
def current_user(request:Request):
    token=request.cookies.get('etc_session')
    if not token: raise HTTPException(status_code=401,detail='Please log in to use your personal workspace.')
    c=connect_db(); row=c.execute('SELECT u.id,u.email,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=?',(token_hash(token),)).fetchone()
    if not row or datetime.fromisoformat(row['expires_at'])<datetime.now(timezone.utc):
        if token:c.execute('DELETE FROM sessions WHERE token_hash=?',(token_hash(token),)); c.commit()
        c.close(); raise HTTPException(status_code=401,detail='Your session expired. Please log in again.')
    c.close(); return {'id':row['id'],'email':row['email']}
def norm(s):return ' '.join(s.strip().lower().split())
def haversine(lat1,lon1,lat2,lon2):
    p1,p2=math.radians(lat1),math.radians(lat2); dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1); a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2; return 6371*2*math.atan2(math.sqrt(a),math.sqrt(max(0,1-a)))
def estimate_options(origin,dest,distance,urgency,budget,priority):
    modes=[('Bus',0,max(25,int(distance/55*60)),max(80,int(distance*.75)),1),('Train',1,max(20,int(distance/70*60)),max(100,int(distance*.55)),0),('Flight',2,max(60,int(distance/500*60)+100),max(1800,int(distance*4.0)),0)]
    rows=[]
    for mode,code,mins,cost,transfers in modes:
        if distance<120 and mode=='Flight':continue
        if distance<40 and mode=='Train': mins=max(20,int(distance/45*60)); cost=max(30,int(distance*.45))
        hour=datetime.now().hour; pred=max(0,float(MODEL.predict([[distance,code,transfers,hour,.35]])[0])); total=mins+round(pred)
        if priority=='fastest': score=total+transfers*25
        elif priority=='cheapest': score=cost/10+mins/100+pred
        else: score=total/5+cost/100+transfers*10
        affordability='Budget-friendly' if cost<=budget*.4 else ('Normal cost' if cost<=budget else 'Expensive')
        rows.append({'id':f'est-{code}','origin':origin,'destination':dest,'mode':mode,'provider':f'{mode} estimate (illustrative)','departure':'Estimate only','arrival':'Estimate only','duration_minutes':mins,'cost_inr':cost,'distance_km':round(distance),'transfers':transfers,'predicted_delay_minutes_demo':round(pred),'estimated_total_minutes_demo':total,'under_budget':cost<=budget,'priority_score_demo':round(score,2),'recommended':False,'affordability':affordability,'data_status':'Calculated estimate, not a live fare or schedule'})
    rows.sort(key=lambda x:x['priority_score_demo']);
    if rows: rows[0]['recommended']=True
    return rows

@app.get('/',response_class=HTMLResponse)
def index():return (STATIC_DIR/'index.html').read_text(encoding='utf-8')
@app.get('/api/health')
def health():return {'status':'ok','application':'Easy to Connect','version':'1.0.0'}
@app.post('/api/auth/register')
def register(req:Credentials):
    email=norm(req.email)
    if '@' not in email or '.' not in email.split('@')[-1]:raise HTTPException(400,'Enter a valid email address.')
    c = connect_db()
    try:
        cur = c.execute(
            'INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)',
            (email, hash_password(req.password), datetime.now(timezone.utc).isoformat()),
        )
        user_id = cur.lastrowid
        c.commit()
    except sqlite3.IntegrityError:
        c.close()
        raise HTTPException(409, 'An account with this email already exists. Please log in.')
    c.close()
    return create_session_response(user_id, email, 'registered')
@app.post('/api/auth/login')
def login(req:Credentials):
    c=connect_db(); row=c.execute('SELECT id,password_hash FROM users WHERE email=?',(norm(req.email),)).fetchone(); c.close()
    if not row or not verify_password(req.password,row['password_hash']):
        raise HTTPException(401,'Email or password is incorrect.')
    return create_session_response(row['id'], norm(req.email))

@app.get('/api/auth/google/config')
def google_login_config():
    """Expose only the public Google OAuth client ID needed by the browser SDK."""
    return {'client_id': os.getenv('GOOGLE_CLIENT_ID', '').strip()}

@app.post('/api/auth/google')
def google_login(req: GoogleCredential):
    client_id = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    if not client_id:
        raise HTTPException(503, 'Google sign-in is not configured yet. Add GOOGLE_CLIENT_ID to your environment; see README.md.')
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
        identity = id_token.verify_oauth2_token(req.credential, google_requests.Request(), client_id)
    except ImportError:
        raise HTTPException(503, 'Google sign-in dependency is missing. Run: python -m pip install -r requirements.txt')
    except Exception:
        raise HTTPException(401, 'Google could not verify this sign-in. Please try again.')

    email = norm(identity.get('email', ''))
    if not identity.get('email_verified') or not email.endswith('@gmail.com'):
        raise HTTPException(403, 'Please sign in with a verified @gmail.com account.')

    c = connect_db()
    row = c.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
    if row:
        user_id = row['id']
    else:
        cursor = c.execute(
            'INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)',
            (email, hash_password(secrets.token_urlsafe(40)), datetime.now(timezone.utc).isoformat()),
        )
        user_id = cursor.lastrowid
    c.commit()
    c.close()
    return create_session_response(user_id, email)
@app.post('/api/auth/logout')
def logout(request:Request):
    token=request.cookies.get('etc_session'); c=connect_db()
    if token:c.execute('DELETE FROM sessions WHERE token_hash=?',(token_hash(token),)); c.commit()
    c.close(); r=Response(content='{"status":"logged_out"}',media_type='application/json'); r.delete_cookie('etc_session',path='/'); return r
@app.get('/api/auth/me')
def me(user=Depends(current_user)):return {'id':user['id'],'email':user['email']}
@app.get('/api/cities')
def cities():
    c=connect_db(); rows=c.execute('SELECT name,lat,lon FROM cities ORDER BY name').fetchall(); c.close(); return [dict(r) for r in rows]
@app.get('/api/places/search')
def search_places(q:str):
    global LAST_GEO_REQUEST
    q=q.strip()
    if len(q)<3:raise HTTPException(400,'Enter at least 3 characters to search for a place.')
    if len(q)>120:raise HTTPException(400,'Search text is too long.')
    # Nominatim is called only on explicit form submission, not autocomplete; respect public API rate limits.
    wait=1.05-(time.monotonic()-LAST_GEO_REQUEST)
    if wait>0:time.sleep(wait)
    LAST_GEO_REQUEST=time.monotonic()
    try:
        res=requests.get('https://nominatim.openstreetmap.org/search',params={'q':q,'format':'jsonv2','limit':8,'countrycodes':'in','addressdetails':1},headers={'User-Agent':'EasyToConnectStudentProject/1.0 (educational demo)'},timeout=12); res.raise_for_status(); data=res.json()
        return [{'name':x.get('display_name',q),'lat':float(x['lat']),'lon':float(x['lon']),'type':x.get('type','place'),'source':'OpenStreetMap'} for x in data]
    except requests.RequestException:raise HTTPException(502,'Place search is temporarily unavailable. Try again or use the sample city list.')
@app.get('/api/compare')
def compare(origin:str,destination:str,urgency_hours:int=8,budget_inr:int=5000,priority:str='balanced',origin_lat:Optional[float]=None,origin_lon:Optional[float]=None,destination_lat:Optional[float]=None,destination_lon:Optional[float]=None,user=Depends(current_user)):
    if norm(origin)==norm(destination):raise HTTPException(400,'Choose different boarding and destination points.')
    if not 1<=urgency_hours<=168 or budget_inr<0 or priority not in ('fastest','cheapest','balanced'):raise HTTPException(400,'Invalid journey preferences.')
    c=connect_db(); a=c.execute('SELECT lat,lon FROM cities WHERE lower(name)=?',(norm(origin),)).fetchone(); b=c.execute('SELECT lat,lon FROM cities WHERE lower(name)=?',(norm(destination),)).fetchone(); c.close()
    lat1=origin_lat if origin_lat is not None else (a['lat'] if a else None); lon1=origin_lon if origin_lon is not None else (a['lon'] if a else None); lat2=destination_lat if destination_lat is not None else (b['lat'] if b else None); lon2=destination_lon if destination_lon is not None else (b['lon'] if b else None)
    if None not in (lat1,lon1,lat2,lon2):
        distance=haversine(lat1,lon1,lat2,lon2); results=estimate_options(origin,destination,distance,urgency_hours,budget_inr,priority)
        return {'results':results,'recommendation_basis':priority,'message':'Calculated travel-mode estimates only. Fares, departures and journey durations are not live provider data. Verify with official operators.'}
    return {'results':[],'recommendation_basis':priority,'message':'Select locations from search results or use cities in the sample list so coordinates are available.'}
# Offline fallback locations are approximate educational examples, not live verified POIs.
# Keep labels explicit so users never mistake fallback records for confirmed real-world services.
FALLBACK_PLACES = {
    'Bhubaneswar': [
        {'name':'Bhubaneswar Railway Station area (sample marker)','type':'railway station','lat':20.2688,'lon':85.8407},
        {'name':'Baramunda Bus Terminal area (sample marker)','type':'bus terminal','lat':20.2692,'lon':85.8081},
        {'name':'Biju Patnaik International Airport area (sample marker)','type':'airport','lat':20.2539,'lon':85.8178},
        {'name':'Capital Hospital area (sample marker)','type':'hospital','lat':20.2705,'lon':85.8274},
        {'name':'Pharmacy search area — Master Canteen (sample)','type':'pharmacy area','lat':20.2645,'lon':85.8420},
    ],
    'Cuttack': [
        {'name':'Cuttack Railway Station area (sample marker)','type':'railway station','lat':20.4625,'lon':85.8828},
        {'name':'Badambadi Bus Stand area (sample marker)','type':'bus terminal','lat':20.4627,'lon':85.8790},
        {'name':'SCB Medical College area (sample marker)','type':'hospital','lat':20.4620,'lon':85.8960},
        {'name':'Pharmacy search area — College Square (sample)','type':'pharmacy area','lat':20.4700,'lon':85.8860},
    ],
    'Puri': [
        {'name':'Puri Railway Station area (sample marker)','type':'railway station','lat':19.8074,'lon':85.8315},
        {'name':'Puri Bus Stand area (sample marker)','type':'bus terminal','lat':19.8060,'lon':85.8240},
        {'name':'District Headquarters Hospital area (sample marker)','type':'hospital','lat':19.8050,'lon':85.8180},
    ],
    'Kolkata': [
        {'name':'Howrah Railway Station area (sample marker)','type':'railway station','lat':22.5839,'lon':88.3428},
        {'name':'Kolkata Airport area (sample marker)','type':'airport','lat':22.6547,'lon':88.4467},
        {'name':'Esplanade Bus Terminal area (sample marker)','type':'bus terminal','lat':22.5650,'lon':88.3510},
        {'name':'Medical College Kolkata area (sample marker)','type':'hospital','lat':22.5726,'lon':88.3629},
    ],
    'Delhi': [
        {'name':'New Delhi Railway Station area (sample marker)','type':'railway station','lat':28.6430,'lon':77.2197},
        {'name':'Kashmere Gate ISBT area (sample marker)','type':'bus terminal','lat':28.6675,'lon':77.2280},
        {'name':'Indira Gandhi International Airport area (sample marker)','type':'airport','lat':28.5562,'lon':77.1000},
        {'name':'LNJP Hospital area (sample marker)','type':'hospital','lat':28.6392,'lon':77.2400},
    ],
    'Mumbai': [
        {'name':'Chhatrapati Shivaji Maharaj Terminus area (sample marker)','type':'railway station','lat':18.9402,'lon':72.8356},
        {'name':'Mumbai Central Bus Depot area (sample marker)','type':'bus terminal','lat':18.9690,'lon':72.8205},
        {'name':'Chhatrapati Shivaji Maharaj International Airport area (sample marker)','type':'airport','lat':19.0896,'lon':72.8656},
        {'name':'JJ Hospital area (sample marker)','type':'hospital','lat':18.9633,'lon':72.8355},
    ],
    'Bengaluru': [
        {'name':'KSR Bengaluru City Railway Station area (sample marker)','type':'railway station','lat':12.9774,'lon':77.5690},
        {'name':'Kempegowda Bus Station area (sample marker)','type':'bus terminal','lat':12.9773,'lon':77.5713},
        {'name':'Kempegowda International Airport area (sample marker)','type':'airport','lat':13.1986,'lon':77.7066},
        {'name':'Victoria Hospital area (sample marker)','type':'hospital','lat':12.9634,'lon':77.5740},
    ],
    'Hyderabad': [
        {'name':'Secunderabad Railway Station area (sample marker)','type':'railway station','lat':17.4344,'lon':78.5013},
        {'name':'Mahatma Gandhi Bus Station area (sample marker)','type':'bus terminal','lat':17.3783,'lon':78.4804},
        {'name':'Rajiv Gandhi International Airport area (sample marker)','type':'airport','lat':17.2403,'lon':78.4294},
        {'name':'Osmania General Hospital area (sample marker)','type':'hospital','lat':17.3713,'lon':78.4804},
    ],
}

def fallback_nearby(lat: float, lon: float):
    """Return approximate bundled sample markers near a supported city."""
    # Use the database city catalogue so fallback coverage follows installed city records.
    c=connect_db(); cities=c.execute('SELECT name,lat,lon FROM cities').fetchall(); c.close()
    if not cities:
        return []
    nearest=min(cities,key=lambda r:haversine(lat,lon,r['lat'],r['lon']))
    if haversine(lat,lon,nearest['lat'],nearest['lon']) > 100:
        return []
    city_name=nearest['name']
    # Exact curated sets where available; for other catalogue cities, return clearly-labelled city-level reference markers.
    base=FALLBACK_PLACES.get(city_name)
    if not base:
        base=[
            {'name':f'{city_name} city-centre reference point (sample)','type':'city reference','lat':nearest['lat'],'lon':nearest['lon']},
            {'name':f'{city_name} transport hub search area (sample)','type':'transport hub area','lat':nearest['lat']+0.012,'lon':nearest['lon']+0.012},
            {'name':f'{city_name} essential-services search area (sample)','type':'essential services area','lat':nearest['lat']-0.010,'lon':nearest['lon']+0.009},
        ]
    results=[]
    for item in base:
        dist=haversine(lat,lon,item['lat'],item['lon'])
        if dist <= 35:
            results.append({**item,'distance_km':round(dist,2),'source':'Bundled demonstration dataset','is_fallback':True})
    return sorted(results,key=lambda x:x['distance_km'])

@app.get('/api/nearby')
def nearby(city:Optional[str]=None,lat:Optional[float]=None,lon:Optional[float]=None,user=Depends(current_user)):
    if lat is None or lon is None:
        if not city:raise HTTPException(400,'Provide a city or coordinates.')
        c=connect_db(); target=c.execute('SELECT name,lat,lon FROM cities WHERE lower(name)=?',(norm(city),)).fetchone(); rows=c.execute('SELECT name,lat,lon FROM cities').fetchall(); c.close()
        if not target:raise HTTPException(404,'City not found. Search for a place first.')
        return {'origin':target['name'],'nearby_cities':sorted([{'city':r['name'],'distance_km':round(haversine(target['lat'],target['lon'],r['lat'],r['lon']),1)} for r in rows if r['name']!=target['name']],key=lambda x:x['distance_km'])[:8],'source':'Bundled sample city coordinates','notice':'Straight-line distances, not road distances.'}
    try:
        query=f'[out:json][timeout:15];(node(around:{5000},{lat},{lon})[amenity~"hospital|pharmacy"];way(around:{5000},{lat},{lon})[amenity~"hospital|pharmacy"];node(around:{5000},{lat},{lon})[railway=station];way(around:{5000},{lat},{lon})[railway=station];node(around:{5000},{lat},{lon})[highway=bus_stop];node(around:{5000},{lat},{lon})[aeroway=aerodrome];);out center 50;'
        res=requests.post('https://overpass-api.de/api/interpreter',data={'data':query},headers={'User-Agent':'EasyToConnectStudentProject/1.1 (educational demo)'},timeout=12); res.raise_for_status(); data=res.json()
        places=[]
        for x in data.get('elements',[]):
            tags=x.get('tags',{}); name=tags.get('name') or tags.get('amenity') or tags.get('railway') or tags.get('highway') or tags.get('aeroway') or 'Nearby place'; center=x.get('center',{}); plat=x.get('lat',center.get('lat')); plon=x.get('lon',center.get('lon'))
            if plat is not None and plon is not None:
                places.append({'name':name,'type':tags.get('amenity') or tags.get('railway') or tags.get('highway') or tags.get('aeroway') or 'place','lat':plat,'lon':plon,'distance_km':round(haversine(lat,lon,plat,plon),2),'source':'OpenStreetMap contributors','is_fallback':False})
        if places:
            return {'places':sorted(places,key=lambda x:x['distance_km']),'source':'OpenStreetMap contributors','is_fallback':False,'notice':'Live mapped results. Coverage and opening status are not guaranteed. © OpenStreetMap contributors.'}
    except Exception:
        # Public community endpoints can be busy or unavailable; keep the feature usable offline.
        pass
    sample=fallback_nearby(lat,lon)
    if sample:
        return {'places':sample,'source':'Bundled demonstration dataset','is_fallback':True,'notice':'Live map data could not be reached. These approximate sample markers are for demonstration only; verify real addresses before travelling.'}
    return {'places':[],'source':'No bundled coverage for this area','is_fallback':True,'notice':'Live map data is unavailable and this location is outside the bundled sample coverage. Try searching a supported Indian city.'}

@app.post('/api/plans')
def save_plan(req:SavePlan,user=Depends(current_user)):
    import json
    c=connect_db(); cur=c.execute('INSERT INTO saved_plans(user_id,origin,destination,option_json,reason,created_at) VALUES(?,?,?,?,?,?)',(user['id'],req.origin,req.destination,json.dumps(req.option),req.reason,datetime.now(timezone.utc).isoformat())); pid=cur.lastrowid; c.commit(); c.close(); return {'plan_id':pid,'status':'saved'}
@app.get('/api/plans')
def plans(user=Depends(current_user)):
    import json
    c=connect_db(); rows=c.execute('SELECT * FROM saved_plans WHERE user_id=? ORDER BY id DESC LIMIT 50',(user['id'],)).fetchall(); c.close(); return [{'id':r['id'],'origin':r['origin'],'destination':r['destination'],'option':json.loads(r['option_json']),'reason':r['reason'],'created_at':r['created_at']} for r in rows]
@app.delete('/api/plans/{plan_id}')
def delete_plan(plan_id:int,user=Depends(current_user)):
    c=connect_db(); cur=c.execute('DELETE FROM saved_plans WHERE id=? AND user_id=?',(plan_id,user['id'])); c.commit(); c.close()
    if not cur.rowcount:raise HTTPException(404,'Saved plan not found.')
    return {'status':'deleted'}
@app.post('/api/budget')
def add_budget(req:BudgetEntry,user=Depends(current_user)):
    c=connect_db(); cur=c.execute('INSERT INTO budget_entries(user_id,title,category,amount_inr,journey_id,created_at) VALUES(?,?,?,?,?,?)',(user['id'],req.title,req.category,req.amount_inr,req.journey_id,datetime.now(timezone.utc).isoformat())); c.commit(); eid=cur.lastrowid; c.close(); return {'id':eid,'status':'saved'}
@app.get('/api/budget')
def budget(user=Depends(current_user)):
    c=connect_db(); rows=c.execute('SELECT id,title,category,amount_inr,journey_id,created_at FROM budget_entries WHERE user_id=? ORDER BY id DESC LIMIT 100',(user['id'],)).fetchall(); total=c.execute('SELECT COALESCE(SUM(amount_inr),0) FROM budget_entries WHERE user_id=?',(user['id'],)).fetchone()[0]; c.close(); return {'entries':[dict(r) for r in rows],'total_inr':round(total,2)}
@app.delete('/api/budget/{entry_id}')
def delete_budget(entry_id:int,user=Depends(current_user)):
    c=connect_db(); cur=c.execute('DELETE FROM budget_entries WHERE id=? AND user_id=?',(entry_id,user['id'])); c.commit(); c.close()
    if not cur.rowcount:raise HTTPException(404,'Expense not found.')
    return {'status':'deleted'}
@app.get('/api/model-info')
def model_info(user=Depends(current_user)):
    if MODEL is None or not MODEL_METRICS:
        raise HTTPException(503,'Prediction model is not ready. Restart the app and check scikit-learn installation.')
    return {'status':'ready',**MODEL_METRICS}

@app.post('/api/predict')
def predict_delay(req:PredictionRequest,user=Depends(current_user)):
    if MODEL is None:
        raise HTTPException(503,'Prediction model is not ready. Run: python -m pip install -r requirements.txt, then restart the app.')
    mode_code={'bus':0,'train':1,'flight':2}[req.mode]
    features=[[req.distance_km,mode_code,req.transfers,req.departure_hour,req.weather_factor]]
    prediction=max(0.0,float(MODEL.predict(features)[0]))
    # Tree spread is a rough model-variation indicator, not a statistical confidence interval.
    tree_predictions=np.array([tree.predict(np.asarray(features))[0] for tree in MODEL.estimators_],dtype=float)
    spread=float(np.std(tree_predictions))
    risk='Low' if prediction<10 else ('Moderate' if prediction<25 else 'High')
    return {'predicted_delay_minutes':round(prediction,1),'delay_risk':risk,'model_variation_minutes':round(spread,1),'input':{'distance_km':req.distance_km,'mode':req.mode,'transfers':req.transfers,'departure_hour':req.departure_hour,'weather_factor':req.weather_factor},'model_type':'RandomForestRegressor','data_status':'Educational demonstration trained on synthetic data. This is not a real-world delay forecast.'}
