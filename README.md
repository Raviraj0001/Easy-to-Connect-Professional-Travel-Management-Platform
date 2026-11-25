# Easy to Connect — Personal Travel Workspace

A professional, interview-focused travel assistance application built with **HTML, CSS, JavaScript, Python, FastAPI, SQLite, and scikit-learn**. The project is designed to demonstrate multiple related features instead of a single fare-comparison screen.

## Modules

1. **Smart Journey Planner** — compares illustrative bus/train/flight estimates and ranks them by fastest, cheapest, or balanced preference. Supports city names from the bundled catalogue and locations searched through OpenStreetMap.
2. **Emergency Travel Assistant** — an urgency-first workflow, departure checklist, backup-route reminder, and recommendation explanation. It is not an emergency dispatch service.
3. **Travel Budget Manager** — create and delete expense entries by category; see total logged spending. Expenses are private to the signed-in account.
4. **Nearby Places and Map** — Leaflet map, browser geolocation, and a user-triggered nearby search for mapped bus stops, railway stations, airports, hospitals, and pharmacies.
5. **ML Travel Predictor** — a scikit-learn Random Forest model and an evaluation panel explaining MAE, R², test-set size, features, and limitations.
6. **Personal Workspace** — registration, email/password login, optional verified Gmail sign-in through Google Identity Services, logout, one-year persistent session cookie, saved journeys, per-user data separation, and a dashboard with trip and expense summaries.
7. **Booking shortcuts** — colour-coded bus, train and flight links open redBus, IRCTC and Google Flights in a new tab. Booking is completed on those provider websites; this app does not purchase tickets.

## Project layout

- `app.py` — FastAPI routes, database access, account/session logic, travel estimates, nearby-data fallback and ML endpoints. Related routes are grouped by feature.
- `static/index.html` — page structure and module screens.
- `static/styles.css` — all visual styling, responsive layout and booking-card colours.
- `static/app.js` — browser interactions, API calls, Google sign-in UI, maps, forms and dashboard updates.
- `tests/test_app.py` — automated API, prediction, fallback and per-user privacy tests.
- `requirements.txt` — Python dependencies.

The browser presentation is intentionally split into separate HTML, CSS and JavaScript files so each part is easier to discuss and maintain in an interview.

## Run on Windows / VS Code

1. Extract the ZIP and open the `easy_to_connect` folder in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead.
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```
5. (Optional) Configure Google sign-in as described in the next section.
6. Start the app:

   ```powershell
   python -m uvicorn app:app --reload
   ```
7. Open `http://127.0.0.1:8000`. Create a test account and sign in. API docs are at `http://127.0.0.1:8000/docs`.

Use `Ctrl+C` in the terminal to stop the server. The SQLite database `easy_to_connect.db` is created automatically and is ignored by Git.

## Run with Docker

```bash
docker compose up --build
```

Open `http://127.0.0.1:8000`. For data persistence across container rebuilds, add a Docker volume for `/app/easy_to_connect.db` before using it beyond a demo.

## Test

Install the app requirements, then run:

```bash
python -m pip install pytest httpx
python -m pytest -q
```


## Optional Gmail / Google sign-in setup

The login screen supports Google Identity Services for verified `@gmail.com` accounts. It is a real Google sign-in flow, not a form that merely accepts a Gmail-looking address. For security, Google must verify the credential on the backend. Email/password login remains available if you have not configured Google yet.

1. Open the Google Cloud Console: https://console.cloud.google.com/ and create/select a project.
2. Configure the OAuth consent screen / Google Auth Platform and create an **OAuth Client ID** of type **Web application**.
3. Add these Authorized JavaScript origins for local development: `http://127.0.0.1:8000` and `http://localhost:8000`.
4. In the VS Code PowerShell terminal, set the client ID before starting the server (replace the example with your own public client ID):

   ```powershell
   $env:GOOGLE_CLIENT_ID = "YOUR_CLIENT_ID.apps.googleusercontent.com"
   python -m uvicorn app:app --reload
   ```

   Keep the variable set in the same terminal where Uvicorn runs. The client ID is public configuration; never place a client secret in frontend code. The app verifies the Google ID token on the backend and only accepts a verified Gmail address.
5. Refresh the page. The official Google button appears once the client ID is configured.

## Per-account data separation

Every saved journey and expense row stores a `user_id`. All list, delete and summary queries filter by the authenticated session's user ID. A user cannot read or delete another account's journey or expense by changing an ID in the URL. The tests include two independent accounts to verify that their plans and expenses stay separate. The database file is local to this project folder; do not share the database file if it contains personal data.

## Booking links

The dashboard includes colour-coded shortcuts: redBus (`https://www.redbus.in/`), IRCTC (`https://www.irctc.co.in/nget/train-search`) and Google Flights (`https://www.google.com/travel/flights`). These are external links only; the app does not access account credentials, book tickets, or claim live inventory. Verify details on the provider website.

## Data sources and API requirements

### OpenStreetMap Nominatim (place search)

- Endpoint used: `https://nominatim.openstreetmap.org/search`.
- Search only runs after the user clicks **Search map**; it is not autocomplete.
- Requests identify this educational app with a User-Agent and are throttled to approximately one request per second per running process.
- Attribution: OpenStreetMap contributors.
- Public Nominatim is a limited-capacity community service. Follow its [usage policy](https://operations.osmfoundation.org/policies/nominatim/), cache where practical, and replace it with a suitable provider or self-hosted instance before a larger deployment.

### OpenStreetMap Overpass (nearby places)

- Endpoint used: `https://overpass-api.de/api/interpreter`.
- User-triggered searches request nearby mapped transport stops and selected essential services. Public endpoints can be busy or temporarily unavailable. When that happens, the app now uses a bundled fallback dataset of approximate sample markers around supported cities (including Bhubaneswar, Cuttack, Puri, Kolkata, Delhi, Mumbai, Bengaluru and Hyderabad). These fallback markers are clearly labelled as samples and must not be used as verified navigation or emergency-service information. For other catalogue cities, the app may show generic city-reference markers, also labelled as samples.
- Attribution: OpenStreetMap contributors. Data coverage and opening status are not guaranteed.
- For a production deployment, use an approved, monitored provider or self-hosted dataset and implement caching/rate limits.

### Public transport feeds

There is **no authorized live transport provider API configured** in this project. Journey fares, durations, departure/arrival fields, and delay estimates are calculated illustrative estimates. They must not be presented as real schedules, quotes, seat availability, or live tracking.

A future live integration should use an operator or transport agency feed with documented permission. GTFS Schedule can provide static agency routes, stops and scheduled service; GTFS Realtime can provide trip updates, service alerts and vehicle positions **only when a relevant agency publishes an accessible feed**. See [GTFS overview](https://gtfs.org/documentation/overview/). Never scrape ticketing sites or claim a feed exists without verifying its availability and terms.

### Map tiles

The browser uses Leaflet with standard OpenStreetMap tiles and visible attribution. Public tile servers have a usage policy and are not a guaranteed production CDN. Review the [tile usage policy](https://operations.osmfoundation.org/policies/tiles/) before deployment at scale.

## ML design and limitations

The model uses `RandomForestRegressor` with distance, transport mode, transfers, departure hour and a simulated weather/traffic factor. It trains on generated synthetic data on app startup. The reported metrics describe a holdout split of that synthetic dataset only; they are **not evidence of real-world prediction accuracy**. To make a defensible real predictor, obtain a licensed dataset with scheduled and actual arrival times, build a leakage-safe time-based split, compare to a baseline, and report errors by route/mode/time period.

## Security and production caveats

- Passwords are salted and derived with PBKDF2-HMAC-SHA256; raw passwords are never stored.
- Session IDs are random opaque tokens; only their SHA-256 hashes are stored in SQLite. The browser cookie is HttpOnly and SameSite=Lax, and expires after one year or on logout.
- The local-development cookie is configured without `Secure` so it works on `http://127.0.0.1`. Set `Secure` when deployed behind HTTPS, add CSRF protection for state-changing routes, rate-limit authentication, add password reset/email verification, and use a managed secret/configuration setup before public deployment.
- SQLite is appropriate for a learning/demo app. Consider PostgreSQL for a multi-user deployment.
- User location is accessed only after browser permission. The app sends requested search text or coordinates to OpenStreetMap services for the selected functionality.

## API overview

- `GET /api/health`, `GET /api/cities`
- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/google/config`, `POST /api/auth/google`, `POST /api/auth/logout`, `GET /api/auth/me`
- `GET /api/places/search?q=...`
- `GET /api/compare?...` (authenticated)
- `GET /api/nearby?...` (authenticated)
- `GET/POST /api/plans`, `DELETE /api/plans/{plan_id}` (authenticated)
- `GET/POST /api/budget`, `DELETE /api/budget/{entry_id}` (authenticated)
- `GET /api/model-info` (authenticated)
- `POST /api/predict` (authenticated; interactive prediction form in the ML Predictor page)

## Interview walkthrough

A clear demonstration sequence is: register/login → search for a boarding and destination point → compare estimates using different priorities → try the urgent-travel checklist → save an itinerary → add travel expenses → inspect the map and nearby places → explain the ML features and synthetic-data limitation → log out and verify that protected endpoints require authentication.


## Troubleshooting: Nearby Places and ML Predictor

### Nearby places show a temporary-unavailable message
The app first tries OpenStreetMap Overpass. If the public endpoint is offline, overloaded, or returns no mapped results, the backend now returns a bundled sample dataset for supported city areas instead of failing. Sample markers are approximate educational data, not confirmed business addresses, opening hours, or emergency service availability. Live OpenStreetMap results include attribution. The public OSM services are community resources with usage limits and no uptime guarantee; see https://operations.osmfoundation.org/policies/nominatim/ and https://openstreetmap.org/copyright.

### ML Predictor page does not predict
The Predictor page now includes an input form that calls `POST /api/predict`. The model is trained in memory when `app.py` starts. From the project folder, activate the virtual environment and run:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Then log in again if needed, open **ML predictor**, enter a distance/mode/transfers/departure hour/weather factor and select **Run prediction**. If the model status is unavailable, look at the PowerShell terminal for the original Python import/startup error. The demo model uses synthetic data and must not be described as a validated real-world delay forecast.
