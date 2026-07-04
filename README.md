# PronounceMate AI Backend 🎙️🤖

The backend engine for **PronounceMate**, an AI-powered real-time English pronunciation detection and improvement system designed specifically for language learners. This server handles secure authentication, session state management, and aggregates deep learning analytics for phoneme-level and audio classification pipelines.

---

## 🚀 Key Features Shipped
* **Token-Based Authentication:** Secure user sign-up and login pipelines managed via Django REST Framework (`rest_framework.authtoken`).
* **Granular Analytics Datastore:** Tracks user proficiency levels (`Beginner`, `Intermediate`, `Advanced`), targeted reference accents, evaluation high scores, and persistent history.
* **Relational Pipeline Schema:** Seamless relational persistence mapping individual `PhonemeError` instances (`substitution`, `omission`, `insertion`, `skipped`) directly back to their parent `PracticeSession`.
* **Dynamic Content Engine:** Supports system-shipped standard evaluation materials as well as custom multi-format document text uploading (`python-docx` integrated).

---

## 📁 Repository Structure

The core codebase is organized within an industry-standard full-stack layout:
* **`core/`** - Project configuration root housing global settings (`settings.py`), routing architectures (`urls.py`), and WSGI definitions.
* **`api/`** - The primary MVC application containing evaluation engines, relational endpoint models, and serialization rules.
    * `models.py` - Custom schema rules processing data flows for profiles, script items, practice logs, and error records.
    * `serializers.py` - Data parsers translating complex DB records to mobile-friendly JSON formats.
    * `views.py` / `urls.py` - System endpoint request controllers mapping REST requests safely.

---

## 🛠️ Tech Stack & Dependencies
* **Language/Framework:** Python 3.12, Django 5.x
* **API Wrapper:** Django REST Framework (DRF)
* **Cross-Origin Security:** `django-cors-headers` wired to handle physical mobile emulators and native/web execution.
* **Document Parsing:** `python-docx` & `lxml` for custom file intake.
* **Database Driver:** `psycopg2-binary` supporting connection routing directly to local/cloud PostgreSQL configurations.

---

## ⚙️ Local Installation & Setup

### 1. System Requirements & Clone
Ensure you are running Python 3.12+ on your host environment (Ubuntu 24.04 LTS recommended):
```bash
git clone [https://github.com/rafymemon/pronouncemate-backend.git](https://github.com/rafymemon/pronouncemate-backend.git)
cd pronouncemate-backend
```

### 2. Isolate Dependencies
```
python3 -m venv env
source env/bin/activate
```

### 3. Install Package Buffers
```
pip install --default-timeout=1000 -r requirements.txt
```

### 4. Database Schema Lifecycle & Migrations

```
python manage.py makemigrations
python manage.py migrate
```

### 5. Launch the Local Server Network Area
```
python manage.py runserver 0.0.0.0:8000
```

