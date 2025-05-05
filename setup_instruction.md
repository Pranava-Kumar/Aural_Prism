### Setup Instructions for Streamlit App (Python 3.11)

#### 1. Install Python 3.11

* Download Python 3.11 from the official website: [https://www.python.org/downloads/release](https://www.python.org/downloads/release)
* During installation, **make sure to check the box** that says:

  ```
  Add Python 3.11 to PATH
  ```

#### 2. Verify Python Installation

Open a terminal or command prompt and run:

```
python --version
```

You should see:

```
Python 3.11.x
```

#### 3. Install Required Python Packages

Make sure you're in the project directory (where `requirements.txt` is located), then run:

```
music.bat
```

This will:

* Install all necessary packages listed in `requirements.txt`
* Start the Streamlit app

#### 4. Run the Streamlit App Manually (Optional)

If you want to run the commands manually, open a terminal in the project folder and run:

```
pip install -r requirements.txt -q
python -m streamlit run app.py
```

#### 5. Open the App

After running the app, your default browser should automatically open at:

```
http://localhost:8501
```

---
