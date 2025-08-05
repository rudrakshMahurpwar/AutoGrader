
# Python Installation

Download Python 3.9.13 from the official archive:

```yaml
https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe
```

### Run the installer

Tick the checkbox "Add Python 3.9 to PATH"

Click "Next" → "Install"

### Verify installation (Optional)

```cmd
python --version
```


## Run the Project

### Clone the Repository

```bash
git clone https://github.com/rudrakshMahurpwar/AutoGrader.git
```

### Move inside the Project Directory

```cmd
cd AutoGrader
```

### Create a Virtual Environment

```cmd
python -m venv venv
```

```cmd
venv\Scripts\activate
```

You should now see (venv) at the beginning of your command prompt line.

### Install Required Libraries

```cmd
pip install sentence-transformers torch
```

### Run the Project
```cmd
python main.py
```

You should now see the expected output as in *LLM_OUTPUT.txt*.