# Face Recognition Attendance System  

A Python-based **Face Recognition Attendance System** that automates the process of logging attendance using real-time facial recognition. The system uses **OpenCV** and **face-recognition** libraries to detect and identify faces, and records login/logout events into CSV files.  

---

## 🔹 Features  
- Real-time face detection and recognition using a webcam.  
- Automated logging of **login** and **logout** times.  
- Data stored in CSV files for easy access and analysis.  
- Simple dataset creation and encoding process.  
- Built using **Python, OpenCV, dlib, face-recognition, pandas, imutils**.  

---

## 🔹 Project Workflow  
1. **Data Collection**  
   - Run `create_data.py` to capture face images via webcam and store them in a dataset folder.  

2. **Encoding**  
   - Run `encode.py` to convert face images into numerical encodings and save them in `encodings.pickle`.  

3. **Attendance System**  
   - Run `face.py` to start real-time recognition.  
   - Recognized faces are logged in `login.csv` and `logout.csv`.  

---

## 🔹 Installation  

### Prerequisites  
- Python 3.7+  
- pip3  
- A webcam  

### Setup (Mac/Linux/Windows)  
Clone the repository:  
```bash
git clone https://github.com/YOUR-USERNAME/face-recognition-attendance.git
cd face-recognition-attendance
```

Create and activate a virtual environment (recommended):  
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:  
```bash
pip3 install -r requirements.txt
```

---

## 🔹 Usage  

### Step 1: Collect Data  
```bash
python3 create_data.py
```

### Step 2: Encode Faces  
```bash
python3 encode.py
```

### Step 3: Run Attendance System  
```bash
python3 face.py
```

---

## 🔹 Output  
- **login.csv** → Logs when a person is first recognized.  
- **logout.csv** → Logs when the person leaves.  
- **encodings.pickle** → Stores trained face encodings.  

---

## 🔹 Tech Stack  
- **Python**  
- **OpenCV**  
- **dlib / face-recognition**  
- **NumPy, pandas, imutils**  

---

## 🔹 Future Enhancements  
- Integration with a web-based dashboard for attendance reports.  
- Database (MySQL/PostgreSQL) storage instead of CSV files.  
- Multi-camera support for larger environments.  

---

## 🔹 Author  
Developed by [Your Name](https://github.com/YOUR-USERNAME) ✨  
