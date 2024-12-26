# License Plate Detection System

A real-time license plate detection and recognition system built with Python, Flask, YOLO, and PaddleOCR. This web application processes video input to detect and recognize license plates, storing the results in a MySQL database.

## 🚀 Features

- Real-time license plate detection using YOLO
- Optical Character Recognition (OCR) using PaddleOCR
- Web interface for video upload and processing
- MySQL database integration for storing detections
- Detection area configuration
- Real-time detection counter
- Confidence score display
- License plate crop visualization

## 🛠️ Technologies Used

- **Backend:**
  - Python 3.x
  - Flask (Web Framework)
  - OpenCV (Video Processing)
  - Ultralytics YOLO (Object Detection)
  - PaddleOCR (Text Recognition)
  - MySQL (Database)

- **Frontend:**
  - HTML5
  - CSS3
  - JavaScript

## 📋 Prerequisites

- Python 3.x
- MySQL Server
- CUDA-capable GPU (recommended for better performance)

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/license-plate-detection.git
   cd license-plate-detection
   ```

2. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MySQL database:**
   - Install MySQL Server
   - Create a database named 'numberplate'
   - Update database credentials in `server.py` if needed

4. **Download YOLO model:**
   - Place your trained YOLO model (`best.pt`) in the project root directory

## 🔧 Configuration

### Database Configuration
In `server.py`, update MySQL connection parameters if needed:
```python
host = "127.0.0.1"
user = "root"
password = ""
database = "numberplate"
port = 3306
```

### Detection Area
In `numberplate.py`, modify the detection area coordinates if needed:
```python
area = [(5, 180), (3, 249), (984, 237), (950, 168)]
```

## 🚀 Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   - Open your browser and go to `http://localhost:5000`
   - Upload a video file for processing
   - View real-time detections and results

## 📁 Project Structure

- `app.py`: Main Flask application and route handlers
- `numberplate.py`: License plate detection and OCR processing
- `server.py`: Database operations and management
- `templates/index.html`: Web interface template
- `static/css/style.css`: CSS styling
- `requirements.txt`: Python dependencies

## 🎯 Key Components

### Video Processing (`numberplate.py`)
- Implements YOLO-based license plate detection
- Performs OCR on detected plates
- Handles real-time video processing
- Manages detection visualization

### Database Management (`server.py`)
- Handles MySQL database operations
- Stores detection results with timestamps
- Manages data retrieval for display

### Web Interface (`app.py` & `index.html`)
- Provides user interface for video upload
- Displays detection results
- Shows real-time processing status

## 📊 Features in Detail

1. **License Plate Detection:**
   - Uses YOLO model for accurate plate detection
   - Configurable detection area
   - Real-time tracking with unique IDs

2. **OCR Processing:**
   - Text extraction from detected plates
   - Confidence score calculation
   - Text cleaning and formatting

3. **Database Storage:**
   - Automatic database and table creation
   - Stores plate numbers with timestamps
   - Maintains detection history

4. **Visualization:**
   - Real-time video display
   - Detection boxes and text overlay
   - Confidence score display
   - Detection counter
   - Plate crop visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- PaddleOCR by PaddlePaddle
- Flask framework
- OpenCV community

## 📧 Contact

For any queries or suggestions, please reach out to [your-email@example.com]

## 🐛 Known Issues

- Video processing speed may vary based on hardware capabilities
- OCR accuracy depends on image quality and lighting conditions

## 🔜 Future Improvements

- Add user authentication
- Implement real-time statistics dashboard
- Add export functionality for detection data
- Enhance OCR accuracy with pre-processing
- Add support for multiple camera streams
