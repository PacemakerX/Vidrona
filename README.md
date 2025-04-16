<img src="./assests/image.png" alt="LinkedIn Automation" style="width:100%; height:300px; object-fit: cover;">
<br><br>


# Vidrona - IoT-Based Utility Monitoring System

Vidrona is an AI-powered utility monitoring system that uses IoT devices, Fog and Edge computing, and cloud services to monitor and analyze real-time data from various sensors. The system integrates AWS services for cloud storage, we can use Kafka for stream processing, and Streamlit for visualization on a web dashboard.

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazonaws&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#))
[![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=fff)](#)
[![Visual Studio Code](https://custom-icon-badges.demolab.com/badge/Visual%20Studio%20Code-0078d7.svg?logo=vsc&logoColor=white)](#)

# Who are We?
[![Watch the video](https://img.youtube.com/vi/yPIS7ShKdms/0.jpg)](https://www.youtube.com/watch?v=yPIS7ShKdms&autoplay=1)

## 🚀 Features

- 🌐 **Real-Time Monitoring**: Continuously monitor environmental data from IoT sensors (e.g., temperature, humidity).
- 🌩️ **Edge/Fog Processing**: Process sensor data locally on Edge/Fog nodes to reduce latency.
- ☁️ **Cloud Integration**: Leverage AWS services (DynamoDB, Lambda, etc.) for cloud storage and processing.
- 📊 **Dashboard Visualization**: Visualize real-time sensor data using a Streamlit dashboard.
- 🔒 **Data Security**: Secure communication and storage using AWS IAM roles and encryption.

## 🛠️ Tech Stack

### Core Technologies

- **Python**: The main programming language for automation and integration.
- **AWS**: Utilizes various AWS services including DynamoDB, Lambda, and EC2 for cloud computing.
- **MQTT**: Messaging protocol for communication between IoT devices and cloud.
- **Streamlit**: Framework for building interactive dashboards to visualize real-time data.
- **paho-mqtt**: MQTT client for communication between Edge/Fog and Cloud.

## 📋 Installation

```bash
# Clone the repository
git clone https://github.com/PacemakerX/Vidrona.git
cd Vidrona

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure aws configuration
cp .aws_config.json.example aws_config.json
# Edit .env with your API keys and settings

```

## 🔧 Configuration

### Requirements.txt

```
boto3==1.26.13
paho-mqtt==1.6.1
streamlit==1.10.0
pandas==1.4.2
matplotlib==3.5.2
seaborn==0.11.2
scikit-learn==1.1.0
numpy==1.23.1
pytz==2022.1
```

## 📝 Usage

```bash
# Run the main file
python main.py

```

## Link to the Repository:

<p align="center">
<a href="https://github.com/PacemakerX/Vidrona.git">
  <img src="https://user-images.githubusercontent.com/74038190/212748842-9fcbad5b-6173-4175-8a61-521f3dbb7514.gif" alt="Website Link" style="width:100%; height:300px; object-fit: cover;">
</a>
<p>

## Feel free to connect with me!

<p align="center">
  <a href="mailto:sparsh.officialwork@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-sparsh.officialwork@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail Badge" />
  </a>
  <a href="https://www.linkedin.com/in/sparshsoni">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge" />
  </a>
</p>
