import subprocess
import time

def start_sensor_simulation():
    print("Starting sensor simulation...")
    subprocess.Popen(["python", "edge/sim_data.py"])

def start_fog_device():
    print("Starting Fog device simulation...")
    subprocess.Popen(["python", "fog/fog_device.py"])

def start_dynamodb_writer():
    print("Starting DynamoDB writer...")
    subprocess.Popen(["python", "aws/dynamodb_writer.py"])

def start_dashboard():
    print("Starting Streamlit dashboard...")
    subprocess.Popen(["python", "-m", "streamlit", "run", "dashboard/streamlit_app.py"])

def main():
    # Start all processes in parallel
    start_sensor_simulation()
    start_dynamodb_writer()
    start_dashboard()
    start_fog_device()

    # Keep the main thread alive to allow background processes to continue
    try:
        while True:
            time.sleep(10)  # Sleep to keep the main process running
    except KeyboardInterrupt:
        print("Main process stopped.")

if __name__ == "__main__":
    main()
