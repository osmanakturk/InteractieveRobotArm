![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Expo](https://img.shields.io/badge/Expo-Mobile-black)
![Robotics](https://img.shields.io/badge/Robotics-xArm6-orange)


# InteractieveRobotArm

An interactive robotic manipulation system that combines a **mobile application**, a **Jetson-based gateway**, and an **AI server** to control an **xArm6 robot arm** with support for **manual control** and **AI-assisted pick-and-place** using a **RealSense depth camera** and **GG-CNN-based grasp prediction**.

```text
Mobile App
    │
    ▼
Jetson Gateway
    │
    ▼
AI Server
    │
    ▼
xArm6 Robot
```

---

## Academic Context

This project was developed as a **Final Work project** for the **Applied Computer Science program at Erasmushogeschool Brussel**.

The objective of the project was to explore how **mobile interfaces, robotics, and AI-based grasp detection** can be integrated into a modular robotic manipulation platform.

---

## Overview

This project explores how **artificial intelligence and robotics can be integrated** to make robotic arm interaction more accessible and practical in real-world environments.

The system is built around three main components:

- a **mobile application** for user interaction
- a **Jetson Gateway API** for hardware communication
- an **AI Server API** for grasp detection and pick-and-place logic

The current implementation supports:

- manual robot control from a mobile app
- live camera streaming
- RealSense RGB and depth input
- AI-assisted pick-and-place workflow
- modular communication between mobile, gateway, and AI server

---

## Key Features

- Mobile control interface for robotic manipulation
- AI-assisted grasp detection using GG-CNN
- RealSense RGB-D perception pipeline
- Modular architecture (Mobile – Gateway – AI Server)
- Real-time camera streaming
- Robot motion control through REST APIs
- Pick-and-place interaction from mobile UI

---

## Repository Structure

```
InteractieveRobotArm
│
├── mobile        # React Native mobile application
├── jetson        # Jetson Gateway API
├── ai_server     # AI processing server
└── README.md
```

---

## System Architecture

The system follows a distributed architecture composed of three main layers:

1. **Mobile Application**
   - user interface
   - connection management
   - manual control
   - pick-and-place interaction

2. **Jetson Gateway API**
   - connects to the xArm6 robot
   - manages the RealSense camera
   - exposes REST and WebSocket endpoints
   - proxies AI-related requests

3. **AI Server API**
   - manages AI modes
   - runs the pick-and-place worker
   - processes grasp selection
   - uses GG-CNN for grasp prediction

---

## GG-CNN Based Grasp Detection

This project uses a GG-CNN (Generative Grasping Convolutional Neural Network) based grasp detection pipeline for robotic manipulation.

The implementation used in this project is based on the following repository:

**UFactory Vision System**

Repository:
```
https://github.com/xArm-Developer/ufactory_vision
```
This repository provides an example pipeline for integrating computer vision and grasp detection with xArm robotic manipulators.

The system combines:
- depth camera input
- neural network based grasp detection
- robotic arm motion execution



### What is GG-CNN?

GG-CNN (Generative Grasping Convolutional Neural Network) is a deep learning model designed for real-time robotic grasp detection.

Instead of detecting objects first and then planning a grasp, GG-CNN directly predicts grasp parameters for each pixel in the depth image.

The network produces three main maps:

| **Output** | **Description** |
| ------------- | ------------- |
| Grasp Quality  | Probability that a grasp is successful  |
| Grasp Angle  | Rotation angle of the gripper  |
| Gripper Width  | Required opening width of the gripper  |

From these maps the system selects the best grasp candidate, which is then converted into a robot pose.

This allows robots to perform grasp planning directly from depth images without explicit object segmentation.



### How GG-CNN Is Used in This Project

In this project, the GG-CNN based approach from the ufactory_vision repository was adapted and integrated into the AI server architecture.

The pipeline works as follows:
   - The RealSense camera captures RGB and depth images.
   - The depth image is processed by the AI server using the GG-CNN model to estimate potential grasp poses.
   - A cropped region around the selected target point is extracted.
   - The GG-CNN model predicts grasp parameters.
   - The predicted grasp is transformed into the robot coordinate system.
   - The Jetson Gateway sends the motion command to the xArm6 robot arm.
   - The robot executes the pick operation.

This integration allows the system to perform AI-assisted pick-and-place operations directly from the mobile application interface.



### Adaptations Made in This Project

While the ufactory_vision repository provided the base implementation, several modifications were required to integrate it into this project:
   - integration with a FastAPI based AI server
   - communication with the Jetson Gateway API
   - support for mobile-based interaction
   - modular pick-and-place workflow management
   - coordinate transformation between camera space and robot space
   - asynchronous execution of grasp tasks

These modifications allow the system to operate within a distributed architecture consisting of a mobile application, a Jetson gateway, and an AI server.



### Acknowledgment

This project builds upon the open-source work provided by UFactory and the contributors of the ufactory_vision repository.

Their implementation demonstrates how neural network based grasp detection can be integrated with the xArm robotic platform, which served as the foundation for the AI-assisted manipulation system developed in this project.

GG-CNN paper
```
- Morrison, D., Corke, P., & Leitner, J. (2018). Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach.
- Robotics: Science and Systems (RSS).
```

UFactory Vision repo
```
https://github.com/xArm-Developer/ufactory_vision
```


---

## Architecture Diagram

<img width="1460" height="505" alt="System Architecture" src="https://github.com/user-attachments/assets/6fcea69f-2934-47ec-bc0d-89ec29b7a186" />

---

## Connection Flow

<img width="1765" height="2357" alt="Connection Hub Flow" src="https://github.com/user-attachments/assets/8c5b8d66-d128-4089-a20a-6ddafeb5211c" />

---

## Prerequisites

Before running the system ensure the following are available:

**Hardware:**
- xArm6 robot arm
- Intel RealSense depth camera
- Jetson Orin Nano
- AI processing computer

**Software:**
- Python 3.10+
- Node.js
- npm
- Expo CLI
- Intel RealSense SDK
- xArm Python SDK

---

## Installation

1. Clone the repository
   ```
   git clone https://github.com/osmanakturk/InteractieveRobotArm.git
   cd InteractieveRobotArm
   ```
   
2. Mobile Application Setup

   Navigate to the mobile folder:
   ```
   cd mobile
   npm install
   ```
   
   Start the mobile application:
   ```
   npm run start
   ```

   APK Installation (Prebuilt)
   
   A prebuilt Android APK is included in this repository. The APK is provided for demonstration purposes and allows testing the mobile interface without installing the development environment.
   ```
   InteractieveRobotArm.apk.zip
   ```
   Steps:

      - Download the zip file
      - Extract the archive
      - Install the APK on an Android device
      - Enable “Install from unknown sources” if required

      This allows testing the mobile application without building the project locally.


3. Jetson Gateway Setup

   Create a virtual environment:
   ```
   cd jetson
   python -m venv venv
   source venv/bin/activate
   ```
   Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Run the gateway:
   ```
   python main.py
   ```

4. AI Server Setup

   Create a virtual environment:
   ```
   cd ai_server
   python -m venv venv
   source venv/bin/activate
   ```
   Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Run the AI server:
   ```
   python main.py
   ```

---

## Mobile Application Workflow

### Splash Screen

The application starts with a splash screen before navigating to the connection hub.

<img width="2400" height="1080" alt="Screenshot_1772634664" src="https://github.com/user-attachments/assets/6727c771-bdf2-46e9-adab-20d478c72c64" />


### Connection Hub

Users can configure connections to:
- Gateway
- Robot
- AI Server

<img width="2400" height="1080" alt="Screenshot_1772630735" src="https://github.com/user-attachments/assets/daf24821-7ff3-4ad1-91c6-b14096bf9618" />


### Mode Selection

Available modes:
- Manual Control
- Pick & Place
- Voice Control (future work)

<img width="2400" height="1080" alt="Screenshot_1772630853" src="https://github.com/user-attachments/assets/6a4bb1af-7fc9-474e-a5c5-a7e11efc063d" />


### Manual Control

Manual control allows the user to:
- enable or disable the robot
- jog robot motion
- control the gripper
- change speed settings
- view the camera stream
- stop robot movement


<img width="1726" height="786" alt="Screenshot 2026-03-08 at 06 41 39" src="https://github.com/user-attachments/assets/29091470-c721-4f80-8f1b-f52bf5d2742a" />



https://github.com/user-attachments/assets/f8771318-7903-4559-bef1-1ef33642d4a1

### Pick & Place

Pick-and-place mode allows:
- camera visualization
- object selection on the camera image
- AI-based grasp planning
- automated robot motion execution


<img width="1726" height="786" alt="Screenshot 2026-03-08 at 06 42 18" src="https://github.com/user-attachments/assets/6257ef36-e5a3-46b5-8758-c8de7559b1c1" />



https://github.com/user-attachments/assets/3ab10a33-9425-4673-a4ec-abe4331b8255



https://github.com/user-attachments/assets/f80cc5ea-600c-4894-8404-39dd5b1a89a9


---


## API Overview

The system exposes two REST APIs:

- **Jetson Gateway API**: responsible for robot control, camera management, and communication with the AI server.
- **AI Server API**: responsible for AI-based grasp detection and pick-and-place task management.



### Jetson Gateway API

Main responsibilities:
   - robot control
   - camera streaming
   - AI proxy communication
   - system monitoring



### AI Server API

Responsibilities:
   - AI mode management
   - grasp selection
   - pick-and-place execution


---


## Current Limitations

   - Voice control is currently a placeholder
   - Some dependencies require manual installation
   - Platform-specific configuration may be required for RealSense
   - Deployment scripts are not yet automated

---


## Future Work

Future improvements may include:
   - full voice control integration
   - improved AI grasp planning
   - containerized deployment
   - enhanced user interface
   - additional robotic manipulation modes

---


## Disclaimer

This project is an experimental robotics platform.
Use caution when operating physical robotic hardware.

Always ensure the robot is operated in a safe environment.


---


## License

This project is released for **academic and research purposes**.

The software is provided as-is without warranty and is intended for educational and experimental robotics use.


---

## Author

Osman Aktürk  
Applied Computer Science  
Erasmushogeschool Brussel

---

