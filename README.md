# InteractieveRobotArm

An interactive robotic manipulation system that combines a **mobile application**, a **Jetson-based gateway**, and an **AI server** to control an **xArm6 robot arm** with support for **manual control** and **AI-assisted pick-and-place** using a **RealSense depth camera** and **GG-CNN-based grasp prediction**.

---

## Overview

This project explores how **AI and robotics can be combined** to make robotic arm interaction more accessible and practical in real-world environments.

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

## System Architecture

The architecture is distributed across three layers:

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