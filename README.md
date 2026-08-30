**To Start Project do the following Steps:**
1. Connect ESP32 CAM with the Desktop/ Laptop
2. Open the **CameraWebServer,ino** Code
3. Change the Wifi Name, Password to Connected Wifi
4. Run the python script: **cam_receiver_code.py**
5. Now connect the ESP32 Dev Module which is connected to all other sensors (PIR, Ultrasonic, RTC, Buzzer)
6. Once connected change the Wifi Name, Password to connected Wifi
7. Finally when the System Starts detecting any violation of OVER-SPEEDING
8. A Trigger is sent via ESP32 Dev module to ESP32 CAM via WIFI to caputre the image 
These captured images are stored in "snapshots" folder with their their recorded Timestamps
