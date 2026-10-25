#include <WiFi.h>

const char* ssid = "27";
const char* password = "987654321";

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("==============================");
  Serial.println("ESP32 DevKit V1");
  Serial.println("Wi-Fi Connection Test");
  Serial.println("==============================");

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected!");
  Serial.print("DevKit IP address: ");
  Serial.println(WiFi.localIP());

  Serial.print("Camera IP address: ");
  Serial.println("10.136.145.144");

  Serial.println("==============================");
  Serial.println("Both devices can now be on Wi-Fi.");
  Serial.println("==============================");
}


void loop() {
  // Nothing needed for this test
}