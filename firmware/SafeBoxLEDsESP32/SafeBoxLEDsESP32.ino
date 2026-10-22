// Indicador LED independiente para ESP32-WROOM (Arduino-ESP32 3.x).
// No hay control ni inicialización de servomotor en este firmware.
#include <Arduino.h>
#include <esp_system.h>
#include <stdio.h>
#include <string.h>

constexpr uint8_t LED_BLUE_PIN = 25;
constexpr uint8_t LED_RED_PIN = 26;
constexpr uint32_t LED_DURATION_MS = 3000;
constexpr uint32_t LINK_TIMEOUT_MS = 15000;

char lineBuffer[48];
size_t lineLength = 0;
bool discardingLine = false;
uint32_t bootId = 0;
uint32_t offAt = 0;
bool ledsOn = false;
uint32_t lastContactAt = 0;

void setLeds(bool blueOn, bool redOn) {
  digitalWrite(LED_BLUE_PIN, blueOn ? HIGH : LOW);
  digitalWrite(LED_RED_PIN, redOn ? HIGH : LOW);
  ledsOn = blueOn || redOn;
  if (!ledsOn) offAt = 0;
}

void reply(const char* type, unsigned long sequence, const char* command) {
  Serial.print(type);
  Serial.print(' ');
  Serial.print(sequence);
  Serial.print(' ');
  Serial.print(command);
  Serial.print(' ');
  Serial.println(bootId, HEX);
}

void handleLine(const char* line) {
  char command[16] = {};
  char extra = 0;
  unsigned long sequence = 0;
  unsigned long session = 0;
  const int fields = sscanf(line, "%15s %lu %lx %c", command, &sequence, &session, &extra);
  if (fields < 2) return;
  if (strcmp(command, "PING") == 0) {
    if (fields == 2) lastContactAt = millis();
    reply(fields == 2 ? "ACK" : "ERR", sequence, command);
    return;
  }
  if (fields != 3 || session != bootId) {
    setLeds(false, false);
    reply("ERR", sequence, command);
    return;
  }
  if (strcmp(command, "AUTH_OK") == 0) {
    lastContactAt = millis();
    setLeds(true, false);
    offAt = millis() + LED_DURATION_MS;
    reply("ACK", sequence, command);
  } else if (strcmp(command, "AUTH_FAIL") == 0) {
    lastContactAt = millis();
    setLeds(false, true);
    offAt = 0;
    reply("ACK", sequence, command);
  } else if (strcmp(command, "LED_OFF") == 0) {
    lastContactAt = millis();
    setLeds(false, false);
    reply("ACK", sequence, command);
  } else {
    setLeds(false, false);
    reply("ERR", sequence, command);
  }
}

void setup() {
  pinMode(LED_BLUE_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  setLeds(false, false);
  Serial.begin(115200);
  bootId = esp_random();
  lastContactAt = millis();
}

void loop() {
  if ((offAt != 0 && static_cast<int32_t>(millis() - offAt) >= 0) ||
      (ledsOn && millis() - lastContactAt >= LINK_TIMEOUT_MS)) {
    setLeds(false, false);
  }
  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      if (!discardingLine) {
        lineBuffer[lineLength] = '\0';
        handleLine(lineBuffer);
      }
      lineLength = 0;
      discardingLine = false;
    } else if (ch != '\r' && !discardingLine) {
      if (lineLength < sizeof(lineBuffer) - 1) {
        lineBuffer[lineLength++] = ch;
      } else {
        lineLength = 0;
        discardingLine = true;
        setLeds(false, false);
      }
    }
  }
}
