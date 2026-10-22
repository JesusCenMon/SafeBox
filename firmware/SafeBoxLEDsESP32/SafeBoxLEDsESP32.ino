// Indicador LED independiente para ESP32-WROOM (Arduino-ESP32 3.x).
// No hay control ni inicialización de servomotor en este firmware.
#include <Arduino.h>
#include <esp_system.h>
#include <stdio.h>
#include <string.h>

constexpr uint8_t LED_1_PIN = 18;
constexpr uint8_t LED_2_PIN = 19;
constexpr uint32_t LED_DURATION_MS = 3000;

char lineBuffer[48];
size_t lineLength = 0;
bool discardingLine = false;
uint32_t bootId = 0;
uint32_t offAt = 0;
bool ledsOn = false;

void setLeds(bool on) {
  digitalWrite(LED_1_PIN, on ? HIGH : LOW);
  digitalWrite(LED_2_PIN, on ? HIGH : LOW);
  ledsOn = on;
  if (!on) offAt = 0;
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
    reply(fields == 2 ? "ACK" : "ERR", sequence, command);
    return;
  }
  if (fields != 3 || session != bootId) {
    setLeds(false);
    reply("ERR", sequence, command);
    return;
  }
  if (strcmp(command, "AUTH_OK") == 0) {
    setLeds(true);
    offAt = millis() + LED_DURATION_MS;
    reply("ACK", sequence, command);
  } else if (strcmp(command, "AUTH_FAIL") == 0) {
    setLeds(false);
    reply("ACK", sequence, command);
  } else {
    setLeds(false);
    reply("ERR", sequence, command);
  }
}

void setup() {
  pinMode(LED_1_PIN, OUTPUT);
  pinMode(LED_2_PIN, OUTPUT);
  setLeds(false);
  Serial.begin(115200);
  bootId = esp_random();
}

void loop() {
  if (ledsOn && static_cast<int32_t>(millis() - offAt) >= 0) {
    setLeds(false);
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
        setLeds(false);
      }
    }
  }
}
