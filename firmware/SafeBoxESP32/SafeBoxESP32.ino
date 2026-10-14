// SAFEBOX AI: WeMos D1 ESP32-WROOM, Arduino-ESP32 3.x.
// Ajusta pines y pulsos al montaje antes de conectar el MG90S a la cerradura.
#include <Arduino.h>
#include <stdio.h>
#include <string.h>

#ifndef SAFEBOX_SERVO_PIN
#define SAFEBOX_SERVO_PIN 18
#endif
#ifndef SAFEBOX_LED_BLUE_PIN
#define SAFEBOX_LED_BLUE_PIN 25
#endif
#ifndef SAFEBOX_LED_RED_PIN
#define SAFEBOX_LED_RED_PIN 26
#endif
#ifndef SAFEBOX_LOCK_US
#define SAFEBOX_LOCK_US 1100
#endif
#ifndef SAFEBOX_OPEN_US
#define SAFEBOX_OPEN_US 1900
#endif

static_assert(SAFEBOX_LOCK_US >= 1000 && SAFEBOX_LOCK_US <= 2000, "LOCK_US fuera de límites");
static_assert(SAFEBOX_OPEN_US >= 1000 && SAFEBOX_OPEN_US <= 2000, "OPEN_US fuera de límites");
static_assert(SAFEBOX_SERVO_PIN != SAFEBOX_LED_BLUE_PIN &&
              SAFEBOX_SERVO_PIN != SAFEBOX_LED_RED_PIN &&
              SAFEBOX_LED_BLUE_PIN != SAFEBOX_LED_RED_PIN, "Pines repetidos");

constexpr uint32_t SERVO_PERIOD_US = 20000;  // 50 Hz
constexpr uint32_t PWM_MAX = 65535;           // 16 bits
char lineBuffer[48];
size_t lineLength = 0;
bool servoReady = false;

uint32_t dutyForPulse(uint32_t pulseUs) {
  return (pulseUs * PWM_MAX) / SERVO_PERIOD_US;
}

void leds(bool blue, bool red) {
  digitalWrite(SAFEBOX_LED_BLUE_PIN, blue ? HIGH : LOW);
  digitalWrite(SAFEBOX_LED_RED_PIN, red ? HIGH : LOW);
}

void reply(const char* type, unsigned long sequence, const char* command) {
  Serial.print(type);
  Serial.print(' ');
  Serial.print(sequence);
  Serial.print(' ');
  Serial.println(command);
}

void handleLine(const char* line) {
  char command[12] = {};
  char extra = 0;
  unsigned long sequence = 0;
  if (sscanf(line, "%11s %lu %c", command, &sequence, &extra) != 2) {
    return;
  }
  if (strcmp(command, "PING") == 0) {
    reply("ACK", sequence, command);
  } else if (strcmp(command, "OPEN") == 0) {
    if (!servoReady || !ledcWrite(SAFEBOX_SERVO_PIN, dutyForPulse(SAFEBOX_OPEN_US))) {
      reply("ERR", sequence, command);
      return;
    }
    leds(true, false);
    reply("ACK", sequence, command);
  } else if (strcmp(command, "LOCK") == 0) {
    if (!servoReady || !ledcWrite(SAFEBOX_SERVO_PIN, dutyForPulse(SAFEBOX_LOCK_US))) {
      reply("ERR", sequence, command);
      return;
    }
    leds(false, false);
    reply("ACK", sequence, command);
  } else if (strcmp(command, "DENIED") == 0) {
    leds(false, true);
    reply("ACK", sequence, command);
  } else if (strcmp(command, "IDLE") == 0) {
    leds(false, false);
    reply("ACK", sequence, command);
  } else {
    reply("ERR", sequence, command);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(SAFEBOX_LED_BLUE_PIN, OUTPUT);
  pinMode(SAFEBOX_LED_RED_PIN, OUTPUT);
  leds(false, false);
  servoReady = ledcAttach(SAFEBOX_SERVO_PIN, 50, 16);
  if (servoReady) {
    ledcWrite(SAFEBOX_SERVO_PIN, 0);  // No mover el servo al arrancar.
  }
}

void loop() {
  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      lineBuffer[lineLength] = '\0';
      handleLine(lineBuffer);
      lineLength = 0;
    } else if (ch != '\r') {
      if (lineLength < sizeof(lineBuffer) - 1) {
        lineBuffer[lineLength++] = ch;
      } else {
        lineLength = 0;  // Descartar línea desbordada.
      }
    }
  }
}
