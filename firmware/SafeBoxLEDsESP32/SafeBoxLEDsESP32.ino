// SAFEBOX AI: LEDs y MG90S en ESP32-WROOM, Arduino-ESP32 3.x.
// Ningún comando de LED mueve el servo. OPEN/LOCK requieren calibración.
#include <Arduino.h>
#include <Preferences.h>
#include <esp_system.h>
#include <stdio.h>
#include <string.h>
#include "calibration.h"

constexpr uint8_t LED_BLUE_PIN = 25;
constexpr uint8_t LED_RED_PIN = 26;
constexpr uint8_t SERVO_PIN = 18;
constexpr uint32_t LED_DURATION_MS = 3000;
constexpr uint32_t LINK_TIMEOUT_MS = 15000;
constexpr uint32_t TEST_SIGNAL_MS = 600;
constexpr uint32_t SERVO_PERIOD_US = 20000;  // 50 Hz
constexpr uint32_t PWM_MAX = 65535;          // 16 bits
constexpr uint16_t TEST_MIN_US = 1000;
constexpr uint16_t TEST_MAX_US = 2000;
constexpr uint16_t TEST_FIRST_US = 1500;
constexpr uint16_t TEST_MAX_STEP_US = 50;
constexpr bool CALIBRATED = SAFEBOX_OPEN_US != 0 && SAFEBOX_LOCK_US != 0;

static_assert(SAFEBOX_DEMO_ONLY == 0 || SAFEBOX_DEMO_ONLY == 1,
              "SAFEBOX_DEMO_ONLY debe ser 0 o 1");
static_assert((SAFEBOX_OPEN_US == 0 && SAFEBOX_LOCK_US == 0) ||
              (SAFEBOX_OPEN_US >= 1000 && SAFEBOX_OPEN_US <= 2000 &&
               SAFEBOX_LOCK_US >= 1000 && SAFEBOX_LOCK_US <= 2000 &&
               SAFEBOX_OPEN_US != SAFEBOX_LOCK_US),
              "Configura ambos pulsos distintos entre 1000 y 2000 us, o ambos en 0");
static_assert(SERVO_PIN != LED_BLUE_PIN && SERVO_PIN != LED_RED_PIN &&
              LED_BLUE_PIN != LED_RED_PIN, "Pines repetidos");

char lineBuffer[64];
size_t lineLength = 0;
bool discardingLine = false;
bool servoReady = false;
bool storageReady = false;
bool ledsOn = false;
Preferences preferences;
uint16_t openPulseUs = SAFEBOX_OPEN_US;
uint16_t lockPulseUs = SAFEBOX_LOCK_US;
uint16_t lastTestPulse = 0;
uint32_t bootId = 0;
uint32_t blueOffAt = 0;
uint32_t testStopAt = 0;
uint32_t lastContactAt = 0;

void setLeds(bool blue, bool red) {
  digitalWrite(LED_BLUE_PIN, blue ? HIGH : LOW);
  digitalWrite(LED_RED_PIN, red ? HIGH : LOW);
  ledsOn = blue || red;
  if (!blue) blueOffAt = 0;
}

bool writeServoPulse(uint16_t pulseUs) {
  return servoReady && ledcWrite(SERVO_PIN,
                                (static_cast<uint32_t>(pulseUs) * PWM_MAX) / SERVO_PERIOD_US);
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

void replyStatus(unsigned long sequence) {
  Serial.print("ACK ");
  Serial.print(sequence);
  Serial.print(" SERVO_STATUS ");
  Serial.print(bootId, HEX);
  Serial.print(' ');
  Serial.println(!servoReady || !storageReady ? "ERROR" : !CALIBRATED ? "NO_CONFIGURADO" :
                 SAFEBOX_DEMO_ONLY ? "DEMO" : "LISTO");
}

void replyConfig(unsigned long sequence) {
  Serial.print("ACK ");
  Serial.print(sequence);
  Serial.print(" SERVO_CONFIG ");
  Serial.print(bootId, HEX);
  Serial.print(' ');
  Serial.print(openPulseUs);
  Serial.print(' ');
  Serial.println(lockPulseUs);
}

void handleLine(const char* line) {
  char command[16] = {};
  char extra = 0;
  unsigned long sequence = 0;
  unsigned long session = 0;
  unsigned int pulse = 0;
  unsigned int secondPulse = 0;
  const int fields = sscanf(line, "%15s %lu %lx %u %u %c",
                            command, &sequence, &session, &pulse, &secondPulse, &extra);
  if (fields < 2) return;
  if (strcmp(command, "PING") == 0) {
    if (fields == 2) lastContactAt = millis();
    reply(fields == 2 ? "ACK" : "ERR", sequence, command);
    return;
  }
  // Todos los movimientos requieren el ID del arranque actual.
  if (fields < 3 || session != bootId) {
    setLeds(false, false);
    reply("ERR", sequence, command);
    return;
  }
  if (strcmp(command, "SERVO_STATUS") == 0 && fields == 3) {
    lastContactAt = millis();
    replyStatus(sequence);
  } else if (strcmp(command, "SERVO_CONFIG") == 0 && fields == 3) {
    lastContactAt = millis();
    replyConfig(sequence);
  } else if (strcmp(command, "SET_SERVO") == 0 && fields == 5) {
    if (!SAFEBOX_DEMO_ONLY || !storageReady || !servoReady ||
        pulse < 1000 || pulse > 2000 || secondPulse < 1000 ||
        secondPulse > 2000 || pulse == secondPulse) {
      reply("ERR", sequence, command);
      return;
    }
    const uint32_t pair = (static_cast<uint32_t>(pulse) << 16) | secondPulse;
    if (preferences.putUInt("servo_pair", pair) != sizeof(pair)) {
      reply("ERR", sequence, command);
      return;
    }
    openPulseUs = pulse;
    lockPulseUs = secondPulse;
    lastContactAt = millis();
    reply("ACK", sequence, command);
  } else if (strcmp(command, "TEST_SERVO") == 0 && fields == 4) {
    const bool inRange = pulse >= TEST_MIN_US && pulse <= TEST_MAX_US;
    const bool safeStep = lastTestPulse == 0 ? pulse == TEST_FIRST_US :
        abs(static_cast<int>(pulse) - static_cast<int>(lastTestPulse)) <= TEST_MAX_STEP_US;
    if (CALIBRATED || !inRange || !safeStep || !writeServoPulse(pulse)) {
      reply("ERR", sequence, command);
      return;
    }
    lastTestPulse = pulse;
    testStopAt = millis() + TEST_SIGNAL_MS;
    lastContactAt = millis();
    reply("ACK", sequence, command);
  } else if (strcmp(command, "SERVO_STOP") == 0 && fields == 3) {
    if (!servoReady || !ledcWrite(SERVO_PIN, 0)) {
      reply("ERR", sequence, command);
      return;
    }
    testStopAt = 0;
    lastTestPulse = 0;
    lastContactAt = millis();
    reply("ACK", sequence, command);
  } else if (strcmp(command, "OPEN") == 0 && fields == 3) {
    if (!CALIBRATED || !storageReady || !writeServoPulse(openPulseUs)) {
      reply("ERR", sequence, command);
      return;
    }
    lastContactAt = millis();
    reply("ACK", sequence, command);
  } else if (strcmp(command, "LOCK") == 0 && fields == 3) {
    if (!CALIBRATED || !storageReady || !writeServoPulse(lockPulseUs)) {
      reply("ERR", sequence, command);
      return;
    }
    lastContactAt = millis();
    reply("ACK", sequence, command);
  } else if (strcmp(command, "AUTH_OK") == 0 && fields == 3) {
    lastContactAt = millis();
    setLeds(true, false);
    blueOffAt = millis() + LED_DURATION_MS;
    reply("ACK", sequence, command);
  } else if ((strcmp(command, "AUTH_FAIL") == 0 ||
              strcmp(command, "DENIED") == 0) && fields == 3) {
    lastContactAt = millis();
    setLeds(false, true);
    reply("ACK", sequence, command);
  } else if ((strcmp(command, "LED_OFF") == 0 ||
              strcmp(command, "IDLE") == 0) && fields == 3) {
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
  servoReady = ledcAttach(SERVO_PIN, 50, 16);
  if (servoReady) ledcWrite(SERVO_PIN, 0);  // Sin posición ni movimiento al arrancar.
  storageReady = preferences.begin("safebox", false);
  if (storageReady && SAFEBOX_DEMO_ONLY) {
    const uint32_t pair = preferences.getUInt("servo_pair",
        (static_cast<uint32_t>(SAFEBOX_OPEN_US) << 16) | SAFEBOX_LOCK_US);
    const uint16_t savedOpen = pair >> 16;
    const uint16_t savedLock = pair & 0xFFFF;
    if (savedOpen >= 1000 && savedOpen <= 2000 &&
        savedLock >= 1000 && savedLock <= 2000 && savedOpen != savedLock) {
      openPulseUs = savedOpen;
      lockPulseUs = savedLock;
    }
  }
  lastContactAt = millis();
}

void loop() {
  if (blueOffAt != 0 && static_cast<int32_t>(millis() - blueOffAt) >= 0) {
    setLeds(false, false);
  }
  if (ledsOn && millis() - lastContactAt >= LINK_TIMEOUT_MS) {
    setLeds(false, false);
  }
  if (testStopAt != 0 && static_cast<int32_t>(millis() - testStopAt) >= 0) {
    ledcWrite(SERVO_PIN, 0);
    testStopAt = 0;
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
